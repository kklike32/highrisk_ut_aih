"""Streamlit UI for lesion classification and local-model advisor chat."""

from __future__ import annotations

import uuid
from pathlib import Path


import numpy as np
import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image

from derm_advisor.adk_runner import create_runner, run_turn
from derm_advisor.config import (
    DEFAULT_OLLAMA_API_BASE,
    Paths,
    classifier_checkpoint_path,
    local_llm_model_name,
    ollama_api_base,
)
from derm_advisor.vision.inference import load_checkpoint
from derm_advisor.vision.model import predict_proba
from derm_advisor.vision.transforms import build_eval_tfms


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env() -> None:
    root = _repo_root()
    load_dotenv(root / "agents" / "derm_advisor_agent" / ".env")
    load_dotenv(root / ".env")


def _load_checkpoint(ckpt_path: Path, device: str):
    model, ckpt, resolved_device = load_checkpoint(ckpt_path, device=device)
    class_names = ckpt["class_names"]
    return model, class_names, ckpt, resolved_device


_CONDITION_EXPLANATIONS = {
    "ACK": (
        "Actinic keratosis is a sun-damage related rough or scaly spot that can sometimes "
        "progress toward squamous cell carcinoma, so a clinician should evaluate persistent lesions."
    ),
    "AKIEC": (
        "Actinic keratosis / intraepithelial carcinoma is a sun-damage related category that can "
        "need clinical treatment or monitoring."
    ),
    "BCC": (
        "Basal cell carcinoma is a common type of skin cancer that usually grows slowly, but it "
        "still needs medical evaluation and treatment planning."
    ),
    "BKL": (
        "Benign keratosis-like lesions are usually non-cancerous growths, but new or changing spots "
        "can still deserve a professional look."
    ),
    "DF": (
        "Dermatofibroma is often a firm benign bump, commonly on the legs, but only an in-person "
        "exam can confirm that."
    ),
    "MEL": (
        "Melanoma is a serious skin cancer category, so a possible melanoma prediction should be "
        "treated as a reason to arrange prompt dermatology evaluation."
    ),
    "NEV": (
        "NEV means nevus, which is the medical word for a mole. Many nevi are benign, but it is "
        "still worth monitoring for change in size, shape, color, symptoms, or healing."
    ),
    "NV": (
        "NV means melanocytic nevus, which is the medical word for a mole. Many are benign, but "
        "changes over time are important to watch."
    ),
    "SCC": (
        "Squamous cell carcinoma is a skin cancer category that can grow or spread, so suspected "
        "SCC should be checked by a clinician."
    ),
    "SEK": (
        "Seborrheic keratosis is commonly a benign, waxy or stuck-on growth, though new, irritated, "
        "or changing spots should still be checked."
    ),
    "VASC": (
        "Vascular lesions involve blood vessels and are often benign, but sudden change, bleeding, "
        "or symptoms should be evaluated."
    ),
}


def _condition_explanation(label: str) -> str:
    """Return plain-language context for a classifier label."""
    normalized = str(label).strip().upper()
    return _CONDITION_EXPLANATIONS.get(
        normalized,
        "This label is a model category rather than a diagnosis, so it should be interpreted with caution.",
    )


def _infer_classifier_source(ckpt_path: Path, ckpt: dict) -> dict[str, str]:
    """Infer the training dataset/model family from checkpoint metadata."""
    train_config = ckpt.get("train_config", {}) or {}
    model_config = ckpt.get("model_config", {}) or {}
    class_names = [str(name) for name in ckpt.get("class_names", [])]
    metadata = " ".join(
        [
            str(ckpt_path),
            str(train_config.get("run_name", "")),
            str(train_config.get("dataset_root", "")),
            " ".join(class_names),
        ]
    ).lower()
    backbone = str(
        model_config.get("backbone")
        or model_config.get("model_name")
        or train_config.get("backbone")
        or "unknown backbone"
    )

    if "pad" in metadata or {"ACK", "BCC", "MEL", "NEV", "SCC", "SEK"}.issubset(set(class_names)):
        return {
            "name": "PAD-UFES-20",
            "description": "PAD-UFES-20 smartphone-photo classifier",
            "backbone": backbone,
        }

    ham_classes = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    if "ham10000" in metadata or ham_classes.issubset({name.lower() for name in class_names}):
        return {
            "name": "HAM10000",
            "description": "HAM10000 dermoscopy classifier",
            "backbone": backbone,
        }

    return {
        "name": "custom",
        "description": "custom skin-lesion classifier",
        "backbone": backbone,
    }


def _build_advisor_prompt(classification_context: dict) -> str:
    """Build hidden advisor context without exposing local file paths."""
    label = classification_context.get("label", "unknown")
    confidence = float(classification_context.get("confidence", 0))
    model_source = classification_context.get("model_source", {}) or {}
    model_description = model_source.get("description", "skin-lesion classifier")

    return (
        f"The uploaded image was classified as {label} with {confidence:.1%} confidence "
        f"using the {model_description}. "
        f"Condition context: {_condition_explanation(label)} "
        "Use the class label exactly as written; do not expand it into a made-up diagnosis. "
        "If confidence is below 60%, call it uncertain or low confidence rather than fairly certain. "
        "Answer the user's question naturally and explain which model was used, what the confidence means, "
        "what the condition label usually means, and what to watch for next. "
        "Sound like a genuine human: calm, concise, practical, and not robotic. "
        "Do not mention file paths. Avoid pretending this is certain; instead, recommend a dermatologist "
        "if the spot is changing, bleeding, painful, itchy, non-healing, or otherwise worrying."
    )


def _build_advisor_messages(classification_context: dict) -> dict[str, str]:
    """Return separate visible user text and hidden model context."""
    return {
        "display": "What should I know about this spot?",
        "internal": _build_advisor_prompt(classification_context),
    }


def _build_followup_prompt(prompt: str, classification_context: dict) -> str:
    """Attach classification context to follow-up questions without showing it in chat."""
    if not classification_context:
        return prompt

    return (
        f"User follow-up: {prompt} "
        f"Context from the uploaded image: {_build_advisor_prompt(classification_context)} "
        "Answer the follow-up directly and keep the same safety-focused tone."
    )


def _ensure_adk_runner():
    if st.session_state.get("adk_runner") is None:
        try:
            # Import lazily so the UI can show setup guidance instead of
            # failing during module import when ADK/LiteLLM is not available.
            from derm_advisor.adk_agent import root_agent

            st.session_state.adk_runner = create_runner(root_agent)
            st.session_state.pop("adk_runner_error", None)
        except Exception as e:
            st.session_state.pop("adk_runner", None)
            st.session_state.adk_runner_error = str(e)


def _render_local_model_help(err: str) -> None:
    st.warning(f"Could not initialize the advisor runtime ({err}).")
    st.markdown(
        "The chat agent now expects a local Ollama server with a tool-capable model."
    )
    st.code(
        "\n".join(
            [
                "pip install -r requirements.txt",
                "docker compose -f compose.local-model.yml up -d",
                "docker compose -f compose.local-model.yml "
                f"exec ollama ollama pull {local_llm_model_name()}",
            ]
        ),
        language="bash",
    )
    st.caption(
        "Configured endpoint: "
        f"`{ollama_api_base()}`. The default is `{DEFAULT_OLLAMA_API_BASE}` "
        "and can be overridden with `OLLAMA_API_BASE`."
    )


def _run_classifier(ckpt_path: Path, device: str, image_path: Path) -> dict:
    """Run the classifier on an image and return results."""
    if not ckpt_path.exists():
        return {"error": f"Checkpoint not found: {ckpt_path}"}

    model, class_names, ckpt, model_device = _load_checkpoint(ckpt_path, device)
    image_size = int(ckpt["train_config"].get("image_size", 224))
    tfm = build_eval_tfms(image_size)

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    t = tfm(image=arr)["image"]
    proba = predict_proba(model, t, device=model_device)

    top_idx = int(torch.argmax(proba).item())
    conf = float(proba[top_idx].item())

    return {
        "label": class_names[top_idx],
        "confidence": conf,
        "model_source": _infer_classifier_source(ckpt_path, ckpt),
        "all_probabilities": {
            class_names[i]: float(proba[i]) for i in range(len(class_names))
        },
    }


def _render_unified_advisor(paths: Paths, ckpt_path: Path, device: str) -> None:
    """Single unified interface with image upload and advisor chat."""

    # Initialize ADK runner
    _ensure_adk_runner()
    if err := st.session_state.get("adk_runner_error"):
        _render_local_model_help(err)
    runner = st.session_state.get("adk_runner")

    # Session state initialization
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "adk_user_id" not in st.session_state:
        st.session_state.adk_user_id = "streamlit_user"
    if "adk_session_id" not in st.session_state:
        st.session_state.adk_session_id = str(uuid.uuid4())

    # Layout: sidebar for image upload, main area for chat
    with st.sidebar:
        st.subheader("📷 Upload Skin Lesion Image")
        uploaded = st.file_uploader(
            "Upload a skin lesion photo for analysis",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload an image to get AI-powered classification and advice",
        )

        if uploaded:
            suffix = Path(uploaded.name).suffix or ".jpg"
            tmp_path = paths.artifacts_dir / f"_upload_{uuid.uuid4().hex}{suffix}"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            uploaded_bytes = uploaded.getvalue()
            tmp_path.write_bytes(uploaded_bytes)
            st.session_state["last_image_path"] = str(tmp_path)
            st.session_state["last_image_bytes"] = uploaded_bytes

            st.image(uploaded_bytes, caption="Uploaded image", use_container_width=True)

            # Run classifier
            with st.spinner("Analyzing image..."):
                results = _run_classifier(ckpt_path, device, tmp_path)

            if "error" in results:
                st.error(results["error"])
            else:
                st.success(f"**Prediction:** {results['label']}")
                st.metric("Confidence", f"{results['confidence']:.1%}")
                st.info(_condition_explanation(results["label"]))
                model_source = results.get("model_source", {})
                st.caption(
                    "Model used: "
                    f"{model_source.get('description', 'skin-lesion classifier')} "
                    f"({model_source.get('backbone', 'unknown backbone')})"
                )

                with st.expander("All class probabilities"):
                    for cls, prob in sorted(
                        results["all_probabilities"].items(), key=lambda x: -x[1]
                    ):
                        st.progress(prob, text=f"{cls}: {prob:.1%}")

                # Store results for advisor context
                st.session_state["last_classification"] = results

            st.divider()
            if st.button("🔍 Get Advisor Analysis", type="primary", use_container_width=True):
                if runner is not None:
                    classification_context = st.session_state.get("last_classification", {})
                    advisor_messages = _build_advisor_messages(classification_context)
                    prompt = advisor_messages["internal"]
                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": advisor_messages["display"],
                            "image_bytes": st.session_state.get("last_image_bytes"),
                        }
                    )
                    with st.spinner("Advisor is analyzing..."):
                        try:
                            reply = run_turn(
                                runner=runner,
                                user_id=st.session_state.adk_user_id,
                                session_id=st.session_state.adk_session_id,
                                text=prompt,
                            )
                        except Exception as e:
                            reply = f"**Error:** `{e}`"
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": reply}
                        )
                    st.rerun()
                else:
                    st.error("ADK Runner not available")

        st.divider()
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.adk_session_id = str(uuid.uuid4())
            st.session_state.pop("last_image_path", None)
            st.session_state.pop("last_classification", None)
            st.rerun()

        st.caption(f"Device: `{device}`")

    # Main chat area
    st.subheader("💬 Derm Advisor Chat")

    if runner is None:
        st.info(
            "The local-model advisor is not available yet. "
            f"Start Ollama at `{ollama_api_base()}` and make sure "
            f"`{local_llm_model_name()}` is installed."
        )
        return

    # Display chat messages
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            if image_bytes := m.get("image_bytes"):
                st.image(image_bytes, caption="Uploaded skin photo", width=320)
            st.markdown(m["content"])

    # Chat input
    if prompt := st.chat_input("Ask the dermatology advisor..."):
        classification_context = st.session_state.get("last_classification", {})
        model_prompt = _build_followup_prompt(prompt, classification_context)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.spinner("Advisor is thinking..."):
            try:
                reply = run_turn(
                    runner=runner,
                    user_id=st.session_state.adk_user_id,
                    session_id=st.session_state.adk_session_id,
                    text=model_prompt,
                )
            except Exception as e:
                reply = f"**Error:** `{e}`"
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()


def main() -> None:
    """Render the Streamlit demo."""
    _load_env()
    st.set_page_config(page_title="Derm Advisor", layout="wide")
    st.title("🩺 Derm Advisor")
    st.caption("Educational demo only. Not medical advice.")

    paths = Paths.default()
    ckpt_default = classifier_checkpoint_path()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    _render_unified_advisor(paths, ckpt_default, device)


if __name__ == "__main__":
    main()
