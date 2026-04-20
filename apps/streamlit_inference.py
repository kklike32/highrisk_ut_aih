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
    local_llm_model_name,
    ollama_api_base,
)
from derm_advisor.vision.model import ModelConfig, create_model, predict_proba
from derm_advisor.vision.transforms import build_eval_tfms


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env() -> None:
    root = _repo_root()
    load_dotenv(root / "agents" / "derm_advisor_agent" / ".env")
    load_dotenv(root / ".env")


def _load_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["model_config"]
    class_names = ckpt["class_names"]
    model = create_model(ModelConfig(**model_cfg))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, class_names, ckpt


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

    model, class_names, ckpt = _load_checkpoint(ckpt_path, device)
    image_size = int(ckpt["train_config"].get("image_size", 224))
    tfm = build_eval_tfms(image_size)

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    t = tfm(image=arr)["image"]
    proba = predict_proba(model, t, device=device)

    top_idx = int(torch.argmax(proba).item())
    conf = float(proba[top_idx].item())

    return {
        "label": class_names[top_idx],
        "confidence": conf,
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
            tmp_path.write_bytes(uploaded.getvalue())
            st.session_state["last_image_path"] = str(tmp_path)

            st.image(uploaded.getvalue(), caption="Uploaded image", use_container_width=True)

            # Run classifier
            with st.spinner("Analyzing image..."):
                results = _run_classifier(ckpt_path, device, tmp_path)

            if "error" in results:
                st.error(results["error"])
            else:
                st.success(f"**Prediction:** {results['label']}")
                st.metric("Confidence", f"{results['confidence']:.1%}")

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
                    prompt = (
                        "Please analyze this skin lesion image and provide "
                        "practical, safety-focused guidance.\n"
                        f"Image path: {st.session_state['last_image_path']}\n"
                        f"Classifier prediction: {classification_context.get('label', 'unknown')} "
                        f"(confidence: {classification_context.get('confidence', 0):.1%})"
                    )
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
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
            st.markdown(m["content"])

    # Chat input
    if prompt := st.chat_input("Ask the dermatology advisor..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.spinner("Advisor is thinking..."):
            try:
                reply = run_turn(
                    runner=runner,
                    user_id=st.session_state.adk_user_id,
                    session_id=st.session_state.adk_session_id,
                    text=prompt,
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
    ckpt_default = paths.artifacts_dir / "best_model.pt"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    _render_unified_advisor(paths, ckpt_default, device)


if __name__ == "__main__":
    main()
