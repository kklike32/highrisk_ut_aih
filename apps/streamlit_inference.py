from __future__ import annotations

import json
import uuid
from pathlib import Path

import streamlit as st
import torch
from dotenv import load_dotenv

from derm_advisor.adk_agent import root_agent
from derm_advisor.adk_runner import create_runner, run_turn
from derm_advisor.config import Paths
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
            st.session_state.adk_runner = create_runner(root_agent)
            st.session_state.pop("adk_runner_error", None)
        except Exception as e:
            st.session_state.adk_runner_error = str(e)


def _render_classifier(paths: Paths, ckpt_path: Path, device: str) -> None:
    st.subheader("Lesion classifier")
    default_ckpt = paths.artifacts_dir / "best_model.pt"
    ckpt_path_str = st.text_input("Checkpoint path", value=str(ckpt_path or default_ckpt), key="cls_ckpt")
    ckpt_path_resolved = Path(ckpt_path_str)

    uploaded = st.file_uploader("Upload a skin lesion photo", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        return

    suffix = Path(uploaded.name).suffix or ".jpg"
    tmp_path = paths.artifacts_dir / f"_upload_{uuid.uuid4().hex}{suffix}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(uploaded.getvalue())
    st.session_state["last_image_path"] = str(tmp_path)

    st.image(uploaded.getvalue(), caption="Uploaded image", use_container_width=True)

    if not ckpt_path_resolved.exists():
        st.error(f"Checkpoint not found: {ckpt_path_resolved}")
        return

    model, class_names, ckpt = _load_checkpoint(ckpt_path_resolved, device)
    image_size = int(ckpt["train_config"].get("image_size", 224))
    tfm = build_eval_tfms(image_size)

    import numpy as np
    from PIL import Image

    img = Image.open(tmp_path).convert("RGB")
    arr = np.array(img)
    t = tfm(image=arr)["image"]
    proba = predict_proba(model, t, device=device)

    top_idx = int(torch.argmax(proba).item())
    conf = float(proba[top_idx].item())
    st.markdown("**Prediction**")
    st.json({"label": class_names[top_idx], "confidence": conf})

    with st.expander("All class probabilities"):
        st.json({class_names[i]: float(proba[i]) for i in range(len(class_names))})

    metrics_path = paths.reports_dir / "metrics.json"
    if metrics_path.exists():
        with st.expander("Last training run test metrics"):
            st.json(json.loads(metrics_path.read_text()).get("test", {}))

    st.caption(f"Image saved for the advisor tab: `{tmp_path}`")


def _render_adk_chat() -> None:
    st.subheader("Derm advisor (ADK chat)")
    st.caption(
        "Same Google ADK **Runner** stack as `adk web` / `adk run`, embedded here so you get a single browser UI."
    )

    if err := st.session_state.get("adk_runner_error"):
        st.warning(
            f"Could not initialize ADK Runner ({err}). "
            "Install deps with `pip install -r requirements.txt` and set `GOOGLE_API_KEY` in "
            "`agents/derm_advisor_agent/.env`."
        )

    _ensure_adk_runner()
    runner = st.session_state.get("adk_runner")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "adk_user_id" not in st.session_state:
        st.session_state.adk_user_id = "streamlit_user"
    if "adk_session_id" not in st.session_state:
        st.session_state.adk_session_id = str(uuid.uuid4())

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("New chat session", help="Clears messages and starts a fresh ADK session id."):
            st.session_state.chat_messages = []
            st.session_state.adk_session_id = str(uuid.uuid4())
            st.rerun()

    last_path = st.session_state.get("last_image_path")
    with c2:
        if last_path and st.button("Ask advisor to analyze last uploaded image"):
            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please analyze this skin lesion image using your tools and give practical, "
                        "safety-focused guidance. The image file path on this machine is:\n"
                        f"{last_path}"
                    ),
                }
            )
            if runner is not None:
                with st.spinner("Advisor is thinking…"):
                    try:
                        reply = run_turn(
                            runner=runner,
                            user_id=st.session_state.adk_user_id,
                            session_id=st.session_state.adk_session_id,
                            text=st.session_state.chat_messages[-1]["content"],
                        )
                    except Exception as e:
                        reply = f"**Error:** `{e}`"
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()

    if last_path:
        st.info(f"**Last uploaded image path:** `{last_path}`")

    if runner is None:
        return

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask the dermatology advisor…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.spinner("Advisor is thinking…"):
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
    _load_env()
    st.set_page_config(page_title="Derm Advisor", layout="wide")
    st.title("Derm Advisor")
    st.caption("Educational demo only. Not medical advice.")

    paths = Paths.default()
    ckpt_default = paths.artifacts_dir / "best_model.pt"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    tab_cls, tab_chat = st.tabs(["Lesion classifier", "Derm advisor (ADK chat)"])

    with tab_cls:
        st.write(f"Vision backend device: `{device}`")
        _render_classifier(paths, ckpt_default, device)

    with tab_chat:
        _render_adk_chat()


if __name__ == "__main__":
    main()
