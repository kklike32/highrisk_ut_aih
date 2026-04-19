from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import torch

from derm_advisor.config import Paths
from derm_advisor.vision.model import ModelConfig, create_model, predict_proba
from derm_advisor.vision.transforms import build_eval_tfms


def _load_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["model_config"]
    class_names = ckpt["class_names"]
    model = create_model(ModelConfig(**model_cfg))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, class_names, ckpt


def main() -> None:
    st.set_page_config(page_title="Derm Advisor (demo)", layout="centered")
    st.title("Derm Advisor — lesion classifier demo")
    st.caption("Educational demo only. Not medical advice.")

    paths = Paths.default()
    default_ckpt = paths.artifacts_dir / "best_model.pt"
    ckpt_path_str = st.text_input("Checkpoint path", value=str(default_ckpt))
    ckpt_path = Path(ckpt_path_str)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    st.write(f"Using device: `{device}`")

    uploaded = st.file_uploader("Upload a skin lesion photo", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        return

    img_bytes = uploaded.getvalue()
    tmp_path = paths.artifacts_dir / "_uploaded_image"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(img_bytes)
    st.image(img_bytes, caption="Uploaded image", use_container_width=True)

    if not ckpt_path.exists():
        st.error(f"Checkpoint not found: {ckpt_path}")
        return

    model, class_names, ckpt = _load_checkpoint(ckpt_path, device)
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
    st.subheader("Prediction")
    st.write({"label": class_names[top_idx], "confidence": conf})

    st.subheader("All class probabilities")
    st.json({class_names[i]: float(proba[i]) for i in range(len(class_names))})

    metrics_path = paths.reports_dir / "metrics.json"
    if metrics_path.exists():
        st.subheader("Last training run metrics")
        st.json(json.loads(metrics_path.read_text()).get("test", {}))


if __name__ == "__main__":
    main()

