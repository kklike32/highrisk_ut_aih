from __future__ import annotations

from pathlib import Path

from google.adk.agents.llm_agent import Agent

from derm_advisor.config import Paths
from derm_advisor.vision.inference import classify_image


def classify_lesion(image_path: str) -> dict:
    """
    Classify a skin lesion image using the locally trained checkpoint.

    Input:
      image_path: path to a local image file (jpg/png/webp).

    Output:
      dict with label/confidence + probabilities.
    """
    paths = Paths.default()
    ckpt = paths.artifacts_dir / "best_model.pt"
    if not ckpt.exists():
        return {
            "status": "error",
            "message": f"Model checkpoint not found at {ckpt}. Train first (see README).",
        }

    p = Path(image_path).expanduser()
    if not p.exists():
        return {"status": "error", "message": f"Image not found: {p}"}

    r = classify_image(p, ckpt)
    return {
        "status": "success",
        "label": r.label,
        "confidence": r.confidence,
        "probabilities": r.probabilities,
    }


def safety_triage(label: str, confidence: float) -> dict:
    """
    Non-prescriptive triage guardrail.
    """
    red_flag_labels = {"mel", "melanoma", "malignant"}
    label_norm = label.lower()

    if any(k in label_norm for k in red_flag_labels) and confidence >= 0.5:
        return {
            "risk_level": "higher",
            "guidance": (
                "This result can be wrong; however, because the model sees potentially higher-risk features, "
                "please arrange an in-person dermatology evaluation soon. If it’s changing quickly, bleeding, "
                "or painful, seek urgent care."
            ),
        }

    if confidence < 0.6:
        return {
            "risk_level": "uncertain",
            "guidance": (
                "The model is not confident. Consider retaking the photo in bright, indirect light, "
                "include a ruler/coin for scale, and consult a clinician if you’re concerned."
            ),
        }

    return {
        "risk_level": "lower",
        "guidance": (
            "This appears lower-risk by the model, but this is not a diagnosis. Monitor for ABCDE changes "
            "(asymmetry, border, color, diameter, evolution) and seek care if any red flags appear."
        ),
    }


root_agent = Agent(
    model="gemini-flash-latest",
    name="derm_advisor_agent",
    description="A virtual dermatology health advisor that can classify lesion photos and provide safety-guarded guidance.",
    instruction=(
        "You are a virtual dermatology health advisor.\n"
        "- When the user gives a local filesystem path to an image, call classify_lesion with that path.\n"
        "- After classification, call safety_triage with the predicted label and confidence.\n"
        "- NEVER prescribe medications or claim a definitive diagnosis.\n"
        "- Always include safety guidance and when to seek professional care.\n"
        "- Be concise and use everyday language.\n"
        "- If the tool returns an error, explain how to fix it (usually train the model first).\n"
    ),
    tools=[classify_lesion, safety_triage],
)
