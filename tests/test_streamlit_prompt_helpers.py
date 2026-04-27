import unittest
from pathlib import Path

from apps.streamlit_inference import (
    _build_advisor_messages,
    _build_advisor_prompt,
    _build_followup_prompt,
    _condition_explanation,
    _infer_classifier_source,
)


class StreamlitPromptHelperTests(unittest.TestCase):
    def test_infer_pad_ufes_model_from_run_name(self):
        checkpoint = {
            "class_names": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
            "train_config": {"run_name": "pad_ufes20_efficientnetv2_s"},
            "model_config": {"backbone": "tf_efficientnetv2_s.in21k"},
        }

        source = _infer_classifier_source(Path("artifacts/best_model.pt"), checkpoint)

        self.assertEqual(source["name"], "PAD-UFES-20")
        self.assertEqual(source["description"], "PAD-UFES-20 smartphone-photo classifier")
        self.assertEqual(source["backbone"], "tf_efficientnetv2_s.in21k")

    def test_infer_ham_model_from_class_names(self):
        checkpoint = {
            "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
            "train_config": {"dataset_root": "data/ham10000_imagefolder"},
            "model_config": {"backbone": "convnext_tiny.fb_in22k"},
        }

        source = _infer_classifier_source(Path("ham100000_bestmodel.pt"), checkpoint)

        self.assertEqual(source["name"], "HAM10000")
        self.assertEqual(source["description"], "HAM10000 dermoscopy classifier")
        self.assertEqual(source["backbone"], "convnext_tiny.fb_in22k")

    def test_build_advisor_prompt_hides_path_and_uses_human_question(self):
        prompt = _build_advisor_prompt(
            {
                "label": "NEV",
                "confidence": 0.42,
                "model_source": {"description": "PAD-UFES-20 smartphone-photo classifier"},
            }
        )

        self.assertNotIn("Image path:", prompt)
        self.assertNotIn("make clear this is not a diagnosis", prompt)
        self.assertIn("PAD-UFES-20 smartphone-photo classifier", prompt)
        self.assertIn("NEV", prompt)
        self.assertIn("42.0%", prompt)

    def test_build_advisor_messages_separates_display_from_hidden_context(self):
        messages = _build_advisor_messages(
            {
                "label": "NEV",
                "confidence": 0.42,
                "model_source": {"description": "PAD-UFES-20 smartphone-photo classifier"},
            }
        )

        self.assertEqual(messages["display"], "What should I know about this spot?")
        self.assertIn("PAD-UFES-20 smartphone-photo classifier", messages["internal"])
        self.assertIn("42.0%", messages["internal"])
        self.assertNotIn("Image path:", messages["display"])
        self.assertNotIn("Do not mention file paths", messages["display"])

    def test_prompt_warns_low_confidence_is_uncertain(self):
        prompt = _build_advisor_prompt(
            {
                "label": "NEV",
                "confidence": 0.42,
                "model_source": {"description": "PAD-UFES-20 smartphone-photo classifier"},
            }
        )

        self.assertIn("below 60%", prompt)
        self.assertIn("uncertain", prompt)
        self.assertIn("Use the class label exactly", prompt)

    def test_condition_explanation_for_pad_nev(self):
        text = _condition_explanation("NEV")

        self.assertIn("nevus", text.lower())
        self.assertIn("mole", text.lower())
        self.assertIn("monitor", text.lower())

    def test_advisor_prompt_includes_condition_explanation(self):
        prompt = _build_advisor_prompt(
            {
                "label": "NEV",
                "confidence": 0.42,
                "model_source": {"description": "PAD-UFES-20 smartphone-photo classifier"},
            }
        )

        self.assertIn("Condition context", prompt)
        self.assertIn("nevus", prompt.lower())
        self.assertIn("mole", prompt.lower())

    def test_followup_prompt_preserves_visible_question_and_context(self):
        prompt = _build_followup_prompt(
            "What should I do next, and how urgent is it?",
            {
                "label": "MEL",
                "confidence": 0.997,
                "model_source": {"description": "PAD-UFES-20 smartphone-photo classifier"},
            },
        )

        self.assertIn("What should I do next, and how urgent is it?", prompt)
        self.assertIn("MEL", prompt)
        self.assertIn("99.7%", prompt)
        self.assertIn("Melanoma is a serious skin cancer category", prompt)
        self.assertNotIn("Image path:", prompt)


if __name__ == "__main__":
    unittest.main()
