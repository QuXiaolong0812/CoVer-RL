import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from transformers import AutoProcessor, GenerationConfig
from transformers.generation import GenerationMixin

from evaluate import inspect_prediction, predict, score_predictions


EVENTS = [{
    "id": "E_1", "event_type": "比赛", "trigger": "比赛",
    "arguments": [{"id": "A_1_1", "role": "参赛者", "argument": "甲"}],
}]
MODEL_DIR = Path(os.environ.get(
    "COVER_RL_TEST_MODEL_DIR", Path(__file__).resolve().parents[1] / "models/Qwen3-VL-2B-Instruct"
))


def feedback(label="correct", *, missing_role=None, event_feedback=None):
    result = {
        "event_level": [{"event_id": "E_1", "status": "correct", "error_type": None, "feedback": "保留。"}],
        "argument_level": [{"event_id": "E_1", "argument_id": "A_1_1", "role": "参赛者",
                            "status": "correct", "error_type": None, "feedback": "保留。"}],
        "final_verification": label,
    }
    if missing_role:
        result["argument_level"].append({
            "event_id": "E_1", "argument_id": None, "role": missing_role,
            "status": "missing", "error_type": "missing_argument", "feedback": "补充论元。",
        })
    if event_feedback:
        result["event_level"].append({
            "event_id": None, "status": "missing", "error_type": "missing_event", "feedback": event_feedback,
        })
    return result


def row(operator, expected):
    return {
        "input": {"events": EVENTS}, "target": expected,
        "provenance": {"operator": operator, "edit": {"value": {"event_type": "写类-签署类", "trigger": "写"}}},
    }


class EvaluateVerifierTest(unittest.TestCase):
    def test_greedy_decoding_does_not_inherit_model_sampling_defaults(self):
        class Decoder(GenerationMixin):
            generation_config = GenerationConfig.from_pretrained(MODEL_DIR, local_files_only=True)

            def generate(self, input_ids, generation_config, use_model_defaults=None, **inputs):
                self.effective, _ = self._prepare_generation_config(generation_config, use_model_defaults)
                return torch.cat((input_ids, torch.tensor([[self.effective.eos_token_id]])), dim=1)

        processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        decoder = Decoder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (64, 64)).save(root / "sample.png")
            args = SimpleNamespace(dataset_dir=root, max_image_pixels=8192, max_new_tokens=1)
            generation = GenerationConfig(do_sample=False, max_new_tokens=1,
                                          eos_token_id=processor.tokenizer.eos_token_id)
            candidate = {"image_path": "sample.png", "title": "", "text": "", "description": "", "events": EVENTS}
            predict(decoder, processor, candidate, args, torch.device("cpu"), generation)
        self.assertFalse(decoder.effective.do_sample)

    def test_strict_json_and_complete_candidate_numbering(self):
        clean = feedback()
        good = inspect_prediction(json.dumps(clean), EVENTS)
        self.assertTrue(good["format_valid"])
        code_block = inspect_prediction(f"```json\n{json.dumps(clean)}\n```", EVENTS)
        self.assertFalse(code_block["json_valid"])
        repeated = inspect_prediction('{"event_level":[],"event_level":[]}', EVENTS)
        self.assertFalse(repeated["json_valid"])

        clean["argument_level"][0]["argument_id"] = "A_9_1"
        bad_id = inspect_prediction(json.dumps(clean), EVENTS)
        self.assertTrue(bad_id["json_valid"])
        self.assertFalse(bad_id["format_valid"])
        clean["argument_level"] = []
        self.assertFalse(inspect_prediction(json.dumps(clean), EVENTS)["format_valid"])

    def test_rejects_unpaired_unicode_surrogate_before_writing_predictions(self):
        clean = feedback()
        clean["event_level"][0]["feedback"] = chr(0xD800)
        output = inspect_prediction(json.dumps(clean), EVENTS)
        self.assertFalse(output["format_valid"])

    def test_invalid_outputs_count_against_metrics_and_wrong_a1_role_is_not_a_hit(self):
        positive = feedback()
        a1 = feedback("partial", missing_role="地点")
        a2 = feedback("partial", event_feedback="补充「写类-签署类」事件，触发词为「写」。")
        examples = [row("clean", positive), row("A1", a1), row("A2", a2)]
        outputs = ["invalid", json.dumps(feedback("partial", missing_role="时间")), json.dumps(a2)]
        report = score_predictions(examples, [inspect_prediction(o, EVENTS) for o in outputs])
        self.assertAlmostEqual(report["format_pass_rate"], 2 / 3)
        self.assertAlmostEqual(report["final"]["macro_f1"], 0.5)
        self.assertEqual(report["final"]["confusion"]["correct"]["invalid"], 1)
        self.assertEqual(report["error_detection"]["missing_argument"]["f1"], 1.0)
        self.assertEqual(report["localization"]["missing_argument"]["tp"], 0)
        self.assertEqual(report["localization"]["missing_argument"]["fp"], 1)
        self.assertEqual(report["localization"]["missing_argument"]["fn"], 1)
        self.assertEqual(report["localization"]["missing_event"]["tp"], 1)

    def test_a2_requires_an_explicit_trigger_not_just_its_characters_in_the_type(self):
        expected = feedback("partial", event_feedback="补充「写类-签署类」事件，触发词为「写」。")
        output = feedback("partial", event_feedback="补充「写类-签署类」事件，触发词为「签署」。")
        report = score_predictions([row("A2", expected)], [inspect_prediction(json.dumps(output), EVENTS)])
        self.assertEqual(report["error_detection"]["missing_event"]["tp"], 1)
        self.assertEqual(report["localization"]["missing_event"]["tp"], 0)
        self.assertEqual(report["localization"]["missing_event"]["fn"], 1)


if __name__ == "__main__":
    unittest.main()
