import os
import tempfile
import unittest
from pathlib import Path

import peft
import torch
from PIL import Image
from transformers import AutoProcessor

from train_verifier_sft import encode_sample, read_samples, save_and_verify_adapter, target_only_loss


PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("COVER_RL_TEST_MODEL_DIR", PROJECT / "models/Qwen3-VL-2B-Instruct"))


class VerifierSFTTest(unittest.TestCase):
    def test_masks_prompt_and_keeps_supervision_and_filename_out_of_prompt(self):
        processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "secret_class_name.png"
            Image.new("RGB", (96, 64), "green").save(root / name)
            row = {
                "sample_id": "train:sample:A2",
                "input": {"title": "比赛新闻", "text": "队员正在比赛。", "description": "比赛当天。",
                          "image_path": name, "events": [{"id": "E_1", "event_type": "比赛"}]},
                "target": {"event_level": [], "argument_level": [], "final_verification": "partial"},
                "provenance": {"operator": "SECRET_OPERATOR"},
            }
            batch, target_count = encode_sample(row, processor, root, 8192, 1024)
        labels = batch["labels"]
        inputs = batch["input_ids"]
        prefix = inputs[0, :-target_count]
        target = inputs[0, -target_count:]
        self.assertTrue((labels[0, :-target_count] == -100).all().item())
        self.assertTrue(torch.equal(labels[0, -target_count:], target))
        self.assertEqual(target[-1].item(), processor.tokenizer.eos_token_id)
        prompt_text = processor.tokenizer.decode(prefix, skip_special_tokens=True)
        self.assertIn("队员正在比赛", prompt_text)
        self.assertNotIn(name, prompt_text)
        self.assertNotIn("SECRET_OPERATOR", prompt_text)
        self.assertNotIn('"partial"', prompt_text)
        self.assertIn('"partial"', processor.tokenizer.decode(target, skip_special_tokens=True))
        self.assertEqual(batch["image_grid_thw"].shape, (1, 3))

    def test_suffix_loss_matches_causal_loss_with_masked_prompt(self):
        full = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4) / 20
        labels = torch.tensor([[-100, -100, -100, -100, -100, 1, 0, 3]])
        baseline = torch.nn.functional.cross_entropy(full[:, :-1, :].flatten(0, 1), labels[:, 1:].flatten())
        loss = target_only_loss(full[:, -4:, :], labels, 3)
        torch.testing.assert_close(loss, baseline)

    def test_split_validation_rejects_wrong_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dev.jsonl"
            path.write_text('{"schema_version":1,"sample_id":"train:1","split":"train","doc_id":"1"}\n',
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dev"):
                read_samples(path, "dev")

    def test_reloads_fp32_adapter_without_rounding_to_base_bf16(self):
        processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        base = torch.nn.Sequential(torch.nn.Linear(8, 8, dtype=torch.bfloat16))
        model = peft.get_peft_model(base, peft.LoraConfig(r=2, target_modules=["0"]))
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.fill_(0.0012313)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            save_and_verify_adapter(model, processor, checkpoint)
            self.assertTrue((checkpoint / "adapter_model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
