from pathlib import Path
import tempfile
import unittest

import peft
import torch

from grpo import SampledLogProbs, load_named_adapter
from train_coevolution import clipped_loss, token_terms, validate_training_splits


class CoevolutionTest(unittest.TestCase):
    def test_categorical_kl_and_selected_log_prob(self):
        head = torch.nn.Linear(3, 4, bias=False)
        current = torch.tensor([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True)
        reference = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        tokens = torch.tensor([1, 2])
        logp, kl = token_terms(head, current, tokens, reference, 0.75)
        policy = (head(current) / 0.75).softmax(-1)
        baseline = (head(reference) / 0.75).softmax(-1)
        self.assertTrue(torch.allclose(logp, policy.log()[range(2), tokens]))
        self.assertTrue(torch.allclose(kl, (policy * (policy.log() - baseline.log())).sum(-1), atol=1e-7))
        (logp.sum() + kl.sum()).backward()
        self.assertTrue(torch.isfinite(current.grad).all())

    def test_clipping_respects_advantage_sign_and_penalizes_kl(self):
        old = torch.zeros(3)
        new = torch.tensor([1.5, 0.5, 1.1]).log()
        kl = torch.tensor([0.0, 0.0, 0.5])
        loss = clipped_loss(new, old, torch.tensor([1.0, -1.0, 1.0]), kl, 0.2, 0.01)
        self.assertTrue(torch.allclose(loss, torch.tensor([-1.2, 0.8, -1.095]).mean()))

    def test_sampler_records_selected_probabilities_including_terminal_token(self):
        recorder = SampledLogProbs(0.75)
        first = torch.tensor([[0.0, 1.0, -torch.inf, 3.0]])
        second = torch.tensor([[1.0, 0.0, -torch.inf, 1.0]])
        recorder(torch.tensor([[2]]), first)
        recorder(torch.tensor([[2, 1]]), second)
        expected = torch.stack(((first / 0.75).log_softmax(-1)[0, 1], (second / 0.75).log_softmax(-1)[0, 3]))
        torch.testing.assert_close(recorder.finish([1, 3]), expected)

    def test_categorical_kl_excludes_suppressed_visual_tokens(self):
        head = torch.nn.Identity()
        hidden = torch.tensor([[0.0, 1.0, 100.0]], requires_grad=True)
        reference = torch.tensor([[0.0, 1.0, -100.0]])
        logp, kl = token_terms(head, hidden, torch.tensor([1]), reference, 1.0, torch.tensor([2]))
        torch.testing.assert_close(logp, torch.tensor([-0.3132617]))
        torch.testing.assert_close(kl, torch.zeros(1))
        (logp.sum() + kl.sum()).backward()
        self.assertEqual(hidden.grad[0, 2].item(), 0)

    def test_categorical_kl_has_zero_gradient_when_reference_is_equal(self):
        head = torch.nn.Identity()
        hidden = torch.arange(128, dtype=torch.float32).div(31).unsqueeze(0).requires_grad_()
        _, kl = token_terms(head, hidden, torch.tensor([1]), hidden.detach(), 0.9)
        kl.sum().backward()
        self.assertEqual(kl.item(), 0)
        self.assertEqual(hidden.grad.count_nonzero().item(), 0)

    def test_named_adapters_keep_saved_fp32_values_on_a_bf16_base(self):
        source = peft.get_peft_model(torch.nn.Sequential(torch.nn.Linear(8, 8, dtype=torch.bfloat16)),
                                     peft.LoraConfig(r=2, target_modules=["0"]))
        shared = peft.get_peft_model(torch.nn.Sequential(torch.nn.Linear(8, 8, dtype=torch.bfloat16)),
                                     peft.LoraConfig(r=2, target_modules=["0"]), adapter_name="generator")
        with torch.no_grad():
            for parameter in source.parameters():
                if parameter.requires_grad:
                    parameter.fill_(0.0012313)
        expected = peft.get_peft_model_state_dict(source)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source.save_pretrained(path, safe_serialization=True)
            load_named_adapter(shared, path, "verifier", trainable=True)
            load_named_adapter(shared, path, "anchor", trainable=False)
        for name in ("verifier", "anchor"):
            actual = peft.get_peft_model_state_dict(shared, adapter_name=name)
            self.assertEqual(set(actual), set(expected))
            for key, value in actual.items():
                torch.testing.assert_close(value, expected[key], atol=0, rtol=0)

    def test_disallows_shared_images_and_preprocessing_audit_leakage(self):
        train = [{"id": "a", "group_id": "x", "image": {"sha256": "same"}}]
        dev = [{"id": "b", "group_id": "y", "image": {"sha256": "same"}}]
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.json"
            with self.assertRaisesRegex(ValueError, "image hashes"):
                validate_training_splits(train, dev, audit)
            dev[0]["image"]["sha256"] = "other"
            validate_training_splits(train, dev, audit)
            audit.write_text('{"issues":{"group_crosses_supplied_splits":[{"group_id":"x"}]}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crossing splits"):
                validate_training_splits(train, dev, audit)


if __name__ == "__main__":
    unittest.main()
