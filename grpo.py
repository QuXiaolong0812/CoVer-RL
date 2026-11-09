"""Qwen3-VL LoRA rollouts and token-normalized GRPO with categorical KL."""

from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import peft
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoProcessor, GenerationConfig, LogitsProcessor, LogitsProcessorList
from transformers import Qwen3VLForConditionalGeneration

from vlm_io import generate_completion, vision_token_ids


LORA_TARGETS = (
    r"model\.(language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))|visual\.(merger|deepstack_merger_list\.\d+)\.linear_fc[12])"
)


def token_terms(head, hidden, token_ids, reference_hidden, temperature, suppressed=None):
    def distribution(states):
        logits = head(states).float() / temperature
        if suppressed is not None:
            logits = logits.index_fill(-1, suppressed, torch.finfo(logits.dtype).min)
        return logits.log_softmax(-1)

    logp = distribution(hidden)
    with torch.no_grad():
        ref_logp = distribution(reference_hidden)
    selected = logp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    # Adding the zero-sum q - p centers the derivative and removes numerical
    # drift when the policy and reference distributions are identical.
    probability = logp.exp()
    kl = (probability * (logp - ref_logp) + (ref_logp.exp() - probability)).sum(-1)
    return selected, kl.clamp_min(0.0)


def clipped_loss(new_logp, old_logp, advantage, kl, epsilon, beta):
    ratio = (new_logp - old_logp).exp()
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    return (beta * kl - objective).mean()


@dataclass
class Rollout:
    inputs: object
    raw: str
    stats: dict
    advantages: list = None
    old_logp: torch.Tensor = None


class SampledLogProbs(LogitsProcessor):
    """Apply rollout temperature and retain the sampled log-probability per token."""

    def __init__(self, temperature):
        self.temperature = temperature
        self.previous = None
        self.chosen = []

    def __call__(self, input_ids, scores):
        if self.previous is not None:
            self.chosen.append(self.previous.gather(0, input_ids[0, -1:]))
        scaled = scores / self.temperature
        self.previous = scaled.log_softmax(-1)[0]
        return scaled

    def finish(self, tokens):
        if self.previous is None or len(tokens) != len(self.chosen) + 1:
            raise RuntimeError("Rollout sampling did not record one probability per token.")
        selected_last = self.previous[tokens[-1]:tokens[-1] + 1]
        return torch.cat([*self.chosen, selected_last]).cpu()


def load_backbone(path, device):
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("GRPO devices must name CUDA indices, for example cuda:0.")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError(f"GRPO needs BF16 support on {device}.")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True, use_safetensors=True,
        device_map={"": str(device)}, attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, processor


def load_named_adapter(model, path, name, *, trainable):
    config = peft.LoraConfig.from_pretrained(path, local_files_only=True)
    config.inference_mode = not trainable
    model.add_adapter(name, config)
    # PEFT 0.17.1 casts an added adapter after copying the checkpoint. Promote the
    # empty slots first so loading saved FP32 weights never rounds them to BF16.
    model.base_model._cast_adapter_dtype(name)
    result = model.load_adapter(path, adapter_name=name, is_trainable=trainable,
                                torch_device="cpu", local_files_only=True)
    missing = [key for key in result.missing_keys if f".{name}." in key or key.endswith(f".{name}")]
    if missing or result.unexpected_keys:
        raise ValueError(f"Incompatible LoRA {path}: {len(missing)} missing, "
                         f"{len(result.unexpected_keys)} unexpected keys.")


class Policy:
    def __init__(self, model, processor, name, device, max_new_tokens, temperature, lr, reference=None):
        self.model = model
        self.processor = processor
        self.name = name
        self.device = torch.device(device)
        self.temperature = temperature
        self.reference = reference
        self.model.set_adapter(name)
        self.parameters = [param for param in self.model.parameters() if param.requires_grad]
        if not self.parameters:
            raise ValueError(f"No trainable parameters in adapter {name}.")
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=0.0)
        suppressed = vision_token_ids(self.base)
        self.suppressed = torch.tensor(suppressed, dtype=torch.long, device=self.device)
        self.generation = GenerationConfig(
            do_sample=True, max_new_tokens=max_new_tokens,
            top_k=0, top_p=1.0, use_cache=True,
            eos_token_id=self.base.generation_config.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            suppress_tokens=suppressed,
        )
        self.greedy = GenerationConfig(
            do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            eos_token_id=self.generation.eos_token_id, pad_token_id=self.generation.pad_token_id,
            suppress_tokens=suppressed,
        )

    @property
    def base(self):
        return self.model.get_base_model()

    def sample(self, inputs, *, greedy=False):
        self.model.set_adapter(self.name)
        self.model.eval()
        generation = self.greedy if greedy else self.generation
        recorder = None if greedy else SampledLogProbs(self.temperature)
        # BatchFeature.to mutates its container. Keep retained rollout prompts on
        # CPU so storing more groups does not consume additional GPU memory.
        raw, stats = generate_completion(self.model, self.processor, copy(inputs), self.device, generation,
                                         include_token_ids=True,
                                         logits_processor=LogitsProcessorList([recorder]) if recorder else None)
        token_ids = stats["completion_token_ids"]
        if not token_ids:
            raise RuntimeError(f"{self.name} returned no completion tokens.")
        return Rollout(inputs, raw, stats, old_logp=recorder.finish(token_ids) if recorder else None)

    def _hidden(self, inputs, tokens):
        prompt = inputs["input_ids"]
        prompt_size = prompt.shape[1]
        combined = torch.cat((prompt, tokens[None, :]), dim=1)
        # The completion starts at the distribution predicted by the last prompt token.
        features = {key: value for key, value in inputs.items()
                    if key in ("pixel_values", "image_grid_thw")}
        output = self.base.model(input_ids=combined, attention_mask=torch.ones_like(combined),
                                 use_cache=False, **features)
        return output.last_hidden_state[0, prompt_size - 1:-1]

    def update(self, rollouts, *, epochs, epsilon, beta, token_chunk, grad_clip):
        if not rollouts:
            raise ValueError("A policy update requires at least one rollout.")
        totals = {"loss": 0.0, "kl": 0.0}
        for _ in range(epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for rollout in rollouts:
                inputs = {key: value.to(self.device) for key, value in rollout.inputs.items()}
                tokens = torch.tensor(rollout.stats["completion_token_ids"], device=self.device)
                advantage = torch.as_tensor(rollout.advantages, dtype=torch.float32, device=self.device)
                self.model.eval()
                self.model.set_adapter(self.reference or self.name)
                frozen = nullcontext() if self.reference else self.model.disable_adapter()
                with torch.no_grad(), frozen:
                    reference_hidden = self._hidden(inputs, tokens)

                self.model.set_adapter(self.name)
                self.model.train()
                hidden = self._hidden(inputs, tokens)
                head = self.base.lm_head

                def compute(chunk, token_ids, reference):
                    return token_terms(head, chunk, token_ids, reference, self.temperature, self.suppressed)

                results = []
                for start in range(0, len(tokens), token_chunk):
                    span = slice(start, start + token_chunk)
                    results.append(checkpoint(compute, hidden[span], tokens[span], reference_hidden[span],
                                              use_reentrant=False))
                logp = torch.cat([item[0] for item in results])
                kl = torch.cat([item[1] for item in results])
                loss = clipped_loss(logp, rollout.old_logp.to(self.device), advantage, kl, epsilon, beta)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite GRPO loss in {self.name}.")
                (loss / len(rollouts)).backward()
                totals["loss"] += loss.item()
                totals["kl"] += kl.detach().mean().item()
                del hidden, reference_hidden, kl, logp, results, loss
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        denominator = len(rollouts) * epochs
        return {**{key: value / denominator for key, value in totals.items()}, "grad_norm": float(norm)}


def load_policies(args, resumed: Path = None):
    generator_base, generator_processor = load_backbone(args.generator_model_dir, args.generator_device)
    if resumed:
        generator_model = peft.PeftModel.from_pretrained(
            generator_base, resumed / "generator", adapter_name="generator", is_trainable=True,
            local_files_only=True,
        )
    else:
        generator_model = peft.get_peft_model(
            generator_base, peft.LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                                            target_modules=LORA_TARGETS, lora_dropout=0.0, bias="none"),
            adapter_name="generator",
        )
    if (args.generator_model_dir, args.generator_device) == (args.verifier_model_dir, args.verifier_device):
        verifier_model, verifier_processor = generator_model, generator_processor
        load_named_adapter(verifier_model, resumed / "verifier" if resumed else args.adapter_dir,
                           "verifier", trainable=True)
    else:
        verifier_base, verifier_processor = load_backbone(args.verifier_model_dir, args.verifier_device)
        verifier_model = peft.PeftModel.from_pretrained(
            verifier_base, resumed / "verifier" if resumed else args.adapter_dir,
            adapter_name="verifier", is_trainable=True, local_files_only=True,
        )
    # PEFT selects adapter keys by substring during saving. Keep the reference
    # name disjoint from both trainable names.
    load_named_adapter(verifier_model, args.adapter_dir, "anchor", trainable=False)
    generator = Policy(generator_model, generator_processor, "generator", args.generator_device,
                       args.generator_max_new_tokens, args.temperature, args.generator_lr)
    verifier = Policy(verifier_model, verifier_processor, "verifier", args.verifier_device,
                      args.verifier_max_new_tokens, args.temperature, args.verifier_lr, reference="anchor")
    return generator, verifier
