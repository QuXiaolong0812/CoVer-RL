"""Run the single-GPU Qwen3-VL verifier LoRA pilot on a CMMEvent SPP run."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter

import peft
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from run_logging import RunLog, sha256_file
from vlm_io import compact_json, encode_image_prompt


PROJECT = Path(__file__).resolve().parent
IGNORE_INDEX = -100
MIB = 1024 ** 2
LORA_TARGETS = r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
INSTRUCTION = (
    "你是跨模态事件验证器。根据图像、标题、正文和背景信息核验候选事件。"
    "只输出一个 JSON 对象，包含 event_level、argument_level、final_verification。"
    "事件与论元的反馈记录包含对应的 event_id 或 argument_id、status、error_type 和 feedback，"
    "论元记录还包含 role。沿用候选编号，缺失项没有候选编号时填 null。"
    "final_verification 取 correct、partial 或 incorrect。"
)


def read_samples(path, split):
    with path.open(encoding="utf-8-sig") as file:
        rows = [json.loads(line) for line in file]
    if not rows or any(row.get("schema_version") != 1 or row.get("split") != split for row in rows):
        raise ValueError(f"Expected nonempty schema-version 1 SPP samples for {split}: {path}")
    ids = {row["sample_id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {split}: {path}")
    return rows


def encode_prompt(candidate, processor, dataset_dir, max_image_pixels):
    evidence = {key: candidate[key] for key in ("title", "text", "description", "events")}
    instruction = INSTRUCTION
    if any("bbox" in arg for event in candidate["events"] for arg in event.get("arguments", ())):
        instruction += "用图像核验给定的 bbox。坐标为原图像素 xyxy，bbox_image_size 为原图宽高。"
    return encode_image_prompt(processor, dataset_dir, candidate["image_path"],
                               f"{instruction}\n{compact_json(evidence)}", max_image_pixels)


def encode_sample(row, processor, dataset_dir, max_image_pixels, max_tokens):
    encoded = encode_prompt(row["input"], processor, dataset_dir, max_image_pixels)
    prefix = encoded["input_ids"]
    target = row["target"]
    if set(target) != {"event_level", "argument_level", "final_verification"} or target[
        "final_verification"
    ] not in ("correct", "partial", "incorrect"):
        raise ValueError(f"Invalid HSFP target: {row['sample_id']}")
    # Encode the completion separately so the image-expanded prompt is fully masked.
    completion = processor.tokenizer(compact_json(target), add_special_tokens=False)["input_ids"]
    completion.append(processor.tokenizer.eos_token_id)
    if prefix.size(1) + len(completion) > max_tokens:
        raise ValueError(f"Sample {row['sample_id']} exceeds --max-tokens={max_tokens}.")
    suffix = torch.tensor([completion], dtype=prefix.dtype)
    inputs = torch.cat((prefix, suffix), dim=1)
    labels = torch.cat((torch.full_like(prefix, IGNORE_INDEX), suffix), dim=1)
    return {
        "input_ids": inputs,
        "attention_mask": torch.ones_like(inputs),
        "pixel_values": encoded["pixel_values"],
        "image_grid_thw": encoded["image_grid_thw"],
        "labels": labels,
    }, len(completion)


def target_only_loss(suffix_logits, labels, target_count):
    # Position i predicts position i+1. The first completion token is predicted
    # by the final prompt token, so the forward pass keeps target_count + 1 logits.
    predictions = suffix_logits[:, :-1, :].float().flatten(0, 1)
    targets = labels[:, -target_count:].reshape(-1)
    return torch.nn.functional.cross_entropy(predictions, targets)


def sample_loss(model, batch, target_count, device):
    labels = batch["labels"].to(device)
    inputs = {name: value.to(device) for name, value in batch.items() if name != "labels"}
    outputs = model(**inputs, logits_to_keep=target_count + 1, use_cache=False)
    return target_only_loss(outputs.logits, labels, target_count)


def pick_eval_rows(rows, limit):
    # Include one example per available operator before filling the remaining slots.
    first = {}
    for row in rows:
        first.setdefault(row["provenance"]["operator"], row)
    leading = list(first.values())
    ids = {row["sample_id"] for row in leading}
    return (leading + [row for row in rows if row["sample_id"] not in ids])[:limit]


def evaluate(model, rows, processor, args, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for row in rows:
            batch, tokens = encode_sample(
                row, processor, args.dataset_dir, args.max_image_pixels, args.max_tokens
            )
            loss = sample_loss(model, batch, tokens, device).item()
            if not math.isfinite(loss):
                raise ValueError(f"Non-finite validation loss for {row['sample_id']}.")
            total_loss += loss * tokens
            total_tokens += tokens
    return total_loss / total_tokens, total_tokens


def peak_memory(device=None):
    return {
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / MIB, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / MIB, 1),
    }


def sft_targets(scope):
    if scope == "pilot":
        return LORA_TARGETS
    if scope == "full":
        from grpo import LORA_TARGETS as FULL_TARGETS

        return FULL_TARGETS
    raise ValueError(f"Unknown LoRA scope {scope!r}.")


def save_and_verify_adapter(model, processor, checkpoint):
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint)
    original = {name: weight.detach().cpu() for name, weight in peft.get_peft_model_state_dict(model).items()}
    # The fresh wrapper promotes its adapter slots to FP32 before loading.
    # Adding a named adapter to the existing BF16 model rounds the saved weights.
    base = model.unload()
    del base.peft_config  # PEFT 0.17 leaves this marker on the unwrapped model.
    reloaded = peft.PeftModel.from_pretrained(base, checkpoint, is_trainable=False, local_files_only=True)
    restored = peft.get_peft_model_state_dict(reloaded)
    if not original or original.keys() != restored.keys() or any(
        not torch.equal(weight, restored[name].detach().cpu()) for name, weight in original.items()
    ):
        raise RuntimeError("Reloaded adapter tensors differ from the saved adapter.")


def train(args, run):
    run.summary["completed_steps"] = 0
    paths = {split: args.spp_dir / f"{split}.jsonl" for split in ("train", "dev")}
    run.update_manifest(
        code={**run.manifest.get("code", {}),
              **{name: sha256_file(PROJECT / name) for name in ("train_verifier_sft.py", "run_logging.py", "vlm_io.py")}},
        inputs={path.name: sha256_file(path) for path in paths.values()},
        packages={"torch": torch.__version__, "transformers": transformers.__version__,
                  "peft": peft.__version__, "accelerate": version("accelerate")},
    )
    train_rows, dev_rows = (read_samples(paths[split], split) for split in ("train", "dev"))
    if {row["doc_id"] for row in train_rows} & {row["doc_id"] for row in dev_rows}:
        raise ValueError("Training and validation samples share document IDs.")
    eval_rows = pick_eval_rows(dev_rows, args.eval_samples)
    run.logger.info("SPP samples train=%d dev=%d selected_dev=%d", len(train_rows), len(dev_rows), len(eval_rows))

    device = torch.device(getattr(args, "device", "cuda:0"))
    if not torch.cuda.is_available() or device.type != "cuda" or device.index is None:
        raise RuntimeError("Verifier SFT requires an explicit CUDA device with BF16 support.")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Verifier SFT requires a CUDA GPU with BF16 support.")
    free, total = torch.cuda.mem_get_info(device)
    run.update_manifest(
        device={"name": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
                "free_mib": free // MIB, "total_mib": total // MIB},
        model_files={path.name: sha256_file(path) for path in sorted(args.model_dir.iterdir())
                     if path.is_file() and path.suffix in (".safetensors", ".json")},
    )
    run.logger.info("CUDA %s free=%d MiB total=%d MiB", torch.cuda.get_device_name(device), free // MIB, total // MIB)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, use_safetensors=True,
        dtype=torch.bfloat16, device_map={"": str(device)}, attn_implementation="sdpa",
    )
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = peft.get_peft_model(base, peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=2 * args.lora_rank,
        target_modules=sft_targets(getattr(args, "lora_scope", "pilot")), lora_dropout=0.0, bias="none",
    ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    run.logger.info("Trainable LoRA parameters=%d", sum(parameter.numel() for parameter in trainable))
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, foreach=False)
    model.train()

    for index in range(args.max_steps):
        if index % len(train_rows) == 0:
            rng.shuffle(train_rows)
        row = train_rows[index % len(train_rows)]
        step_start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch, tokens = encode_sample(row, processor, args.dataset_dir, args.max_image_pixels, args.max_tokens)
        loss = sample_loss(model, batch, tokens, device)
        value = loss.item()
        if not math.isfinite(value):
            raise ValueError(f"Non-finite training loss for {row['sample_id']}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True).item()
        optimizer.step()
        torch.cuda.synchronize(device)
        step = index + 1
        run.summary["completed_steps"] = step
        memory = peak_memory(device)
        run.metric(phase="train", step=step, loss=value, grad_norm=grad_norm, target_tokens=tokens,
                   learning_rate=args.learning_rate, seconds=round(perf_counter() - step_start, 3), **memory)
        run.logger.info("step=%d/%d loss=%.5f grad_norm=%.4f peak_allocated=%.1f MiB",
                        step, args.max_steps, value, grad_norm, memory["peak_allocated_mib"])

    optimizer.zero_grad(set_to_none=True)
    dev_loss, dev_tokens = evaluate(model, eval_rows, processor, args, device)
    run.metric(phase="dev", step=args.max_steps, loss=dev_loss, samples=len(eval_rows),
               target_tokens=dev_tokens, **peak_memory(device))
    run.logger.info("dev_loss=%.5f samples=%d target_tokens=%d", dev_loss, len(eval_rows), dev_tokens)
    checkpoint = run.path / "checkpoint"
    save_and_verify_adapter(model, processor, checkpoint)
    run.update_manifest(checkpoint={path.name: sha256_file(path) for path in sorted(checkpoint.iterdir())
                                    if path.is_file()})
    run.summary.update(checkpoint=str(checkpoint), adapter_reload_verified=True,
                       dev_loss=dev_loss, dev_samples=len(eval_rows), **peak_memory(device))
    run.logger.info("Saved and reloaded adapter %s", checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spp-dir", type=Path, required=True, help="SPP run containing train.jsonl and dev.jsonl.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT / "datasets" / "CMMEvent")
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models" / "Qwen3-VL-2B-Instruct")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT / "runs")
    parser.add_argument("--archives-dir", type=Path, default=PROJECT / "archives")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--max-image-pixels", type=int, default=131072)
    parser.add_argument("--max-tokens", type=int, default=1280,
                        help="Reject sequences above this token count without truncating them (default: 1280).")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(batch_size=1, dtype="bfloat16", lora_alpha=2 * args.lora_rank, lora_targets=sft_targets(args.lora_scope))
    try:
        with RunLog("verifier_sft", args.runs_dir, args.archives_dir, config) as run:
            if min(args.max_steps, args.eval_samples, args.lora_rank) < 1:
                raise ValueError("Step count, validation sample count, and LoRA rank must be positive.")
            if args.max_image_pixels < 1024 or args.max_tokens < 2 or args.learning_rate <= 0:
                raise ValueError("Invalid image, sequence, or learning-rate limit.")
            train(args, run)
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Verifier SFT failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
