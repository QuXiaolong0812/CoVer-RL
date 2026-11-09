"""Shared Qwen3-VL prompt encoding, generation and JSON decoding."""

import json
from pathlib import Path

from PIL import Image
import torch


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<|", r"\u003c|")


def encode_image_prompt(processor, dataset_dir, image_path, text, max_image_pixels):
    relative = Path(image_path)
    resolved = (dataset_dir / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(dataset_dir):
        raise ValueError(f"Image path must stay inside the dataset directory: {relative}")
    with Image.open(resolved) as source:
        image = source.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": text},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(
        text=[prompt], images=[image], return_tensors="pt",
        images_kwargs={"min_pixels": 1024, "max_pixels": max_image_pixels},
    )


def vision_token_ids(model):
    config = model.config
    keys = ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
    return sorted({getattr(config, key) for key in keys if getattr(config, key, None) is not None})


@torch.inference_mode()
def generate_completion(model, processor, inputs, device, generation, *, include_token_ids=False, logits_processor=None):
    prompt_length = inputs["input_ids"].size(1)
    options = {"logits_processor": logits_processor} if logits_processor is not None else {}
    output = model.generate(**inputs.to(device), generation_config=generation, use_model_defaults=False, **options)
    completion = output[0, prompt_length:].tolist()
    eos = generation.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    stopped = bool(completion and completion[-1] in eos_ids)
    tokens = completion[:-1] if stopped else completion
    raw = processor.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    stats = {
        "prompt_tokens": prompt_length,
        "generated_tokens": len(completion),
        "ended_by_eos": stopped,
        "hit_token_limit": not stopped and len(completion) == generation.max_new_tokens,
    }
    if include_token_ids:
        stats["completion_token_ids"] = completion
    return raw, stats


def strict_json_loads(raw):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    # A lone escaped surrogate parses in Python but cannot be written as UTF-8.
    json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return value
