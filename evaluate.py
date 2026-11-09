"""Compare base and LoRA Qwen3-VL verifier generations on CMMEvent SPP dev."""

from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import peft
import torch
import transformers
from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

from run_logging import RunLog, sha256_file, write_json
from train_verifier_sft import MIB, PROJECT, encode_prompt, peak_memory, pick_eval_rows, read_samples
from vlm_io import generate_completion, strict_json_loads


LABELS = ("correct", "partial", "incorrect")
EVENT_ERRORS = {"missing_event", "wrong_type", "modal_conflict", "redundant_event"}
ARGUMENT_ERRORS = {"missing_argument", "wrong_role", "ungrounded_arg", "modal_conflict", "redundant_arg"}
OMISSIONS = ("missing_argument", "missing_event")
FEEDBACK_FIELDS = {"event_id", "status", "error_type", "feedback"}


def check_record(record, fields, errors):
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Unexpected feedback record fields.")
    strings = ("status", "feedback", "role") if "role" in fields else ("status", "feedback")
    if any(not isinstance(record[key], str) or not record[key].strip() for key in strings):
        raise ValueError("Feedback record has a missing or empty string field.")
    error = record["error_type"]
    if error is not None and (not isinstance(error, str) or error not in errors):
        raise ValueError("Invalid error_type for this feedback level.")


def validate_hsfp(value, events):
    if not isinstance(value, dict) or set(value) != {"event_level", "argument_level", "final_verification"}:
        raise ValueError("Expected exactly event_level, argument_level and final_verification.")
    if value["final_verification"] not in LABELS:
        raise ValueError("Invalid final_verification.")
    if not all(isinstance(value[key], list) for key in ("event_level", "argument_level")):
        raise ValueError("Feedback levels must be lists.")

    event_ids = {event["id"] for event in events}
    argument_ids = {(event["id"], arg["id"]) for event in events for arg in event["arguments"]}
    seen_events, seen_arguments, seen_missing = set(), set(), set()
    for record in value["event_level"]:
        check_record(record, FEEDBACK_FIELDS, EVENT_ERRORS)
        event_id = record["event_id"]
        if event_id is None:
            if record["status"] != "missing" or record["error_type"] != "missing_event":
                raise ValueError("A null event_id is reserved for a missing event.")
            continue
        if not isinstance(event_id, str) or event_id not in event_ids or event_id in seen_events:
            raise ValueError("Invalid or repeated candidate event_id.")
        if record["status"] == "missing" or record["error_type"] == "missing_event":
            raise ValueError("Missing-event feedback must use a null event_id.")
        seen_events.add(event_id)

    for record in value["argument_level"]:
        check_record(record, FEEDBACK_FIELDS | {"argument_id", "role"}, ARGUMENT_ERRORS)
        event_id, argument_id = record["event_id"], record["argument_id"]
        if not isinstance(event_id, str) or event_id not in event_ids:
            raise ValueError("Invalid parent event_id in argument feedback.")
        if argument_id is None:
            key = (event_id, record["role"])
            if record["status"] != "missing" or record["error_type"] != "missing_argument" or key in seen_missing:
                raise ValueError("Invalid or repeated missing-argument feedback.")
            seen_missing.add(key)
            continue
        key = (event_id, argument_id)
        if not isinstance(argument_id, str) or key not in argument_ids or key in seen_arguments:
            raise ValueError("Invalid or repeated candidate argument_id.")
        if record["status"] == "missing" or record["error_type"] == "missing_argument":
            raise ValueError("Missing-argument feedback must use a null argument_id.")
        seen_arguments.add(key)
    if seen_events != event_ids or seen_arguments != argument_ids:
        raise ValueError("Feedback does not cover every candidate event and argument.")


def inspect_prediction(raw, events):
    result = {"json_valid": False, "format_valid": False, "format_error": None, "parsed": None}
    try:
        value = strict_json_loads(raw)
        result["json_valid"] = True
        validate_hsfp(value, events)
    except (ValueError, RecursionError) as exc:
        result["format_error"] = str(exc)
        return result
    return {**result, "format_valid": True, "parsed": value}


def counts(tp, fp, fn):
    return {
        "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
    }


def tagged_errors(feedback):
    return {record["error_type"] for level in ("event_level", "argument_level")
            for record in feedback.get(level, []) if record["error_type"] is not None}


def missing_arguments(feedback):
    return {(record["event_id"], record["role"]) for record in feedback.get("argument_level", [])
            if record["error_type"] == "missing_argument"}


def matches_event_target(feedback, target):
    # The pilot represents the missing event's type and trigger in free text.
    # Require an explicitly labeled trigger so one-character substrings don't count.
    trigger = re.escape(target["trigger"])
    name = re.escape(target["event_type"])
    return bool(
        re.search(rf"{name}(?=$|事件|[」”\"'》\s，。；、,.;：:])", feedback)
        and re.search(rf"触发词\s*(?:为|是)?\s*[:：]?\s*(?:「{trigger}」|“{trigger}”|\"{trigger}\"|{trigger}(?=$|[\s，。；、,.;]))",
                      feedback)
    )


def score_predictions(rows, outputs):
    if not rows or len(rows) != len(outputs):
        raise ValueError("Expected one prediction per evaluation sample.")
    confusion = {label: Counter() for label in LABELS}
    detection = {tag: Counter() for tag in sorted(EVENT_ERRORS | ARGUMENT_ERRORS)}
    localization = {tag: Counter() for tag in OMISSIONS}
    for row, output in zip(rows, outputs):
        expected = row["target"]
        predicted = output["parsed"] or {}
        confusion[expected["final_verification"]][predicted.get("final_verification", "invalid")] += 1
        expected_tags, predicted_tags = tagged_errors(expected), tagged_errors(predicted)
        for tag in expected_tags | predicted_tags:
            detection[tag].update(tp=int(tag in expected_tags and tag in predicted_tags),
                                  fp=int(tag in predicted_tags and tag not in expected_tags),
                                  fn=int(tag in expected_tags and tag not in predicted_tags))
        expected_args, predicted_args = missing_arguments(expected), missing_arguments(predicted)
        localization["missing_argument"].update(tp=len(expected_args & predicted_args),
                                                 fp=len(predicted_args - expected_args),
                                                 fn=len(expected_args - predicted_args))
        found_events = [record for record in predicted.get("event_level", []) if record["error_type"] == "missing_event"]
        has_gold_event = "missing_event" in expected_tags
        hit = has_gold_event and any(matches_event_target(record["feedback"], row["provenance"]["edit"]["value"])
                                    for record in found_events)
        localization["missing_event"].update(tp=int(hit), fp=len(found_events) - int(hit),
                                              fn=int(has_gold_event and not hit))

    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(row[label] for other, row in confusion.items() if other != label)
        fn = sum(confusion[label].values()) - tp
        per_label[label] = counts(tp, fp, fn)
    supported = [label for label, stats in per_label.items() if stats["support"]]
    size = len(rows)
    return {
        "samples": size,
        "json_parse_rate": sum(output["json_valid"] for output in outputs) / size,
        "format_pass_rate": sum(output["format_valid"] for output in outputs) / size,
        "final": {
            "accuracy": sum(confusion[label][label] for label in LABELS) / size,
            "macro_f1": sum(per_label[label]["f1"] for label in supported) / len(supported),
            "macro_labels": supported, "per_label": per_label,
            "confusion": {label: {key: row[key] for key in (*LABELS, "invalid")}
                          for label, row in confusion.items()},
        },
        "error_detection": {tag: counts(c["tp"], c["fp"], c["fn"]) for tag, c in detection.items()},
        "localization": {tag: counts(c["tp"], c["fp"], c["fn"]) for tag, c in localization.items()},
    }


def predict(model, processor, candidate, args, device, generation):
    inputs = encode_prompt(candidate, processor, args.dataset_dir, args.max_image_pixels)
    return generate_completion(model, processor, inputs, device, generation)


def evaluate_variant(name, model, processor, rows, args, device, generation, file, run):
    outputs = []
    for index, row in enumerate(rows, 1):
        start = perf_counter()
        candidate = row["input"]
        raw, generation_stats = predict(model, processor, candidate, args, device, generation)
        checked = inspect_prediction(raw, candidate["events"])
        outputs.append({**checked, **generation_stats})
        record = {
            "model": name, "sample_id": row["sample_id"], "doc_id": row["doc_id"],
            "operator": row["provenance"]["operator"], "target": row["target"], "raw_output": raw,
            **checked, **generation_stats, "seconds": round(perf_counter() - start, 3),
        }
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        run.summary["predictions_written"] += 1
        run.logger.info("%s %d/%d format=%s output_tokens=%d seconds=%.2f",
                        name, index, len(rows), checked["format_valid"], generation_stats["generated_tokens"],
                        record["seconds"])
    report = score_predictions(rows, outputs)
    report["token_limit_hits"] = sum(output["hit_token_limit"] for output in outputs)
    report.update(peak_memory(device))
    run.metric(model=name, **report)
    run.logger.info("%s format_pass_rate=%.4f final_macro_f1=%.4f",
                    name, report["format_pass_rate"], report["final"]["macro_f1"])
    return report


def verifier_device(name="cuda:0"):
    device = torch.device(name)
    if not torch.cuda.is_available() or device.type != "cuda" or device.index is None:
        raise RuntimeError("Verifier inference requires a CUDA GPU with BF16 support.")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError(f"Verifier inference requires BF16 support on {device}.")
    return device


def verifier_manifest(args, device):
    return dict(
        packages={"torch": torch.__version__, "transformers": transformers.__version__,
                  "peft": peft.__version__, "accelerate": version("accelerate")},
        device={"name": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
                "total_mib": torch.cuda.get_device_properties(device).total_memory // MIB},
        model_files={path.name: sha256_file(path) for path in sorted(args.model_dir.iterdir())
                     if path.is_file() and path.suffix in (".json", ".safetensors")},
        adapter_files={name: sha256_file(args.adapter_dir / name)
                       for name in ("adapter_config.json", "adapter_model.safetensors")},
    )


def load_verifier(args, device):
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, use_safetensors=True, dtype=torch.bfloat16,
        device_map={"": str(device)}, attn_implementation="sdpa",
    )
    generation = GenerationConfig(
        do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True,
        eos_token_id=base.generation_config.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False, local_files_only=True).eval()
    return model, processor, generation


def evaluate_verifier(args, run):
    path = args.spp_dir / "dev.jsonl"
    rows = read_samples(path, "dev")
    if args.limit is not None:
        rows = pick_eval_rows(rows, args.limit)
    device = verifier_device(args.device)
    run.update_manifest(
        inputs={path.name: sha256_file(path)},
        code={name: sha256_file(PROJECT / name) for name in (
            "evaluate.py", "train_verifier_sft.py", "run_logging.py", "vlm_io.py"
        )},
        **verifier_manifest(args, device),
    )
    run.summary["predictions_written"] = 0
    run.logger.info("Evaluating %d dev samples on %s", len(rows), torch.cuda.get_device_name(device))
    model, processor, generation = load_verifier(args, device)
    predictions = run.path / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8", errors="backslashreplace", newline="\n") as file:
        run.include_in_archive(predictions.name)
        torch.cuda.reset_peak_memory_stats(device)
        # Disable the adapter for the baseline so both variants share the same base weights.
        with model.disable_adapter():
            baseline = evaluate_variant("base", model, processor, rows, args, device, generation, file, run)
        torch.cuda.reset_peak_memory_stats(device)
        adapted = evaluate_variant("lora", model, processor, rows, args, device, generation, file, run)
    report = {
        "split": "dev", "operator_support": dict(Counter(row["provenance"]["operator"] for row in rows)),
        "base": baseline, "lora": adapted,
        "delta_lora_minus_base": {
            "format_pass_rate": adapted["format_pass_rate"] - baseline["format_pass_rate"],
            "final_macro_f1": adapted["final"]["macro_f1"] - baseline["final"]["macro_f1"],
        },
    }
    write_json(run.path / "report.json", report)
    run.include_in_archive("report.json")
    run.update_manifest(outputs={name: sha256_file(run.path / name) for name in ("report.json", predictions.name)})
    run.summary.update(samples=len(rows), report=str(run.path / "report.json"), **report["delta_lora_minus_base"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["verifier"], default="verifier")
    parser.add_argument("--spp-dir", type=Path, required=True, help="SPP run containing dev.jsonl.")
    parser.add_argument("--adapter-dir", type=Path, required=True, help="SFT checkpoint containing the LoRA adapter.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT / "datasets" / "CMMEvent")
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models" / "Qwen3-VL-2B-Instruct")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT / "runs")
    parser.add_argument("--archives-dir", type=Path, default=PROJECT / "archives")
    parser.add_argument("--max-image-pixels", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="Optional smoke-test limit, including one per operator when possible.")
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(dtype="bfloat16", decoding="greedy", split="dev")
    try:
        with RunLog("verifier_eval", args.runs_dir, args.archives_dir, config) as run:
            if args.max_image_pixels < 1024 or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
                raise ValueError("Invalid image, generation, or sample limit.")
            evaluate_verifier(args, run)
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Verifier evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
