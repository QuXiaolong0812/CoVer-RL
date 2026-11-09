"""Generate HSFP feedback for CMMEvent generator predictions."""

from collections import Counter
import json
from time import perf_counter

import torch

from cmmevent_io import read_generator_predictions
from evaluate import LABELS, inspect_prediction, load_verifier, predict, verifier_device, verifier_manifest
from run_logging import sha256_file, write_json
from train_verifier_sft import peak_memory


def candidate_ids(events):
    event_ids, argument_ids = [], []
    for i, event in enumerate(events, 1):
        event_id = f"E_{i}"
        if event["id"] != event_id:
            raise ValueError(f"Expected candidate event ID {event_id}, got {event['id']!r}.")
        event_ids.append(event_id)
        for j, arg in enumerate(event["arguments"], 1):
            arg_id = f"A_{i}_{j}"
            if arg["id"] != arg_id:
                raise ValueError(f"Expected candidate argument ID {arg_id}, got {arg['id']!r}.")
            argument_ids.append(arg_id)
    return event_ids, argument_ids


def verification_record(row, raw=None, generation_stats=None):
    record = {"doc_id": row["doc_id"], "split": row["split"], "generator_format_valid": row["format_valid"]}
    if not row["format_valid"]:
        return {
            **record, "status": "skipped_invalid_generator", "candidate_events": None,
            "generator_format_error": row.get("format_error"), "raw_output": None,
            "json_valid": None, "format_valid": None, "format_error": None, "parsed": None,
        }
    events = row["input"]["events"]
    checked = inspect_prediction(raw, events)
    return {
        **record, "status": "valid_feedback" if checked["format_valid"] else "invalid_feedback",
        "candidate_events": events, "raw_output": raw, **checked, **(generation_stats or {}),
    }


def verify_generated(args, run):
    rows, digest, split = read_generator_predictions(args.predictions)
    selected = rows[:args.limit]
    run.update_manifest(inputs={str(args.predictions): digest}, selected_doc_ids=[row["doc_id"] for row in selected])
    verified = 0
    for row in selected:
        if row["format_valid"]:
            candidate_ids(row["input"]["events"])
            verified += 1

    if verified:
        device = verifier_device(args.device)
        run.update_manifest(**verifier_manifest(args, device))
        run.logger.info("Loading verifier %s with adapter %s for %d/%d documents",
                        args.model_dir, args.adapter_dir, verified, len(selected))
        model, processor, generation = load_verifier(args, device)
        torch.cuda.reset_peak_memory_stats(device)

    statuses, labels = Counter(), Counter()
    token_limit_hits = 0
    output_path = run.path / "feedback.jsonl"
    run.summary["feedback_rows_written"] = 0
    with output_path.open("w", encoding="utf-8", errors="backslashreplace", newline="\n") as file:
        run.include_in_archive(output_path.name)
        for row in selected:
            if row["format_valid"]:
                start = perf_counter()
                raw, stats = predict(model, processor, row["input"], args, device, generation)
                result = verification_record(row, raw, stats)
                result["seconds"] = round(perf_counter() - start, 3)
                token_limit_hits += int(stats["hit_token_limit"])
                if result["format_valid"]:
                    labels[result["parsed"]["final_verification"]] += 1
            else:
                result = verification_record(row)
            file.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            file.flush()
            run.summary["feedback_rows_written"] += 1
            statuses[result["status"]] += 1
            run.logger.info("doc=%s status=%s tokens=%s reason=%s", row["doc_id"], result["status"],
                            result.get("generated_tokens"), result["format_error"] or result.get("generator_format_error"))

    report = {
        "split": split, "source_documents": len(rows), "documents": len(selected), "verified_documents": verified,
        "valid_feedback": statuses["valid_feedback"], "invalid_feedback": statuses["invalid_feedback"],
        "skipped_invalid_generator": statuses["skipped_invalid_generator"],
        "format_pass_rate": statuses["valid_feedback"] / verified if verified else None,
        "token_limit_hits": token_limit_hits,
        "final_verification_counts": {label: labels[label] for label in LABELS},
    }
    if verified:
        report.update(peak_memory(device))
    write_json(run.path / "report.json", report)
    run.include_in_archive("report.json")
    run.update_manifest(outputs={name: sha256_file(run.path / name) for name in ("report.json", output_path.name)})
    run.metric(**report)
    run.summary.update(report)
    run.logger.info("HSFP valid=%d invalid=%d skipped=%d token_limit_hits=%d", report["valid_feedback"],
                    report["invalid_feedback"], report["skipped_invalid_generator"], token_limit_hits)
