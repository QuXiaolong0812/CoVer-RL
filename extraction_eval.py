"""Score CMMEvent arguments with document-level record-table matching."""

from collections import Counter, defaultdict
import json

from cmmevent_io import read_generator_predictions, read_split
from run_logging import sha256_file, write_json


PROTOCOL = {
    "unit": "argument",
    "matching": "greedy_no_replacement_within_document_and_event_type",
    "prediction_order": "descending_filled_roles_stable",
    "gold_selection": "maximum_equal_role_slots_including_joint_empty_slots_then_gold_order",
    "argument_equality": "exact_role_and_value_with_multiplicity",
    "trigger_scored": False,
    "invalid_generator_output": "empty_prediction",
    "aggregation": "micro",
}


def records_by_type(events):
    records = defaultdict(list)
    for event in events:
        roles = defaultdict(Counter)
        for arg in event["arguments"]:
            roles[arg["role"]][arg["argument"]] += 1
        records[event["event_type"]].append(dict(roles))
    return records


def record_table_counts(predicted_events, gold_events):
    predicted, gold = records_by_type(predicted_events), records_by_type(gold_events)
    tp = 0
    for kind, predictions in predicted.items():
        remaining = list(gold.get(kind, ()))
        roles = set().union(*predictions, *remaining)
        for prediction in sorted(predictions, key=len, reverse=True):
            if not remaining:
                break
            index = max(range(len(remaining)), key=lambda i: sum(
                prediction.get(role) == remaining[i].get(role) for role in roles
            ))
            target = remaining.pop(index)
            tp += sum(sum((prediction[role] & target[role]).values()) for role in prediction.keys() & target.keys())
    predicted_total = sum(len(event["arguments"]) for event in predicted_events)
    gold_total = sum(len(event["arguments"]) for event in gold_events)
    return tp, predicted_total - tp, gold_total - tp


def metrics(tp, fp, fn):
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
    }


def evaluate_extraction(args, run):
    predictions, prediction_hash, split = read_generator_predictions(args.predictions, args.split)
    gold_path = args.processed_dir / f"{split}.jsonl"
    gold, gold_hash = read_split(gold_path, split)
    gold_by_id = {row["id"]: row["event_list"] for row in gold}
    unknown_ids = [row["doc_id"] for row in predictions if row["doc_id"] not in gold_by_id]
    if unknown_ids:
        raise ValueError(f"{len(unknown_ids)} prediction IDs missing from {split} gold, first {unknown_ids[0]!r}.")
    run.update_manifest(inputs={str(args.predictions): prediction_hash, str(gold_path): gold_hash})
    groups = {"overall": Counter(), "single_event": Counter(), "multi_event": Counter(), "no_event": Counter()}
    invalid = 0
    scores_path = run.path / "document_scores.jsonl"
    with scores_path.open("w", encoding="utf-8", newline="\n") as file:
        run.include_in_archive(scores_path.name)
        for row in predictions:
            doc_id = row["doc_id"]
            valid = row["format_valid"]
            expected = gold_by_id[doc_id]
            events = row["input"]["events"] if valid else []
            group = "multi_event" if len(expected) > 1 else "single_event" if expected else "no_event"
            tp, fp, fn = record_table_counts(events, expected)
            for name in ("overall", group):
                groups[name].update(documents=1, tp=tp, fp=fp, fn=fn)
            invalid += int(not valid)
            record = {
                "doc_id": doc_id, "split": split, "group": group, "generator_format_valid": valid,
                "gold_events": len(expected), "predicted_events": len(events), **metrics(tp, fp, fn),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "split": split, "protocol": PROTOCOL,
        "coverage": {"evaluated_documents": len(predictions), "split_documents": len(gold),
                     "complete": len(predictions) == len(gold)},
        "invalid_outputs": invalid,
        **{name: {"documents": count["documents"], **metrics(count["tp"], count["fp"], count["fn"])}
           for name, count in groups.items()},
    }
    write_json(run.path / "report.json", report)
    run.include_in_archive("report.json")
    run.update_manifest(outputs={name: sha256_file(run.path / name) for name in ("report.json", scores_path.name)})
    run.metric(**report)
    run.summary.update(report)
    run.logger.info("split=%s documents=%d/%d invalid=%d precision=%.4f recall=%.4f f1=%.4f",
                    split, len(predictions), len(gold), invalid,
                    report["overall"]["precision"], report["overall"]["recall"], report["overall"]["f1"])
