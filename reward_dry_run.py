"""Replay generator and verifier artifacts and report the reward components."""

from collections import Counter
import json

from cmmevent_io import read_generator_predictions, read_jsonl, read_split
from extraction_eval import metrics
from reward_routing import local_units, route_local_rewards, structural_score, verifier_reward
from run_logging import sha256_file, write_json


LABELS = ("correct", "partial", "incorrect")


def verifier_summary(confusion):
    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        support = sum(confusion[label].values())
        per_label[label] = {"support": support, **metrics(tp, fp, support - tp)}
    supported = [stats for stats in per_label.values() if stats["support"]]
    total = sum(stats["support"] for stats in supported)
    return {
        "documents": total,
        "accuracy": sum(stats["tp"] for stats in supported) / total if total else None,
        "macro_f1": sum(stats["f1"] for stats in supported) / len(supported) if supported else None,
        "per_label": per_label,
        "confusion": {label: {key: confusion[label][key] for key in (*LABELS, "invalid")} for label in LABELS},
    }


def reward_dry_run(args, run):
    predictions, pred_hash, split = read_generator_predictions(args.predictions)
    feedback, feedback_hash = read_jsonl(args.feedback)
    docs_path = args.processed_dir / f"{split}.jsonl"
    docs, docs_hash = read_split(docs_path, split)
    docs_by_id = {doc["id"]: doc for doc in docs}
    feedback_by_id = {row["doc_id"]: row for row in feedback}
    if len(feedback_by_id) != len(feedback) or any(row["split"] != split for row in feedback):
        raise ValueError("Feedback requires unique document IDs and the same split as predictions.")
    expected = {row["doc_id"] for row in predictions}
    if not feedback_by_id.keys() <= expected or not expected <= docs_by_id.keys():
        raise ValueError("Feedback, predictions and processed documents disagree on their document IDs.")
    run.update_manifest(inputs={str(args.predictions): pred_hash, str(args.feedback): feedback_hash,
                                str(docs_path): docs_hash})
    evidence = None
    if args.nli_dir is not None:
        from evidence_scoring import EvidenceScorer

        evidence = EvidenceScorer(args.nli_dir, args.siglip_dir, args.dino_dir, args.dataset_dir,
                                  device=args.evidence_device)
        directories = (args.nli_dir, args.siglip_dir, args.dino_dir)
        run.update_manifest(scorer_files={str(path): {
            file.name: sha256_file(file) for file in sorted(path.iterdir())
            if file.is_file() and file.suffix in (".json", ".safetensors", ".bin", ".model")
        } for path in directories})
    tokenizer = None
    counts = Counter()
    confusion = {label: Counter() for label in LABELS}
    result_path = run.path / "rewards.jsonl"
    selected = [row for row in predictions if row["doc_id"] in feedback_by_id][:args.limit]
    if not selected:
        raise ValueError("No prediction rows have corresponding feedback.")
    with result_path.open("w", encoding="utf-8", newline="\n") as file:
        run.include_in_archive(result_path.name)
        for row in selected:
            doc_id = row["doc_id"]
            doc, checked = docs_by_id[doc_id], feedback_by_id[doc_id]
            valid = row["format_valid"]
            events = row["input"]["events"] if valid else None
            if checked["generator_format_valid"] is not valid or checked["candidate_events"] != events:
                raise ValueError(f"Feedback was produced for a different candidate in document {doc_id!r}.")
            score = structural_score(events, doc["event_list"]) if doc.get("reference_complete", True) else None
            parsed = checked["parsed"] if checked["format_valid"] else None
            evidence_result = evidence.score(doc, events, parsed, args.progress) if evidence and valid else None
            if not valid:
                outcome = args.failure_reward
            elif score is not None:
                outcome = score["f1"]
            elif evidence_result is not None:
                outcome = evidence_result["tcr"]
            else:
                raise ValueError(f"Document {doc_id!r} has no complete reference. Supply --nli-dir for TCR.")
            local = local_units(parsed)
            token_ids = row.get("completion_token_ids")
            alignment = {"ok": False, "reason": "legacy_prediction_without_token_ids"}
            returns = None
            if not valid:
                alignment = {"ok": False, "reason": "malformed_generator"}
            elif parsed is None:
                alignment = {"ok": False, "reason": "invalid_verifier_feedback"}
            elif token_ids:
                if tokenizer is None:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(args.generator_model_dir, local_files_only=True)
                returns, alignment = route_local_rewards(row["raw_output"], token_ids, tokenizer, local,
                                                         row["eos_token_ids"])
            ver_reward = verifier_reward(checked, score["label"], args.gamma) if score and valid else None
            if ver_reward is not None:
                label = parsed["final_verification"] if parsed is not None else "invalid"
                confusion[score["label"]][label] += 1
            record = {
                "doc_id": doc_id, "split": split, "generator_format_valid": valid, "structural": score,
                "generator_outcome": outcome, "verifier_reward": ver_reward,
                "verifier_label": parsed["final_verification"] if parsed is not None else None,
                "local_units": local, "local_sum": sum(item["reward"] for item in local),
                "local_returns": returns, "alignment": alignment, "evidence": evidence_result,
            }
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            file.flush()
            counts.update(documents=1, generator_valid=int(valid), aligned=int(alignment["ok"]),
                          outcome_sum=outcome, evidence_scored=int(evidence_result is not None))
            run.logger.info("doc=%s structure=%s outcome=%.4f hsfp=%s aligned=%s",
                            doc_id, score["label"] if score else None, outcome, record["verifier_label"], alignment["ok"])
    report = {"split": split, "predictions": len(predictions), **counts,
              "mean_generator_outcome": counts["outcome_sum"] / counts["documents"],
              "verifier_vs_gold": verifier_summary(confusion)}
    write_json(run.path / "report.json", report)
    run.include_in_archive("report.json")
    run.update_manifest(outputs={name: sha256_file(run.path / name) for name in ("report.json", result_path.name)})
    run.metric(**report)
    run.summary.update(report)
