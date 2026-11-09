import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from extraction_eval import record_table_counts
from verify_generated import candidate_ids, verification_record


PROJECT = Path(__file__).resolve().parents[1]


def event(kind, arguments, trigger="触发"):
    return {"event_type": kind, "trigger": trigger,
            "arguments": [{"role": role, "argument": value} for role, value in arguments]}


class ExtractionEvaluationTest(unittest.TestCase):
    def test_one_gold_event_cannot_match_duplicate_predictions(self):
        gold = event("比赛", [("参与者", "甲"), ("地点", "体育馆")])
        predicted = event("比赛", [("参与者", "甲"), ("地点", "体育馆")], trigger="另一个触发词")
        self.assertEqual(record_table_counts([predicted, predicted], [gold]), (2, 2, 0))

    def test_wrong_role_or_type_and_duplicate_filler_count_as_errors(self):
        gold = [event("比赛", [("参与者", "甲"), ("参与者", "乙"), ("地点", "体育馆")])]
        predicted = [
            event("比赛", [("参与者", "甲"), ("参与者", "甲"), ("参与者", "体育馆")]),
            event("展出", [("参与者", "乙")]),
        ]
        self.assertEqual(record_table_counts(predicted, gold), (1, 3, 2))

    def test_cli_counts_invalid_as_empty_and_reports_subset_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e1 = event("比赛", [("参与者", "甲")])
            e2 = event("拍照", [("拍摄者", "乙"), ("地点", "体育馆")])
            gold = [{"id": d, "split": "dev", "event_list": es}
                    for d, es in [("one", [e1]), ("many", [e1, e2]), ("outside", [e2])]]
            preds = [{"doc_id": d, "split": "dev", "format_valid": ok, "input": {"events": es}}
                     for d, ok, es in [("one", True, [e1, e1]), ("many", False, [e1])]]
            for filename, rows in (("dev.jsonl", gold), ("predictions.jsonl", preds)):
                (root / filename).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            config = {"processed_dir": str(root), "runs_dir": str(root / "runs"),
                      "archives_dir": str(root / "archives")}
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", str(PROJECT / "main.py"), "evaluate-extraction",
                 "--config", str(root / "config.json"), "--predictions", str(root / "predictions.jsonl")],
                capture_output=True, encoding="utf-8",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run = next((root / "runs" / "extraction_eval").iterdir())
            report = json.loads((run / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["coverage"], {"evaluated_documents": 2, "split_documents": 3, "complete": False})
            self.assertEqual(report["invalid_outputs"], 1)
            self.assertEqual(report["overall"]["tp"], 1)
            self.assertEqual(report["overall"]["fp"], 1)
            self.assertEqual(report["overall"]["fn"], 3)
            self.assertAlmostEqual(report["single_event"]["f1"], 2 / 3)
            self.assertEqual(report["multi_event"]["f1"], 0.0)
            archive = next((root / "archives" / "extraction_eval").glob("*.tar.gz"))
            with tarfile.open(archive) as file:
                self.assertIn("report.json", file.getnames())
                self.assertIn("document_scores.jsonl", file.getnames())

    def test_verifier_skips_invalid_generator_and_checks_existing_ids(self):
        e = {"id": "E_1", **event("比赛", [("参与者", "甲")])}
        e["arguments"][0]["id"] = "A_1_1"
        self.assertEqual(candidate_ids([e]), (["E_1"], ["A_1_1"]))
        e["arguments"][0]["id"] = "A_2_1"
        with self.assertRaises(ValueError):
            candidate_ids([e])
        row = {"doc_id": "doc", "split": "dev", "format_valid": False, "input": {"events": []}}
        result = verification_record(row)
        self.assertEqual(result["status"], "skipped_invalid_generator")
        self.assertIsNone(result["format_valid"])
        self.assertIsNone(result["parsed"])

    def test_verifier_accepts_valid_empty_candidate_and_rejects_foreign_feedback_ids(self):
        row = {"doc_id": "doc", "split": "dev", "format_valid": True, "input": {"events": []}}
        hsfp = {"event_level": [], "argument_level": [], "final_verification": "correct"}
        self.assertEqual(verification_record(row, json.dumps(hsfp))["status"], "valid_feedback")
        hsfp["event_level"].append(
            {"event_id": "E_1", "status": "supported", "error_type": None, "feedback": "有证据。"}
        )
        self.assertEqual(verification_record(row, json.dumps(hsfp))["status"], "invalid_feedback")


if __name__ == "__main__":
    unittest.main()
