from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
import unittest

from build_spp import deletion_options, load_inputs, select_balanced_a1
from run_logging import RunLog


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVED = {"run.log", "metrics.jsonl", "config.resolved.json", "manifest.json", "summary.json"}


def make_document(doc_id, split, ambiguous_title=False):
    text = "学生领取奖牌。摄影师拍摄颁奖过程。"
    fields = {
        "id": doc_id, "split": split, "group_id": doc_id, "text": text,
        "title": "学生领取奖牌，摄影师拍摄" if ambiguous_title else "学生领取奖牌",
        "description": "当天下午，学生领取奖牌。",
        "image": {"path": f"image/{doc_id}.jpg", "sha256": doc_id},
    }

    def arg(role, value, source="text"):
        start = fields[source].index(value)
        return {"role": role, "argument": value, "source": source,
                "evidence_span": [start, start + len(value)]}

    def event(kind, trigger, args):
        start = text.index(trigger)
        return {"event_type": kind, "trigger": trigger,
                "trigger_span": [start, start + len(trigger)], "arguments": args}

    # The title-linked primary event is deliberately second.
    fields["event_list"] = [
        event("拍照", "拍摄", [arg("拍摄者", "摄影师"), arg("对象", "颁奖过程")]),
        event("颁奖", "领取", [arg("时间", "下午", "description"), arg("获奖者", "学生")]),
    ]
    return fields


class SPPCommandTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.dataset = self.root / "CMMEvent"
        (self.dataset / "image").mkdir(parents=True)
        self.rows = {
            "train": [make_document("train1", "train"), make_document("train2", "train", True)],
            "dev": [make_document("dev1", "dev"), make_document("dev2", "dev", True)],
        }
        for rows in self.rows.values():
            for row in rows:
                (self.dataset / row["image"]["path"]).touch()

    def write_inputs(self):
        for split, rows in self.rows.items():
            content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            (self.prepared / f"{split}.jsonl").write_text(content, encoding="utf-8")
        splits = {split: [r["id"] for r in rows] for split, rows in self.rows.items()}
        (self.prepared / "splits.json").write_text(json.dumps({**splits, "test": ["heldout"]}), encoding="utf-8")

    def run_cli(self, folder, *args):
        self.write_inputs()
        result = subprocess.run(
            [sys.executable, "-B", str(PROJECT / "build_spp.py"),
             "--data-dir", str(self.prepared), "--dataset-dir", str(self.dataset),
             "--runs-dir", str(self.root / folder / "runs"),
             "--archives-dir", str(self.root / folder / "archives"),
             "--train-per-operator", "1", "--dev-per-operator", "1", *args],
            cwd=self.root, capture_output=True, encoding="utf-8",
        )
        return result

    def run_dir(self, folder):
        paths = list((self.root / folder / "runs" / "spp").iterdir())
        self.assertEqual(len(paths), 1)
        return paths[0]

    def archive(self, folder):
        paths = list((self.root / folder / "archives" / "spp").glob("*.tar.gz"))
        self.assertEqual(len(paths), 1)
        return tarfile.open(paths[0], "r:gz")

    def test_builds_paired_reversible_samples_and_archives_metadata(self):
        result = self.run_cli("first")
        self.assertEqual(result.returncode, 0, result.stderr)
        run_dir = self.run_dir("first")
        for split in ("train", "dev"):
            rows = [json.loads(line) for line in (run_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            clean = {r["doc_id"]: r["input"]["events"] for r in rows if r["provenance"]["operator"] == "clean"}
            negatives = {r["provenance"]["operator"]: r for r in rows if r["provenance"]["operator"] != "clean"}
            self.assertEqual(len(rows), 4)
            self.assertEqual(set(negatives), {"A1", "A2"})
            for operator, row in negatives.items():
                self.assertEqual(row["split"], split)
                self.assertEqual(set(row["input"]), {"title", "text", "description", "image_path", "events"})
                self.assertEqual(row["target"]["final_verification"], "partial")
                edit = row["provenance"]["edit"]
                restored = row["input"]["events"]
                if operator == "A2":
                    self.assertEqual(row["doc_id"], f"{split}1")
                    self.assertEqual(restored[0]["id"], "E_1")
                    missing = [item for item in row["target"]["event_level"] if item["error_type"] == "missing_event"]
                    self.assertEqual(len(missing), 1)
                    self.assertIsNone(missing[0]["event_id"])
                    restored.insert(edit["event_index"], edit["value"])
                else:
                    self.assertEqual(row["provenance"]["evidence"][0]["source"], "description")
                    restored[edit["event_index"]]["arguments"].insert(edit["argument_index"], edit["value"])
                for event_index, event in enumerate(restored, 1):
                    event["id"] = f"E_{event_index}"
                    for arg_index, arg in enumerate(event["arguments"], 1):
                        arg["id"] = f"A_{event_index}_{arg_index}"
                self.assertEqual(restored, clean[row["doc_id"]])
        retry = self.run_cli("second")
        self.assertEqual(retry.returncode, 0, retry.stderr)
        for name in ("train.jsonl", "dev.jsonl"):
            self.assertEqual((run_dir / name).read_bytes(), (self.run_dir("second") / name).read_bytes())
        with self.archive("first") as archive:
            self.assertEqual(set(archive.getnames()), ARCHIVED | {"spp_audit.json"})
            summary = json.load(archive.extractfile("summary.json"))
            self.assertEqual(summary["status"], "succeeded")
            manifest = json.load(archive.extractfile("manifest.json"))
            self.assertEqual(set(manifest["inputs"]), {"train.jsonl", "dev.jsonl", "splits.json"})

    def test_archives_failed_run_and_rejects_cross_split_groups(self):
        self.rows["dev"][0]["group_id"] = self.rows["train"][0]["group_id"]
        result = self.run_cli("failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.run_dir("failure") / "train.jsonl").exists())
        with self.archive("failure") as archive:
            self.assertEqual(set(archive.getnames()), ARCHIVED)
            summary = json.load(archive.extractfile("summary.json"))
            self.assertEqual(summary["status"], "failed")
            self.assertIn("group", summary["error"].lower())
            self.assertIn("Traceback", archive.extractfile("run.log").read().decode("utf-8"))

    def test_rejects_departure_role_when_text_indicates_destination(self):
        row = make_document("train1", "train")
        row["text"] = "队员跑向场边。"
        row["event_list"] = [{
            "event_type": "走", "trigger": "跑", "trigger_span": [2, 3],
            "arguments": [{"role": "出发地", "argument": "场边",
                           "source": "text", "evidence_span": [4, 6]}],
        }]
        for role in ("出发地", "出发点"):
            with self.subTest(role=role):
                row["event_list"][0]["arguments"][0]["role"] = role
                self.assertEqual(deletion_options(row)[0], [])

    def test_repeated_competing_trigger_in_title_is_still_ambiguous(self):
        row = make_document("train1", "train")
        row["title"] = "学生领取奖牌，摄影师拍摄照片、拍摄视频"
        self.assertEqual(deletion_options(row)[1], [])

    def test_a1_sampling_covers_distinct_types_roles_and_groups(self):
        pool = []
        for index in range(6):
            doc = {"id": str(index), "group_id": f"group-{index}",
                   "event_list": [{"event_type": "rare" if index == 5 else "common"}]}
            evidence = {"source": "text", "role": "rare_role" if index == 5 else "common_role"}
            pool.append((doc, [(0, 0, [evidence])]))
        selected = select_balanced_a1(pool, 2, set(), random.Random(42))
        self.assertEqual({doc["event_list"][0]["event_type"] for doc, _ in selected}, {"rare", "common"})
        self.assertEqual({option[2][0]["role"] for _, option in selected}, {"rare_role", "common_role"})

    def test_a1_background_reservation_preserves_available_body_quota(self):
        dual = {"id": "dual", "group_id": "dual", "event_list": [{"event_type": "活动"}]}
        background_only = {"id": "bg", "group_id": "bg", "event_list": [{"event_type": "生活"}]}
        background = (0, 0, [{"source": "description", "role": "时间"}])
        body = (0, 1, [{"source": "text", "role": "参与者"}])
        pool = [(dual, [background, body]), (background_only, [background])]
        for seed in range(10):
            selected = select_balanced_a1(pool, 2, set(), random.Random(seed))
            self.assertEqual({option[2][0]["source"] for _, option in selected}, {"text", "description"})

    def test_input_manifest_hashes_match_the_rows_already_read(self):
        self.write_inputs()
        path = self.prepared / "dev.jsonl"
        initial = path.read_bytes()
        original_title = self.rows["dev"][0]["title"]

        class ConcurrentReplacement:
            def update_manifest(self, **fields):
                self.manifest = fields
                updated = [json.loads(line) for line in initial.decode("utf-8").splitlines()]
                updated[0]["title"] = "另一版本"
                path.write_text("".join(json.dumps(row) + "\n" for row in updated), encoding="utf-8")

        run = ConcurrentReplacement()
        rows = load_inputs(self.prepared, run)
        self.assertEqual(rows["dev"][0]["title"], original_title)
        self.assertEqual(run.manifest["inputs"]["dev.jsonl"], hashlib.sha256(initial).hexdigest())

    def test_a2_review_can_confirm_primary_and_exclude_heuristic_match(self):
        self.write_inputs()
        review = self.work_review({
            "train": [],
            "dev": [
                {"doc_id": "dev1", "decision": "exclude", "reason": "有歧义"},
                {"doc_id": "dev2", "decision": "approve", "primary_event_index": 1, "secondary_event_index": 0,
                 "primary_evidence": {"source": "title", "text": "学生领取奖牌"}, "reason": "领取奖牌是报道主题"},
            ],
        })
        result = self.run_cli("reviewed", "--a2-review", str(review))
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = self.run_dir("reviewed")
        rows = [json.loads(line) for line in (directory / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
        a2 = [row for row in rows if row["provenance"]["operator"] == "A2"]
        self.assertEqual([row["doc_id"] for row in a2], ["dev2"])
        evidence = a2[0]["provenance"]["evidence"]
        self.assertTrue(any(e.get("kind") == "reviewed_primary" and e["text"] == "学生领取奖牌" for e in evidence))
        with self.archive("reviewed") as archive:
            manifest = json.load(archive.extractfile("manifest.json"))
            self.assertEqual(manifest["inputs"]["a2_review"], hashlib.sha256(review.read_bytes()).hexdigest())

    def test_a2_review_rejects_changed_split_input(self):
        self.write_inputs()
        review = self.work_review({"train": [], "dev": []})
        self.rows["dev"][0]["title"] += "改动"
        result = self.run_cli("stale", "--a2-review", str(review))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256", result.stderr)
        with self.archive("stale") as archive:
            self.assertEqual(json.load(archive.extractfile("summary.json"))["status"], "failed")

    def test_reviewed_distinct_events_can_share_an_explicit_participant(self):
        row = make_document("dev1", "dev")
        row["text"] = "学生领奖并拍照。"
        row["title"] = "颁奖活动"
        row["event_list"] = [
            {"event_type": kind, "trigger": trigger, "trigger_span": span, "arguments": [
                {"role": role, "argument": "学生", "source": "text", "evidence_span": [0, 2]},
            ]}
            for kind, trigger, span, role in [
                ("颁奖", "领奖", [2, 4], "获奖者"), ("拍照", "拍照", [5, 7], "拍摄者"),
            ]
        ]
        review = {"decision": "approve", "primary_event_index": 0, "secondary_event_index": 1,
                  "primary_evidence": {"source": "title", "text": "颁奖活动"}}
        self.assertEqual(deletion_options(row)[1], [])
        self.assertEqual(deletion_options(row, review)[1][0][0], 1)

    def work_review(self, records):
        path = self.root / "a2_review.json"
        hashes = {f"{split}.jsonl": hashlib.sha256((self.prepared / f"{split}.jsonl").read_bytes()).hexdigest()
                  for split in ("train", "dev")}
        path.write_text(json.dumps({"schema_version": 1, "inputs": hashes, "records": records}), encoding="utf-8")
        return path

    def test_archive_write_failure_is_recorded_in_run_directory(self):
        blocked = self.root / "archive-is-a-file"
        blocked.touch()
        run = RunLog("spp", self.root / "runs", blocked, {})
        with redirect_stderr(io.StringIO()), self.assertRaises(OSError):
            with run:
                pass
        summary = json.loads((run.path / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["exit_code"], 1)
        self.assertIn("archive_error", summary)


if __name__ == "__main__":
    unittest.main()
