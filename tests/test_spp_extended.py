from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


def document(doc_id, split, event_type, trigger, text, arguments, verb, boxes):
    def span(value):
        return [text.index(value), text.index(value) + len(value)]

    return {
        "id": doc_id, "split": split, "group_id": doc_id,
        "text": text, "title": "", "description": "",
        "image": {
            "path": f"image/{doc_id}.jpg", "sha256": hashlib.sha256(doc_id.encode()).hexdigest(),
            "width": 200, "height": 200,
        },
        "event_list": [{
            "event_type": event_type, "trigger": trigger, "trigger_span": span(trigger),
            "arguments": [
                {"role": role, "argument": value, "source": source,
                 "evidence_span": span(value) if source == "text" else None}
                for role, value, source in arguments
            ],
        }],
        "visual_annotations": [{
            "verb": verb, "annotation_id": f"{verb}_{doc_id}.jpg",
            "bb": {role: box for role, (_, box) in boxes.items()},
            "frames": [{role: synset for role, (synset, _) in boxes.items()}],
        }],
    }


def fixtures(split):
    return [
        document(f"{split}-roles", split, "交流类-介绍类", "介绍", "老师向学生介绍乐器。",
                 [("介绍者", "老师", "text"), ("介绍对象", "学生", "text")], "instructing",
                 {"agent": ("n10287213", [0, 0, 50, 50])}),
        document(f"{split}-plant", split, "农业类-种植类", "种植", "村民在农田种植水稻。",
                 [("种植者", "村民", "text"), ("种植地点", "农田", "text"),
                  ("种植植物", "水稻", "text")], "planting",
                 {"agent": ("n10287213", [0, 0, 50, 50])}),
        document(f"{split}-plant2", split, "农业类-种植类", "插秧", "村民在农田插秧。",
                 [("种植者", "村民", "text"), ("种植地点", "农田", "text")], "planting",
                 {"agent": ("n10287213", [0, 0, 50, 50])}),
        document(f"{split}-photo", split, "旅游类-拍照类", "拍照", "游客在景区拍照。",
                 [("拍照者", "游客", "text"), ("拍照工具", "手机", "image")], "photographing",
                 {"agent": ("n10287213", [0, 0, 60, 100]), "tool": ("n04401088", [90, 20, 120, 60])}),
        document(f"{split}-photo2", split, "旅游类-拍照类", "拍照", "旅客在街头拍照。",
                 [("拍照者", "旅客", "text"), ("拍照工具", "相机", "image")], "photographing",
                 {"agent": ("n10287213", [0, 0, 60, 100]), "tool": ("n02942699", [90, 20, 120, 60])}),
        document(f"{split}-fire", split, "救援类-灭火类", "灭火", "消防员在林区灭火。",
                 [("灭火人员", "消防员", "text")], "extinguishing",
                 {"agent": ("n10287213", [0, 0, 50, 50])}),
    ]


class ExtendedSPPTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "image").mkdir()
        self.rows = {split: fixtures(split) for split in ("train", "dev")}

    def run_cli(self, *args, limit=1):
        for split, rows in self.rows.items():
            (self.root / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
            )
            for row in rows:
                (self.root / row["image"]["path"]).write_bytes(row["id"].encode())
        (self.root / "splits.json").write_text(json.dumps({
            **{split: [row["id"] for row in rows] for split, rows in self.rows.items()},
            "test": ["held-out"],
        }), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-B", str(PROJECT / "build_spp.py"),
             "--data-dir", str(self.root), "--dataset-dir", str(self.root),
             "--runs-dir", str(self.root / "runs"), "--archives-dir", str(self.root / "archives"),
             "--train-per-operator", str(limit), "--dev-per-operator", str(limit), *args],
            cwd=self.root, capture_output=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        run, = (self.root / "runs" / "spp").iterdir()
        samples = {split: [json.loads(line) for line in (run / f"{split}.jsonl").read_text(
            encoding="utf-8").splitlines()] for split in self.rows}
        return samples, json.loads((run / "spp_audit.json").read_text(encoding="utf-8"))

    def test_extended_operators_build_one_evidence_backed_reversible_pair_each(self):
        operators = ("B1", "B2", "C1", "C2", "D1", "D2")
        samples, audit = self.run_cli("--operators", *operators)
        for split, rows in samples.items():
            clean = {row["doc_id"]: row for row in rows if row["provenance"]["operator"] == "clean"}
            self.assertEqual(len(clean), len(operators))
            self.assertEqual(len({row["sample_id"] for row in rows}), 2 * len(operators))
            self.assertEqual(set(audit["splits"][split]["operators"]), set(operators))
            for row in rows:
                operator = row["provenance"]["operator"]
                if operator == "clean":
                    continue
                provenance = row["provenance"]
                self.assertTrue(provenance["evidence"])
                restored = deepcopy(row["input"])
                edit = provenance["edit"]
                events = restored["events"]
                if operator == "B1":
                    left, right = (events[edit["event_index"]]["arguments"][i] for i in edit["argument_indices"])
                    left["argument"], right["argument"] = right["argument"], left["argument"]
                elif operator == "B2":
                    events[edit["event_index"]]["event_type"] = edit["value"]
                elif operator == "C1":
                    args = events[edit["event_index"]]["arguments"]
                    self.assertNotEqual(args[edit["argument_index"]]["bbox"], edit["value"])
                    self.assertEqual(args[edit["argument_index"]]["bbox_image_size"], [200, 200])
                    args[edit["argument_index"]]["bbox"] = edit["value"]
                elif operator == "C2":
                    self.assertNotEqual(restored["image_path"], edit["value"])
                    self.assertIn(restored["image_path"], {src["image"]["path"] for src in self.rows[split]})
                    restored["image_path"] = edit["value"]
                elif operator == "D1":
                    del events[edit["event_index"]]["arguments"][edit["argument_index"]]
                elif operator == "D2":
                    del events[edit["event_index"]]
                self.assertEqual(restored, clean[row["doc_id"]]["input"])
                errors = {unit["error_type"] for key in ("event_level", "argument_level")
                          for unit in row["target"][key] if unit["error_type"] is not None}
                self.assertEqual(errors, {dict(zip(operators, (
                    "wrong_role", "wrong_type", "ungrounded_arg", "modal_conflict",
                    "redundant_arg", "redundant_event",
                )))[operator]})
                self.assertEqual(row["target"]["final_verification"], "partial")

    def test_full_profile_audits_unsupported_visual_and_ambiguous_role_candidates(self):
        for rows in self.rows.values():
            rows[0]["text"] = "老师和学生共同介绍乐器。"
            rows[0]["event_list"][0]["trigger_span"] = [7, 9]
            for arg in rows[0]["event_list"][0]["arguments"]:
                start = rows[0]["text"].index(arg["argument"])
                arg["evidence_span"] = [start, start + len(arg["argument"])]
            rows[3]["visual_annotations"][0]["frames"].append({
                "agent": "n10287213", "tool": "n02942699",
            })
            rows[4]["visual_annotations"][0]["bb"]["agent"] = [80, 10, 140, 100]
            rows[4]["visual_annotations"][0]["bb"]["tool"] = None
            event = rows[1]["event_list"][0]
            rows[1]["description"] = rows[1]["text"]
            rows[1]["text"] = "图为种植现场。"
            event["trigger_span"] = [2, 4]
            for arg in event["arguments"]:
                arg["source"] = "description"
        _, audit = self.run_cli("--profile", "full")
        for split in self.rows:
            stats = audit["splits"][split]["operators"]
            self.assertEqual(set(stats), {"A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"})
            for operator in ("B1", "C1", "D1"):
                self.assertEqual(stats[operator]["selected"], 0)
                self.assertEqual(sum(stats[operator]["rejected"].values()), len(self.rows[split]))
            self.assertEqual(stats["D2"]["rejected"].get("no_grounded_shared_participant"), 1)

    def test_c2_does_not_reuse_same_donor_scene_within_split(self):
        samples, _ = self.run_cli("--operators", "C2", limit=3)
        for rows in samples.values():
            negatives = [row for row in rows if row["provenance"]["operator"] == "C2"]
            self.assertEqual(len(negatives), 3)
            self.assertEqual(len({row["input"]["image_path"] for row in negatives}), 3)


if __name__ == "__main__":
    unittest.main()
