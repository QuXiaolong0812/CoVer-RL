import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "data_preprocess.py"
SPLITS = ("train", "dev", "test")


class CMMEventPreprocessTest(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.work = Path(work.name)
        self.root = self.work / "CMMEvent"
        self.image_root = self.root / "image" / "img"
        self.image_root.mkdir(parents=True)
        self.rows = []
        self.annotations = {}

    def image(self, name, color, folder="图片"):
        path = self.image_root / folder / name
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (20, 10), color).save(path, format="PNG")
        return path

    def annotation(self, name, verb, box=None):
        self.annotations[name] = {
            "verb": verb, "width": 20, "height": 10,
            "bb": {"agent": box if box is not None else [1, 1, 8, 9]},
            "frames": [{"agent": "n00000001"}],
        }

    def document(self, doc_id, title="新闻甲", event_count=1):
        event = {
            "event_type": "活动类-比赛类", "trigger": "比赛", "trigger_start_index": 2,
            "arguments": [
                {"role": "参与者", "argument": "队员", "argument_start_index": 0},
                {"role": "时间", "argument": "上午", "argument_start_index": "des"},
                {"role": "器材", "argument": "球", "argument_start_index": "img"},
            ],
        }
        row = {
            "id": doc_id, "title": title, "text": "队员比赛", "description": "当天上午举行",
            "img_url": "", "imgcaption": "标注生成的图像描述",
            "event_list": [event] * event_count,
        }
        self.rows.append(row)
        return row

    def run_preprocess(self, name="output", *args):
        for filename, data in (
            ("annotation_text.json", self.rows), ("annotation_image.json", self.annotations)
        ):
            (self.root / filename).write_text(json.dumps(data), encoding="utf-8")
        output = self.work / name
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dataset-dir", str(self.root),
             "--output-dir", str(output), *args],
            cwd=self.work, capture_output=True, encoding="utf-8",
        )
        return result, output

    def read_records(self, output):
        return {
            record["id"]: record
            for split in SPLITS
            for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            for record in [json.loads(line)]
        }

    def test_joins_multi_event_aliases_and_keeps_separate_visual_annotations(self):
        image = self.image("scoring_abc_v1.PNG", "red")
        other = self.image_root / "photographing_abc_v1.png"
        shutil.copyfile(image, other)
        self.image("scoring_abc.png", "green")
        self.annotation("scoring_abc_v1.jpg", "scoring")
        self.annotation("photographing_abc_v1.jpg", "photographing", [0, 0, 25, 10])
        self.document("abczv1.jpg", event_count=2)
        self.document("missing.jpg")
        result, output = self.run_preprocess()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.read_records(output)
        self.assertEqual(list(records), ["abczv1.jpg"])
        record = records["abczv1.jpg"]
        self.assertEqual(len(record["event_list"]), 2)
        self.assertEqual(len(record["visual_annotations"]), 2)
        self.assertNotIn("imgcaption", record)
        self.assertTrue((self.root / record["image"]["path"]).is_file())
        event = record["event_list"][0]
        self.assertEqual(event["trigger_span"], [2, 4])
        self.assertEqual(event["arguments"][1]["source"], "description")
        self.assertEqual(event["arguments"][1]["evidence_span"], [2, 4])
        self.assertIsNone(event["arguments"][2]["evidence_span"])
        visual = {item["verb"]: item for item in record["visual_annotations"]}
        self.assertIsNone(visual["photographing"]["bb"]["agent"])
        audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["exclusions"]["missing_image"][0]["id"], "missing.jpg")

    def test_rejects_conflicting_images_and_records_bad_spans(self):
        self.image("scoring_abc.png", "red")
        self.image("photographing_abc.png", "blue")
        self.annotation("scoring_abc.png", "scoring")
        self.annotation("photographing_abc.png", "photographing")
        self.document("abc.jpg", event_count=2)
        self.image("usable.png", "green")
        row = self.document("usable.png")
        row["event_list"][0]["trigger_start_index"] = -1
        row["event_list"][0]["arguments"][0]["argument_start_index"] = "0"
        row["event_list"][0]["arguments"][1]["argument_start_index"] = "0"
        result, output = self.run_preprocess()
        self.assertEqual(result.returncode, 0, result.stderr)
        event = self.read_records(output)["usable.png"]["event_list"][0]
        self.assertIsNone(event["trigger_span"])
        self.assertEqual(event["arguments"][0]["evidence_span"], [0, 2])
        self.assertIsNone(event["arguments"][1]["evidence_span"])
        audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["exclusions"]["ambiguous_image"][0]["id"], "abc.jpg")
        self.assertEqual(len(audit["issues"]["invalid_argument_span"]), 1)

    def test_grouped_split_is_reproducible_and_keeps_multi_event_documents(self):
        first_image = None
        for index in range(40):
            if index == 2:
                shutil.copyfile(first_image, self.image_root / "2.png")
            else:
                image = self.image(f"{index}.png", (index * 5, 10, 0), folder="img")
                if index == 0:
                    first_image = image
            title = "同一新闻" if index in (0, 1) else f"新闻{index}"
            self.document(f"{index}.png", title, event_count=2 if index % 3 == 0 else 1)
        first, first_output = self.run_preprocess("first", "--seed", "7")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.rows.reverse()
        second, second_output = self.run_preprocess("second", "--seed", "7")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(
            (first_output / "splits.json").read_bytes(),
            (second_output / "splits.json").read_bytes(),
        )
        records = self.read_records(first_output)
        self.assertEqual(len(records), 40)
        self.assertEqual(len({records[f"{index}.png"]["split"] for index in (0, 1, 2)}), 1)
        self.assertEqual(
            {r["split"] for r in records.values() if len(r["event_list"]) > 1}, set(SPLITS)
        )

    def test_uses_supplied_split_without_overwriting_existing_output(self):
        for index in range(3):
            self.image(f"{index}.png", (index * 10, 0, 0))
            self.document(f"{index}.png", title=f"新闻{index}")
        split_file = self.work / "original.json"
        split_file.write_text(
            json.dumps({"train": ["2.png"], "dev": ["1.png"], "test": ["0.png"]}),
            encoding="utf-8",
        )
        result, output = self.run_preprocess("output", "--split-file", str(split_file))
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.read_records(output)
        self.assertEqual(records["0.png"]["split"], "test")
        self.assertEqual(records["2.png"]["split"], "train")
        retry, _ = self.run_preprocess()
        self.assertNotEqual(retry.returncode, 0)
        self.assertEqual(self.read_records(output), records)

    def test_excludes_corrupt_image_even_when_header_is_readable(self):
        path = self.image_root / "broken.jpg"
        Image.new("RGB", (40, 30), "red").save(path)
        path.write_bytes(path.read_bytes()[:-15])
        self.image("usable.png", "green")
        self.document("broken.jpg")
        self.document("usable.png")
        result, output = self.run_preprocess()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.read_records(output)), {"usable.png"})
        audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["exclusions"]["unreadable_image"][0]["id"], "broken.jpg")

    def test_keeps_in_bounds_boxes_for_one_pixel_dimension_differences(self):
        for name, height in (("rounding.png", 11), ("unresolved.png", 13)):
            self.image(name, "green")
            self.document(name)
            self.annotation(name, "scoring")
            self.annotations[name]["height"] = height
            self.annotations[name]["bb"]["place"] = [0, 0, 20, 11]
        result, output = self.run_preprocess()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.read_records(output)
        rounding = records["rounding.png"]["visual_annotations"][0]
        self.assertEqual(rounding["bb"]["agent"], [1, 1, 8, 9])
        self.assertIsNone(rounding["bb"]["place"])
        unresolved = records["unresolved.png"]["visual_annotations"][0]
        self.assertIsNone(unresolved["bb"]["agent"])
        self.assertEqual(unresolved["raw_bb"]["agent"], [1, 1, 8, 9])
        audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["issues"]["visual_size_tolerance"][0]["id"], "rounding.png")
        self.assertEqual(audit["issues"]["visual_size_mismatch"][0]["id"], "unresolved.png")


if __name__ == "__main__":
    unittest.main()
