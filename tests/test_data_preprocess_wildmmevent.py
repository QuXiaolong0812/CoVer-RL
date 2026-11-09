import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from data_preprocess_wildmmevent import prepare


class WildMMEventPreprocessTest(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.work = Path(work.name)
        self.root = self.work / "WildMMEvent"
        self.image_root = self.root / "Image"
        self.image_root.mkdir(parents=True)
        self.rows = []
        self.annotations = {}

    def document(self, doc_id, title, color=None, img_url=None):
        self.rows.append({
            "id": doc_id,
            "title": title,
            "text": "熊猫啃食竹子。\n\n熊猫继续啃食。",
            "description": "熊猫啃食竹子。",
            "img_url": img_url or f"Image/{doc_id}",
            "imgcaption": "一只熊猫正在进食。",
            "event_list": [{
                "event_type": "觅食类-进食类", "class": "觅食类", "trigger": "啃食",
                "trigger_start_index": 2,
                "arguments": [{"role": "动物", "argument": "熊猫", "argument_start_index": 0}],
            }],
        })
        self.annotations[doc_id] = {
            "verb": "feeding", "width": 20, "height": 10,
            "bb": {"Animal": [1, 1, 8, 9], "Food": [-1, -1, -1, -1]},
            "frames": [{"Animal": "n02510455", "Food": ""}] * 3,
        }
        if color:
            self.image(doc_id, color)

    def image(self, relative_name, color):
        path = self.image_root / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10), color).save(path)

    def write_json(self, path, data):
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def run_prepare(self, name="output", split_file=None):
        self.write_json(self.root / "annotation_text.json", self.rows)
        self.write_json(self.root / "annotation_image.json", self.annotations)
        return prepare(self.root, self.work / name, 42, split_file)

    def read_records(self, name="output"):
        return {
            record["id"]: record
            for path in (self.work / name).glob("*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()
            for record in [json.loads(line)]
        }

    def test_retains_chinese_offsets_caption_and_visual_provenance(self):
        doc_id = "feeding_WME000001.png"
        self.document(doc_id, "大熊猫进食的局部观察", "red")
        audit = self.run_prepare()

        record = self.read_records()[doc_id]
        source = self.rows[0]
        for field in ("title", "text", "description", "img_url", "imgcaption"):
            self.assertEqual(record[field], source[field])
        self.assertEqual(record["image"]["path"], source["img_url"])
        event = record["event_list"][0]
        self.assertEqual(event["class"], "觅食类")
        self.assertEqual(event["trigger_span"], [2, 4])
        self.assertEqual(event["arguments"][0]["evidence_span"], [0, 2])
        self.assertEqual(event["arguments"][0]["source"], "text")
        visual = record["visual_annotations"][0]
        self.assertEqual(visual["annotation_id"], doc_id)
        self.assertEqual(visual["frames"], self.annotations[doc_id]["frames"])
        self.assertEqual(visual["raw_bb"], self.annotations[doc_id]["bb"])
        self.assertEqual(visual["bb"], {"Animal": [1, 1, 8, 9], "Food": None})
        self.assertEqual(audit["usable"]["documents"], 1)
        self.assertEqual(audit["issues"], {})

    def test_audits_missing_ambiguous_and_incompatible_sources(self):
        self.document("missing.png", "甲")
        self.document("ambiguous.png", "乙")
        self.image("first/ambiguous.png", "red")
        self.image("second/ambiguous.png", "blue")
        self.document("conflict.png", "丙", "red", img_url="Image/missing.png")
        self.document("relocated.png", "丁")
        self.image("first/relocated.png", "green")
        self.annotations["relocated.png"]["width"] = 40
        self.document("no-visual.png", "戊", "black")
        del self.annotations["no-visual.png"]
        audit = self.run_prepare()

        self.assertEqual({reason: [item["id"] for item in items]
                          for reason, items in audit["exclusions"].items()}, {
            "missing_image": ["missing.png"],
            "ambiguous_image": ["ambiguous.png"],
            "image_source_mismatch": ["conflict.png"],
        })
        records = self.read_records()
        record = records["relocated.png"]
        self.assertEqual(record["image"]["path"], "Image/first/relocated.png")
        visual = record["visual_annotations"][0]
        self.assertEqual(visual["raw_bb"]["Animal"], [1, 1, 8, 9])
        self.assertIsNone(visual["bb"]["Animal"])
        self.assertEqual(records["no-visual.png"]["visual_annotations"], [])
        self.assertEqual(set(audit["issues"]), {
            "relocated_image", "visual_size_mismatch", "missing_visual_annotation",
        })

    def test_groups_transitive_duplicates_and_rejects_supplied_split_leakage(self):
        self.document("a.png", "Ａ  熊猫", "red")
        self.document("b.png", "a 熊猫", "blue")
        self.document("c.png", "目击熊猫", "blue")
        self.document("d.png", "金丝猴进食", "green")
        self.run_prepare()
        self.run_prepare("repeated")

        records = self.read_records()
        self.assertEqual(records, self.read_records("repeated"))
        self.assertEqual({records[doc_id]["group_id"] for doc_id in ("a.png", "b.png", "c.png")}, {"a.png"})
        self.assertEqual(len({records[doc_id]["split"] for doc_id in ("a.png", "b.png", "c.png")}), 1)

        supplied = self.work / "supplied.json"
        self.write_json(supplied, {"train": ["a.png", "b.png"], "dev": ["c.png"], "test": ["d.png"]})
        with self.assertRaisesRegex(ValueError, "group 'a.png'.*across splits"):
            self.run_prepare("leaked", supplied)
        self.assertFalse((self.work / "leaked").exists())

        self.write_json(supplied, {"train": ["d.png"], "dev": [], "test": ["a.png", "b.png", "c.png"]})
        audit = self.run_prepare("supplied", supplied)
        self.assertEqual(audit["split_source"], "supplied")
        self.assertEqual(self.read_records("supplied")["a.png"]["split"], "test")

    def test_rejects_duplicate_visual_and_split_keys(self):
        self.document("a.png", "熊猫", "red")
        self.write_json(self.root / "annotation_text.json", self.rows)
        visual_path = self.root / "annotation_image.json"
        visual = json.dumps(self.annotations["a.png"])
        visual_path.write_text(f'{{"a.png":{visual},"a.png":{visual}}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            prepare(self.root, self.work / "duplicate-visual", 42, None)

        split = self.work / "duplicate-split.json"
        split.write_text('{"train":[],"dev":[],"test":[],"train":["a.png"]}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.run_prepare("duplicate-split", split)
        self.assertFalse((self.work / "duplicate-visual").exists())
        self.assertFalse((self.work / "duplicate-split").exists())


if __name__ == "__main__":
    unittest.main()
