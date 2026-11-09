import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


class M2E2PreprocessTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "M2E2"
        self.articles = self.root / "article"
        self.images = self.root / "image" / "image"
        self.annotations = self.root / "m2e2-master" / "m2e2-master" / "data" / "m2e2_annotations"
        for directory in (self.articles, self.images, self.annotations):
            directory.mkdir(parents=True)
        self.output = self.root.parent / "output"
        self.text_only = []
        self.text_multimedia = []
        self.image_only = {}
        self.image_multimedia = {}
        self.coref = ""

    def sentence(self, sentence_id, events=(), images=()):
        return {
            "sentence_id": sentence_id,
            "sentence_start": 0,
            "sentence_end": 10,
            "sentence": "News event",
            "words": ["News", "event", "."],
            "index": [["0", "3"], ["5", "9"], ["10", "10"]],
            "pos-tags": ["NN", "NN", "."],
            "combined-parsing": [],
            "golden-entity-mentions": [],
            "golden-event-mentions": list(events),
            "image": list(images),
        }

    def event(self, event_type, start=1):
        return {
            "event_type": event_type,
            "trigger": {"start": start, "end": start + 1, "text": ["News", "event"][start]},
            "arguments": [{"role": "Entity", "start": 0, "end": 1, "text": "News"}],
        }

    def run_prepare(self):
        import data_preprocess_m2e2

        for name, rows in (
            ("text_only_event", self.text_only),
            ("text_multimedia_event", self.text_multimedia),
            ("image_only_event", self.image_only),
            ("image_multimedia_event", self.image_multimedia),
        ):
            (self.annotations / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
        (self.annotations / "crossmedia_coref.txt").write_text(self.coref, encoding="utf-8")
        audit = data_preprocess_m2e2.prepare(self.root, self.output)
        rows = [json.loads(line) for line in (self.output / "test.jsonl").read_text(encoding="utf-8").splitlines()]
        return audit, {row["id"]: row for row in rows}

    def test_writes_held_out_manifest_with_native_text_image_and_link_targets(self):
        raw = "A raw article with different line endings.\r\nNext line."
        (self.articles / "news.rsd.txt").write_bytes(raw.encode("utf-8"))
        for index in range(3):
            Image.new("RGB", (20, 10)).save(self.images / f"news_{index}.jpg")
        (self.images.parent / "image_url_caption.json").write_text(
            json.dumps({"news": {"0": {"url": "https://example.com/news.jpg", "caption": "A caption"}}}),
            encoding="utf-8",
        )
        attack, meet = self.event("Conflict:Attack"), self.event("Contact:Meet", start=0)
        self.text_only = [self.sentence("news_0", [attack], ["news_0.jpg"]), self.sentence("news_1")]
        self.text_multimedia = [self.sentence("news_0", [meet, attack], ["news_0.jpg"])]
        self.image_only = {"news_1": {"event_type": "Conflict:Attack", "role": {}}}
        role = {"Entity": [["0", 2, 3, 19, 9]]}
        self.image_multimedia = {"news_0": {"event_type": "Contact:Meet", "role": role}}
        self.coref = "news_0\tnews_0.jpg\tContact:Meet\n"

        audit, rows = self.run_prepare()

        self.assertEqual({p.name for p in self.output.iterdir()}, {"audit.json", "splits.json", "test.jsonl"})
        self.assertEqual(json.loads((self.output / "splits.json").read_text(encoding="utf-8")), {"test": ["news"]})
        row = rows["news"]
        self.assertTrue(row["has_annotations"])
        self.assertEqual(row["text"], raw)
        self.assertEqual(row["article"]["path"], "article/news.rsd.txt")
        first, negative = row["sentences"]
        self.assertEqual(first["annotations"][0]["index"], [["0", "3"], ["5", "9"], ["10", "10"]])
        self.assertEqual(first["annotations"][1]["golden-event-mentions"], [meet, attack])
        self.assertEqual(first["events"], [
            {**attack, "sources": ["text_only_event", "text_multimedia_event"]},
            {**meet, "sources": ["text_multimedia_event"]},
        ])
        self.assertEqual(negative["events"], [])
        self.assertEqual([image["image_id"] for image in row["images"]], ["news_0.jpg", "news_1.jpg", "news_2.jpg"])
        annotated, image_only, unannotated = row["images"]
        self.assertEqual(annotated["caption"], "A caption")
        self.assertEqual(annotated["asset"]["path"], "image/image/news_0.jpg")
        self.assertEqual(annotated["visual_annotations"][0]["role"], role)
        self.assertEqual(annotated["visual_annotations"][0]["boxes_xyxy"], {"Entity": [[2, 3, 19, 9]]})
        self.assertEqual(image_only["visual_annotations"][0]["source"], "image_only_event")
        self.assertEqual(unannotated["visual_annotations"], [])
        self.assertEqual([(item["sentence_id"], item["image_id"], item["event_type"]) for item in row["crossmedia_coref"]], [
            ("news_0", "news_0.jpg", "Contact:Meet"),
        ])
        self.assertEqual(audit["split_counts"]["test"]["negative_sentences"], 1)
        self.assertEqual(audit["split_counts"]["test"]["text_events"], 2)
        self.assertEqual(audit["issues"]["duplicate_sentence_ids"][0]["sentence_id"], "news_0")
        self.assertEqual(len(audit["inputs"]), 6)

    def test_keeps_annotations_with_bad_assets_and_unresolved_links(self):
        (self.articles / "broken.rsd.txt").write_bytes(b"\xff")
        (self.images / "broken_0.jpg").write_bytes(b"broken image")
        Image.new("RGB", (20, 10)).save(self.images / "broken_1.jpg")
        aliases = self.images / "duplicate"
        aliases.mkdir()
        Image.new("RGB", (20, 10), "red").save(aliases / "broken_1.jpg")
        Image.new("RGB", (20, 10)).save(self.images / "orphan_0.jpg")
        Image.new("RGB", (20, 10)).save(self.images / "missing_0.jpg")
        self.text_only = [self.sentence("broken_0", [], ["broken_2.jpg"]), self.sentence("missing_0")]
        role = {"Entity": [["1", 2, 3, 21, 9]]}
        self.image_multimedia = {"broken_0": {"event_type": "Contact:Meet", "role": {}},
                                 "missing_0": {"event_type": "Contact:Meet", "role": role}}
        self.coref = "broken_0\tbroken_0.jpg\tConflict:Attack\n"

        audit, rows = self.run_prepare()

        self.assertEqual(set(rows), {"broken", "missing", "orphan"})
        self.assertFalse(rows["orphan"]["has_annotations"])
        self.assertEqual(rows["orphan"]["sentences"], [])
        self.assertEqual(rows["orphan"]["images"][0]["asset"]["path"], "image/image/orphan_0.jpg")
        self.assertEqual(audit["split_counts"]["test"]["documents_with_annotations"], 2)
        self.assertIsNone(rows["broken"]["text"])
        self.assertIsNone(rows["missing"]["text"])
        self.assertEqual(len(rows["broken"]["crossmedia_coref"]), 1)
        self.assertEqual([image["asset"] for image in rows["broken"]["images"]], [None, None, None])
        visual = rows["missing"]["images"][0]["visual_annotations"][0]
        self.assertEqual(visual["role"], role)
        self.assertEqual(visual["boxes_xyxy"], {"Entity": [None]})
        self.assertEqual(audit["exclusions"], {})
        self.assertEqual(len(audit["issues"]["missing_article"]), 2)
        for issue in ("unreadable_article", "unreadable_image", "ambiguous_image",
                      "missing_image", "unresolved_coref", "invalid_box", "unannotated_documents"):
            self.assertEqual(len(audit["issues"][issue]), 1, issue)

    def test_rejects_duplicate_object_keys_without_writing_partial_output(self):
        import data_preprocess_m2e2

        for name in ("text_only_event", "text_multimedia_event"):
            (self.annotations / f"{name}.json").write_text("[]", encoding="utf-8")
        (self.annotations / "image_only_event.json").write_text('{"news_0": {}, "news_0": {}}', encoding="utf-8")
        (self.annotations / "image_multimedia_event.json").write_text("{}", encoding="utf-8")
        (self.annotations / "crossmedia_coref.txt").touch()
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key.*news_0"):
            data_preprocess_m2e2.prepare(self.root, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
