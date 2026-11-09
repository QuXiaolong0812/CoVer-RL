import math
import tempfile
from pathlib import Path
import unittest

from evidence_scoring import (
    EvidenceScorer, applicability, claim_score, load_calibration, proposal_boxes, verbalize,
)
from train_event_nli import build_pairs, fit_temperature, train_inventory


def event(kind, trigger, arguments):
    return {"event_type": kind, "trigger": trigger, "arguments": [
        {"role": role, "argument": filler, "source": source} for role, filler, source in arguments
    ]}


def document(doc_id, split, body, events):
    return {"id": doc_id, "split": split, "title": "", "text": body, "description": "",
            "event_list": events}


class NLIPairsTest(unittest.TestCase):
    def setUp(self):
        self.train = [
            document("a", "train", "队员赛跑，在体育场参赛。", [
                event("活动类-比赛类", "赛跑", [("参赛者", "队员", "text"), ("比赛地点", "体育场", "text"),
                                          ("比赛地点", "看台", "image")]),
            ]),
            document("b", "train", "教练在操场参赛，观众庆祝。", [
                event("活动类-比赛类", "参赛", [("参赛者", "教练", "text"), ("比赛地点", "操场", "text")]),
                event("活动类-庆祝类", "庆祝", []),
            ]),
        ]

    def test_keeps_role_type_and_filler_in_minimal_pair_negatives(self):
        pairs = build_pairs(self.train, train_inventory(self.train), split="train", seed=42)
        first = [p for p in pairs if p["doc_id"] == "a"]
        by_kind = {p["kind"] for p in first}
        self.assertEqual(by_kind, {"header", "argument", "wrong_type", "wrong_role", "wrong_filler"})
        self.assertTrue(all("看台" not in p["hypothesis"] for p in first if p["label"]))
        self.assertTrue(all("比赛类" in p["hypothesis"] and "赛跑" in p["hypothesis"]
                            for p in first if p["kind"] == "wrong_role"))
        self.assertTrue(any("操场" in p["hypothesis"] for p in first if p["kind"] == "wrong_filler"))
        self.assertEqual(len({(p["premise"], p["hypothesis"]) for p in first}), len(first))

    def test_corpus_excludes_test_and_uses_train_donors_on_dev(self):
        inventory = train_inventory(self.train)
        dev = document("c", "dev", "领队在广场赛跑。", [
            event("活动类-比赛类", "赛跑", [("参赛者", "领队", "text"), ("比赛地点", "广场", "text")]),
        ])
        other = document("d", "dev", "保密者在保密地点赛跑。", [
            event("活动类-比赛类", "赛跑", [("参赛者", "保密者", "text"), ("比赛地点", "保密地点", "text")]),
        ])
        pairs = build_pairs([dev, other], inventory, split="dev", seed=42)
        self.assertFalse(any("保密" in p["hypothesis"] for p in pairs if p["doc_id"] == "c"))
        with self.assertRaises(ValueError):
            train_inventory([dev])
        with self.assertRaises(ValueError):
            build_pairs([dict(dev, split="test")], inventory, split="test", seed=42)

    def test_dev_temperature_softens_overconfident_mistakes(self):
        temperature, metrics = fit_temperature([8, -8, 8, -8], [1, 0, 0, 1])
        self.assertGreater(temperature, 1)
        self.assertLess(metrics["calibrated"]["nll"], metrics["uncalibrated"]["nll"])

    def test_duplicate_captions_cannot_label_a_gold_claim_negative(self):
        first = self.train[0]
        second = document("a2", "train", first["text"], [
            event("活动类-比赛类", "赛跑", [("参赛者", "队员", "text"), ("比赛地点", "体育场", "text")]),
            event("活动类-庆祝类", "赛跑", []),
        ])
        rows = [*self.train, second]
        pairs = build_pairs(rows, train_inventory(rows), split="train", seed=42)
        self.assertEqual(len({(p["premise"], p["hypothesis"]) for p in pairs}), len(pairs))
        celebration = [p for p in pairs if p["premise"] == first["text"]
                       and p["hypothesis"] == verbalize("活动类-庆祝类", "赛跑")]
        self.assertEqual([p["label"] for p in celebration], [1])


class TCRTest(unittest.TestCase):
    def test_uses_support_and_anneals_cross_modal_contradiction(self):
        masks = (True, True, True)
        self.assertAlmostEqual(claim_score(0.8, 0.2, masks, 1, 0)["rho"], 0.4)
        self.assertAlmostEqual(claim_score(0.8, 0.2, masks, 1, 1)["rho"], 0.5)
        self.assertEqual(claim_score(0, 0, masks, 0, 0)["rho"], 0)
        self.assertEqual(claim_score(0, 0, (False, False, False), 0, 0)["rho"], 0)
        text_only = claim_score(0.8, 0, (True, False, False), 0, 0)
        self.assertAlmostEqual(text_only["rho"], 0.8)
        self.assertAlmostEqual(text_only["s_x"], 0.8)

    def test_masks_ignore_gold_or_generated_provenance_and_missing_grounding(self):
        doc = {"title": "新闻", "image": {"path": "a.jpg"}, "visual_annotations": []}
        self.assertEqual(applicability(doc, None), (True, True, True))
        self.assertEqual(applicability(doc, "时间"), (True, False, False))
        self.assertEqual(applicability(doc, "参赛者"), (True, True, True))
        self.assertEqual(applicability({"image": {"path": "a.jpg"}}, "参赛者"), (False, True, False))

    def test_reduces_first_within_each_event_and_uses_only_modal_conflicts(self):
        doc = {"id": "d", "text": "新闻", "image": {"path": "a.jpg"}}
        candidate = [dict(event("比赛", "赛跑", []), id="E_1"),
                     dict(event("庆祝", "鼓掌", [("时间", "今天", "text"), ("参与者", "队员", "image")]), id="E_2")]
        scorer = EvidenceScorer.__new__(EvidenceScorer)
        scorer.text_support = lambda premise, hypothesis: 1.0 if "赛跑" in hypothesis else 0.0
        scorer.visual_support = lambda image_path, hypothesis, box: {"s_v": 0.0, "proposals": []}
        feedback = {"event_level": [{"event_id": "E_1", "error_type": "modal_conflict"},
                                    {"event_id": "E_2", "error_type": "wrong_type"}]}
        output = scorer.score(doc, candidate, feedback)
        self.assertAlmostEqual(output["tcr"], 0.2)
        self.assertEqual(output["events"][0]["claims"][0]["c"], 1.0)
        self.assertEqual(output["events"][1]["claims"][0]["c"], 0.0)
        self.assertEqual(output["events"][0]["claims"][0]["a_v"], 1)
        self.assertEqual(scorer.score(doc, [])["tcr"], 0.0)

    def test_preserves_role_and_trigger_in_verbalization(self):
        claim = verbalize("活动类-比赛类", "赛跑", "参赛者", "队员")
        self.assertTrue(all(field in claim for field in ("活动类-比赛类", "赛跑", "参赛者", "队员")))

    def test_proposals_are_ranked_bounded_and_deduplicated(self):
        boxes = [[1, 2, 8, 9], [1, 2, 8, 9], [-2, 0, 7, 12], [5, 5, 5, 5], [math.nan, 0, 1, 1]]
        proposals = proposal_boxes(boxes, [0.8, 0.9, 0.6, 0.99, 0.98], ["a", "b", "c", "d", "e"],
                                   size=(10, 10), limit=2)
        self.assertEqual([(p["box"], p["label"]) for p in proposals], [((1, 2, 8, 9), "b"), ((0, 0, 7, 10), "c")])

    def test_refuses_untrained_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "trained event-NLI"):
                load_calibration(Path(tmp))


if __name__ == "__main__":
    unittest.main()
