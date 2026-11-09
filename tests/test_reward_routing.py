import unittest

from reward_routing import (
    class_weight, generator_advantages, local_units, route_local_rewards, standardized,
    structural_score, verifier_reward,
)


def event(trigger, *arguments, kind="visit"):
    return {
        "event_type": kind,
        "trigger": trigger,
        "arguments": [{"role": role, "argument": value} for role, value in arguments],
    }


class LiteralTokenizer:
    """Each character is a token, with one alternative segmentation for a JSON suffix."""

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [
            (i, i + 1) for i in range(len(text))
        ]}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join("}]}" if token == 2000000 else chr(token) for token in ids)


class StructuralRewardTest(unittest.TestCase):
    def test_structural_matching_uses_canonical_role_values_and_counts_duplicates(self):
        gold = [event("opened", ("host", "Cafe\u0301\u00a0Hall"), ("city", "Paris"))]
        pred = [event("opened", ("host", "  Café Hall "), ("host", "Paris"))]
        self.assertEqual(structural_score(pred, gold), {
            "tp": 2, "fp": 1, "fn": 1, "f1": 2 / 3, "label": "partial",
        })
        self.assertEqual(structural_score(gold * 2, gold), {
            "tp": 3, "fp": 3, "fn": 0, "f1": 2 / 3, "label": "partial",
        })

    def test_structural_matching_aligns_events_without_replacement_and_ignores_order(self):
        gold = [event("opened", ("city", "Paris")), event("closed", ("city", "Rome"))]
        pred = [event("closed", ("city", "Rome")), event("opened", ("city", "Rome"))]
        self.assertEqual(structural_score(pred, gold)["tp"], 3)
        self.assertEqual(structural_score(gold[::-1], gold)["label"], "correct")
        # The locally best first match blocks a later match and loses a filler.
        left = [event("x", ("role", "a"), ("role", "b")), event("x", ("role", "a"))]
        right = [event("x", ("role", "a")), event("x", ("role", "b"))]
        self.assertEqual(structural_score(left, right)["tp"], 4)
        self.assertEqual(structural_score(left[::-1], right[::-1])["tp"], 4)

    def test_empty_is_distinct_from_malformed_and_wrong_header_can_be_partial(self):
        self.assertEqual(structural_score([], [])["f1"], 1.0)
        self.assertEqual(structural_score(None, [])["label"], "malformed")
        gold = [event("opened", ("city", "Rome"))]
        self.assertEqual(structural_score([], gold)["label"], "incorrect")
        self.assertEqual(structural_score([event("closed", ("city", "Rome"))], gold)["tp"], 1)
        self.assertEqual(structural_score([event("opened", ("city", "Rome"), kind="sale")], gold)["tp"], 0)

    def test_verifier_reward_is_gated_by_complete_hsfp_format(self):
        checked = {"format_valid": True, "parsed": {"final_verification": "partial"}}
        self.assertEqual(verifier_reward(checked, "partial"), 1.8)
        self.assertEqual(verifier_reward(checked, "correct"), 0.8)
        self.assertEqual(verifier_reward({**checked, "format_valid": False}, "partial"), 0.0)


class LocalRoutingTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = LiteralTokenizer()
        self.raw = (
            ' \r\n```JSON\r\n'
            '{"events":[{"event_type":"类型","trigger":"tr\\"ig",'
            '"arguments":[{"role":"r","argument":"{中}"}]}]}'
            '\r\n```\t'
        )
        self.eos = 2000001
        self.units = [
            {"level": "event", "event_id": "E_1", "argument_id": None, "reward": 0.4, "omission": False},
            {"level": "argument", "event_id": "E_1", "argument_id": "A_1_1", "reward": 0.3, "omission": False},
            {"level": "argument", "event_id": "E_1", "argument_id": None, "reward": -0.3, "omission": True},
        ]

    def test_local_budgets_include_omissions_and_unresolved_records(self):
        parsed = {
            "event_level": [
                {"event_id": "E_1", "status": "supported", "error_type": None},
                {"event_id": "E_2", "status": "unresolved", "error_type": None},
                {"event_id": None, "status": "missing", "error_type": "missing_event"},
            ],
            "argument_level": [
                {"event_id": "E_1", "argument_id": "A_1_1", "status": "correct", "error_type": None},
                {"event_id": "E_1", "argument_id": "A_1_2", "status": "incorrect", "error_type": "wrong_role"},
            ],
        }
        units = local_units(parsed)
        self.assertEqual([unit["reward"] for unit in units], [0.4 / 3, 0.0, -0.4 / 3, 0.3, -0.3])
        self.assertEqual([unit["omission"] for unit in units], [False, False, True, False, False])
        self.assertEqual(local_units(None), [])

    def test_fenced_raw_positions_and_eos_causal_returns(self):
        ids = [*map(ord, self.raw), self.eos]
        returns, details = route_local_rewards(self.raw, ids, self.tokenizer, self.units, [self.eos])
        self.assertTrue(details["ok"])
        header = self.raw.index(',"arguments"') - 1
        argument = self.raw.index('}"}]') + 2
        self.assertEqual([item["token_index"] for item in details["placements"]], [header, argument, len(ids) - 1])
        self.assertTrue(details["ended_by_eos"])
        self.assertAlmostEqual(returns[header], 0.4)
        self.assertAlmostEqual(returns[header + 1], 0.0)
        self.assertAlmostEqual(returns[argument + 1], -0.3)
        self.assertEqual(returns[-1], -0.3)
        self.assertEqual(
            route_local_rewards(self.raw, ids, self.tokenizer, self.units, self.eos), (returns, details)
        )

    def test_noncanonical_sampled_segmentation_and_last_token_omission_fallback(self):
        raw = '{"events":[{"arguments":[],"trigger":"c","event_type":"visit"}]}'
        ids = [*map(ord, raw[:-3]), 2000000]
        units = [self.units[0], self.units[2]]
        returns, details = route_local_rewards(raw, ids, self.tokenizer, units, [self.eos])
        self.assertTrue(details["ok"])
        self.assertFalse(details["ended_by_eos"])
        self.assertEqual(details["terminal_placement"], "last_observed_token")
        self.assertEqual([item["token_index"] for item in details["placements"]], [len(ids) - 2, len(ids) - 1])
        self.assertEqual(returns[-1], -0.3)

    def test_mismatched_decoding_masks_the_local_channel(self):
        ids = [*map(ord, self.raw[:-1]), self.eos]
        returns, details = route_local_rewards(self.raw, ids, self.tokenizer, self.units, [self.eos])
        self.assertFalse(details["ok"])
        self.assertTrue(details["error"])
        self.assertEqual(returns, [0.0] * len(ids))


class AdvantageTest(unittest.TestCase):
    def test_sequence_balanced_local_variance_and_masking(self):
        actual = generator_advantages([0.5, 0.5], [[0.0], [2.0] * 3], 0.25)
        self.assertAlmostEqual(actual[0][0], -0.25)
        self.assertEqual(len(actual[1]), 3)
        self.assertAlmostEqual(actual[1][2], 0.25)
        masked = generator_advantages([0.5, 0.5], [[0.0], [2.0] * 3], 0.25, local_mask=[False, True])
        self.assertEqual(masked, [[0.0], [0.0, 0.0, 0.0]])

    def test_population_variance_adds_epsilon_inside_the_outcome_root(self):
        actual = standardized([0.0, 2.0], delta=3.0)
        self.assertEqual(actual, [-0.5, 0.5])
        self.assertEqual(standardized([0.8, 0.8]), [0.0, 0.0])
        mixed = generator_advantages([0.0, 2.0], [[0.0], [2.0]], 0.5, delta=3.0)
        self.assertEqual(mixed, [[-0.375], [0.375]])

    def test_class_weight_scales_raw_groups_to_unit_expected_weight(self):
        counts = {"correct": 9, "partial": 1, "incorrect": 0}
        frequent, rare = (class_weight(counts, label, delta=0.1) for label in ("correct", "partial"))
        self.assertAlmostEqual(rare / frequent, 5 ** 0.5)
        self.assertAlmostEqual(0.9 * frequent + 0.1 * rare, 1.0)
        self.assertEqual(class_weight({}, "partial"), 1.0)


if __name__ == "__main__":
    unittest.main()
