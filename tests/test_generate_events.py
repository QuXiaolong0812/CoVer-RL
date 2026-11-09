import json
from pathlib import Path
import tempfile
import unittest

from generate_events import event_schema, generator_prompt, inspect_events
from main import build_parser, resolve_config


SCHEMA = {"比赛": ["参赛者"], "拍照": ["拍摄者"]}
EVENT = {"event_type": "比赛", "trigger": "比赛", "arguments": [{"role": "参赛者", "argument": "队员"}]}


class GeneratorTest(unittest.TestCase):
    def test_prompt_exposes_only_document_evidence_and_schema(self):
        row = {"title": "比赛新闻", "text": "队员正在比赛。", "description": "赛况",
               "event_list": [{"event_type": "SECRET_GOLD"}], "image": {"path": "SECRET_FILENAME.jpg"},
               "visual_annotations": [{"verb": "SECRET_VISUAL"}]}
        schema = event_schema([{"event_list": [EVENT]}])
        self.assertEqual(schema, {"比赛": ["参赛者"]})
        prompt = generator_prompt(row, schema)
        self.assertIn("队员正在比赛", prompt)
        self.assertIn("参赛者", prompt)
        self.assertNotIn("SECRET_", prompt)

    def test_parses_complete_json_or_fence_and_assigns_stable_ids(self):
        raw = json.dumps({"events": [EVENT, EVENT]}, ensure_ascii=False)
        for text, style in [(raw, "json"), (f"```json\n{raw}\n```", "fenced_json")]:
            with self.subTest(style=style):
                result = inspect_events(text, SCHEMA)
                self.assertTrue(result["format_valid"])
                self.assertEqual(result["format"], style)
                self.assertEqual([event["id"] for event in result["events"]], ["E_1", "E_2"])
                self.assertEqual(result["events"][1]["arguments"][0]["id"], "A_2_1")

    def test_rejects_unknown_roles_and_duplicate_json_keys_as_whole_output(self):
        unknown = {**EVENT, "arguments": [{"role": "摄影器材", "argument": "相机"}]}
        for raw in ('{"events":[],"events":[]}', json.dumps({"events": [EVENT, unknown]})):
            with self.subTest(raw=raw):
                result = inspect_events(raw, SCHEMA)
                self.assertFalse(result["format_valid"])
                self.assertEqual(result["events"], [])
        self.assertTrue(inspect_events('{"events":[]}', SCHEMA)["format_valid"])

    def test_role_specific_model_overrides_and_all_documents(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            model_root = Path(tmp)
            generate = parser.parse_args([
                "generate", "--models-dir", str(model_root), "--generator-model", "local-8B", "--all",
            ])
            cfg = resolve_config(generate)
            self.assertEqual(cfg.model_dir, model_root / "local-8B")
            self.assertIsNone(cfg.limit)
            verifier = parser.parse_args([
                "verifier-eval", "--models-dir", str(model_root), "--verifier-model", "local-4B",
                "--spp-dir", tmp, "--adapter-dir", tmp,
            ])
            cfg = resolve_config(verifier)
            self.assertEqual(cfg.model_dir, model_root / "local-4B")


if __name__ == "__main__":
    unittest.main()
