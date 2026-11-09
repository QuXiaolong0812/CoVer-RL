from pathlib import Path
import unittest

from main import PROJECT, build_parser, resolve_config


class PreprocessEntrypointTest(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def resolve(self, *options):
        return resolve_config(self.parser.parse_args(["preprocess", *options]))

    def test_preserves_the_default_cmmevent_entrypoint(self):
        args = self.resolve()
        self.assertEqual(args.dataset, "cmmevent")
        self.assertEqual(args.dataset_dir, (PROJECT / "datasets" / "CMMEvent").resolve())
        self.assertEqual(args.output_dir, (PROJECT / "datasets" / "processed" / "cmmevent").resolve())
        self.assertEqual(args.seed, 42)

    def test_separates_new_dataset_inputs_and_outputs(self):
        for dataset, raw_name in (("m2e2", "M2E2"), ("wildmmevent", "WildMMEvent")):
            with self.subTest(dataset=dataset):
                args = self.resolve("--dataset", dataset)
                self.assertEqual(args.dataset_dir, (PROJECT / "datasets" / raw_name).resolve())
                self.assertEqual(args.output_dir, (PROJECT / "datasets" / "processed" / dataset).resolve())
        custom = self.resolve("--dataset", "m2e2", "--dataset-dir", "raw", "--annotations-dir", "labels")
        self.assertEqual(custom.dataset_dir, Path("raw").resolve())
        self.assertEqual(custom.annotations_dir, Path("labels").resolve())

    def test_rejects_split_options_for_the_held_out_m2e2_benchmark(self):
        for options in (("--seed", "7"), ("--split-file", "splits.json")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "held-out"):
                self.resolve("--dataset", "m2e2", *options)


if __name__ == "__main__":
    unittest.main()
