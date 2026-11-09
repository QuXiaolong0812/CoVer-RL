"""CoVer-RL training, inference and evaluation for CMMEvent."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from cmmevent_io import SPLITS
from run_logging import RunLog, sha256_file


PROJECT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT / "configs" / "inference.json"


def add_evidence_options(parser):
    parser.add_argument("--nli-dir", type=Path, help="Fine-tuned event NLI checkpoint. Enables TCR.")
    parser.add_argument("--evidence-device", help="Torch device for the frozen evidence scorers.")


def build_parser():
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    paths.add_argument("--models-dir", type=Path, help="Root directory for local checkpoint folders.")
    common = argparse.ArgumentParser(parents=[paths], add_help=False)
    common.add_argument("--max-image-pixels", type=int)
    common.add_argument("--max-new-tokens", type=int)
    selection = common.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--all", action="store_true", help="Use all rows instead of the configured limit.")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prep = commands.add_parser("preprocess", help="Prepare CMMEvent on CPU with an archived audit.")
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prep.add_argument("--output-dir", type=Path)
    prep.add_argument("--split-file", type=Path)
    prep.add_argument("--seed", type=int, default=42)
    generator = commands.add_parser("generate", parents=[common], help="Generate CMMEvent candidates.")
    generator.add_argument("--generator-model", help="Checkpoint folder name under --models-dir.")
    generator.add_argument("--generator-adapter-dir", type=Path, help="Generator LoRA from a GRPO cycle.")
    generator.add_argument("--device", help="CUDA device such as cuda:0, or auto.")
    generator.add_argument("--split", choices=SPLITS)
    generator.add_argument("--decode", choices=("greedy", "sample"))
    generator.add_argument("--seed", type=int)
    verifier = commands.add_parser("verifier-eval", parents=[common], help="Evaluate a verifier on SPP dev.")
    verifier.add_argument("--verifier-model", help="Checkpoint folder name under --models-dir.")
    verifier.add_argument("--device")
    verifier.add_argument("--spp-dir", type=Path, required=True)
    verifier.add_argument("--adapter-dir", type=Path, required=True)
    extraction = commands.add_parser("evaluate-extraction", help="Score CMMEvent generator predictions on CPU.")
    extraction.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    extraction.add_argument("--predictions", type=Path, required=True, help="Generator predictions.jsonl.")
    extraction.add_argument("--split", choices=SPLITS, help="Check this split against the predictions.")
    verify = commands.add_parser("verify", parents=[common], help="Verify generator candidates and write HSFP.")
    verify.add_argument("--verifier-model", help="Checkpoint folder name under --models-dir.")
    verify.add_argument("--device")
    verify.add_argument("--predictions", type=Path, required=True, help="Generator predictions.jsonl.")
    verify.add_argument("--adapter-dir", type=Path, required=True, help="LoRA trained on the selected verifier model.")
    dry = commands.add_parser("reward-dry-run", parents=[paths], help="Replay outcomes, HSFP and optional TCR.")
    dry.add_argument("--predictions", type=Path, required=True)
    dry.add_argument("--feedback", type=Path, required=True)
    dry.add_argument("--generator-model", help="Checkpoint that produced the original completion token IDs.")
    dry.add_argument("--progress", type=float, default=0.0)
    dry.add_argument("--limit", type=int)
    add_evidence_options(dry)
    nli = commands.add_parser("train-nli", parents=[paths], help="Fine-tune mDeBERTa on CMMEvent event-NLI.")
    nli.add_argument("--nli-model", help="Pretrained mDeBERTa folder under --models-dir.")
    nli.add_argument("--device")
    nli.add_argument("--epochs", type=int)
    nli.add_argument("--train-limit", type=int)
    nli.add_argument("--dev-limit", type=int)
    warmup = commands.add_parser("train-verifier-sft", parents=[paths], help="Warm up the verifier on SPP.")
    warmup.add_argument("--verifier-model")
    warmup.add_argument("--spp-dir", type=Path, required=True)
    warmup.add_argument("--device")
    warmup.add_argument("--max-steps", type=int)
    warmup.add_argument("--eval-samples", type=int)
    warmup.add_argument("--max-tokens", type=int)
    warmup.add_argument("--lora-rank", type=int)
    warmup.add_argument("--lora-scope", choices=("pilot", "full"))
    warmup.add_argument("--max-image-pixels", type=int)
    rl = commands.add_parser("train-rl", parents=[paths], help="Train both Qwen3-VL policies with interleaved GRPO.")
    rl.add_argument("--generator-model")
    rl.add_argument("--verifier-model")
    rl.add_argument("--generator-device")
    rl.add_argument("--verifier-device")
    rl.add_argument("--adapter-dir", type=Path, required=True, help="Verifier SFT LoRA used for initialization and KL.")
    rl.add_argument("--resume", type=Path, help="Completed cycle directory from an interrupted run.")
    rl.add_argument("--max-image-pixels", type=int)
    rl.add_argument("--max-new-tokens", type=int, help="Override the completion cap for both policies.")
    for name in ("cycles", "groups-per-update", "train-limit", "dev-limit", "generator-group-size",
                 "verifier-group-size", "seed"):
        rl.add_argument(f"--{name}", type=int)
    rl.add_argument("--temperature", type=float)
    rl.add_argument("--stop-after-cycle", type=int, help="Save a complete cycle and stop for a later resume.")
    add_evidence_options(rl)
    return parser


def project_path(value):
    return (PROJECT / Path(value)).resolve()


def positive_options(values, minimums):
    for key, minimum in minimums.items():
        if type(values[key]) is not int or values[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}.")


def evidence_options(args, config, models_dir):
    section = config["evidence"]
    return {
        "nli_dir": args.nli_dir.resolve() if args.nli_dir else None,
        "siglip_dir": (models_dir / section["siglip_model"]).resolve(),
        "dino_dir": (models_dir / section["dino_model"]).resolve(),
        "evidence_device": args.evidence_device or section["device"],
    }


def section_options(section, args):
    return {key: getattr(args, key, None) if getattr(args, key, None) is not None else value
            for key, value in section.items()}


def resolve_training(args, config, models_dir, resolved):
    section = config[{"train-nli": "event_nli", "train-verifier-sft": "verifier_sft", "train-rl": "rl"}[args.stage]]
    resolved.update(section_options(section, args))
    resolved.update(dataset_dir=project_path(config["dataset_dir"]), processed_dir=project_path(config["processed_dir"]))
    if args.stage == "train-nli":
        model = args.nli_model or config["evidence"]["nli_model"]
        resolved["model_dir"] = (models_dir / model).resolve()
        positive_options(resolved, {"epochs": 1, "batch_size": 1, "max_length": 8, "seed": 0})
    elif args.stage == "train-verifier-sft":
        model = args.verifier_model or config["verifier"]["model"]
        resolved.update(model_dir=(models_dir / model).resolve(), spp_dir=args.spp_dir.resolve())
        positive_options(resolved, {"max_steps": 1, "eval_samples": 1, "max_tokens": 2,
                                    "max_image_pixels": 1024, "lora_rank": 1, "seed": 0})
    else:
        generator = args.generator_model or config["generator"]["model"]
        verifier = args.verifier_model or config["verifier"]["model"]
        resolved.update(
            generator_model_dir=(models_dir / generator).resolve(), verifier_model_dir=(models_dir / verifier).resolve(),
            adapter_dir=args.adapter_dir.resolve(), resume=args.resume.resolve() if args.resume else None,
            stop_after_cycle=args.stop_after_cycle,
            **evidence_options(args, config, models_dir),
        )
        if args.max_new_tokens is not None:
            resolved.update(generator_max_new_tokens=args.max_new_tokens, verifier_max_new_tokens=args.max_new_tokens)
        positive_options(resolved, {
            "cycles": 1, "generator_updates": 1, "verifier_updates": 0, "groups_per_update": 1,
            "generator_group_size": 2, "verifier_group_size": 2, "candidate_attempts": 1,
            "ppo_epochs": 1, "dev_every": 1, "patience": 0, "seed": 0, "lora_rank": 1, "token_chunk": 1,
            "max_image_pixels": 1024, "generator_max_new_tokens": 1, "verifier_max_new_tokens": 1,
        })
        for key in ("temperature", "generator_lr", "verifier_lr", "grad_clip"):
            if not math.isfinite(resolved[key]) or resolved[key] <= 0:
                raise ValueError(f"{key} must be positive and finite.")
        for key in ("beta", "gamma"):
            if not math.isfinite(resolved[key]) or resolved[key] < 0:
                raise ValueError(f"{key} must be nonnegative and finite.")
        if not 0 < resolved["epsilon"] < 1:
            raise ValueError("epsilon must be in (0, 1).")
        start, end = resolved["alpha_start"], resolved["alpha_end"]
        if (start, end) != (0, 0) and not 0 < end <= start <= 1:
            raise ValueError("Set 0 < alpha_end <= alpha_start <= 1, or both to zero.")
        if not math.isfinite(resolved["failure_reward"]) or resolved["failure_reward"] >= 0:
            raise ValueError("failure_reward must be negative and finite.")
        if args.stop_after_cycle is not None and not 1 <= args.stop_after_cycle < resolved["cycles"]:
            raise ValueError("stop_after_cycle must be positive and less than the planned cycles.")
    for key in ("train_limit", "dev_limit"):
        if resolved.get(key) is not None:
            positive_options(resolved, {key: 1})
    if "learning_rate" in resolved and (not math.isfinite(resolved["learning_rate"]) or resolved["learning_rate"] <= 0):
        raise ValueError("learning_rate must be positive and finite.")
    return argparse.Namespace(**resolved)


def resolve_config(args):
    snapshot = args.config.read_bytes()
    config = json.loads(snapshot)
    resolved = {
        "config_path": args.config.resolve(), "config_sha256": hashlib.sha256(snapshot).hexdigest(),
        "runs_dir": project_path(config["runs_dir"]), "archives_dir": project_path(config["archives_dir"]),
    }
    if args.stage == "preprocess":
        resolved.update(dataset_dir=project_path(config["dataset_dir"]), seed=args.seed,
                        output_dir=args.output_dir.resolve() if args.output_dir else project_path(config["processed_dir"]),
                        split_file=args.split_file.resolve() if args.split_file else None)
        return argparse.Namespace(**resolved)
    if args.stage == "evaluate-extraction":
        resolved.update(processed_dir=project_path(config["processed_dir"]),
                        predictions=args.predictions.resolve(), split=args.split)
        return argparse.Namespace(**resolved)

    models_dir = project_path(args.models_dir if args.models_dir is not None else config["models_dir"])
    if args.stage in ("train-nli", "train-verifier-sft", "train-rl"):
        return resolve_training(args, config, models_dir, resolved)
    if args.stage == "reward-dry-run":
        if args.limit is not None and args.limit < 1:
            raise ValueError("limit must be positive.")
        if not 0 <= args.progress <= 1:
            raise ValueError("progress must be in [0, 1].")
        resolved.update(
            predictions=args.predictions.resolve(), feedback=args.feedback.resolve(), progress=args.progress,
            limit=args.limit, gamma=config["rl"]["gamma"], failure_reward=config["rl"]["failure_reward"],
            processed_dir=project_path(config["processed_dir"]), dataset_dir=project_path(config["dataset_dir"]),
            generator_model_dir=(models_dir / (args.generator_model or config["generator"]["model"])).resolve(),
            **evidence_options(args, config, models_dir),
        )
        return argparse.Namespace(**resolved)
    generator = args.stage == "generate"
    role = "generator" if generator else "verifier"
    section = config[role]
    model = getattr(args, f"{role}_model") or section["model"]
    resolved.update(
        models_dir=models_dir, model_dir=(models_dir / model).resolve(),
        dataset_dir=project_path(config["dataset_dir"]),
        limit=None if args.all else args.limit if args.limit is not None else section["limit"],
    )
    for key in ("max_image_pixels", "max_new_tokens"):
        override = getattr(args, key)
        resolved[key] = section[key] if override is None else override
    for key, minimum in (("max_image_pixels", 1024), ("max_new_tokens", 1)):
        if type(resolved[key]) is not int or resolved[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}.")
    limit = resolved["limit"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer or null.")

    if generator:
        resolved.update(
            processed_dir=project_path(config["processed_dir"]),
            split=args.split or section["split"], device=args.device or section["device"],
            dtype=section["dtype"], seed=section["seed"] if args.seed is None else args.seed,
            do_sample=section["do_sample"] if args.decode is None else args.decode == "sample",
            temperature=section["temperature"], top_p=section["top_p"],
            generator_adapter_dir=args.generator_adapter_dir.resolve() if args.generator_adapter_dir else None,
        )
        if resolved["split"] not in SPLITS or resolved["dtype"] not in ("bfloat16", "float16"):
            raise ValueError("Invalid generator split or dtype.")
        if type(resolved["do_sample"]) is not bool or type(resolved["seed"]) is not int:
            raise ValueError("do_sample must be boolean and seed must be an integer.")
        if not math.isfinite(resolved["temperature"]) or not 0 < resolved["temperature"]:
            raise ValueError("temperature must be finite and positive.")
        if not 0 < resolved["top_p"] <= 1:
            raise ValueError("top_p must be in (0, 1].")
    elif args.stage == "verifier-eval":
        resolved.update(spp_dir=args.spp_dir.resolve(), adapter_dir=args.adapter_dir.resolve(),
                        device=args.device or section.get("device", "cuda:0"))
    else:
        resolved.update(predictions=args.predictions.resolve(), adapter_dir=args.adapter_dir.resolve(),
                        dtype="bfloat16", decoding="greedy", device=args.device or section.get("device", "cuda:0"))
    return argparse.Namespace(**resolved)


def run_generator(args):
    from generate_events import generate_events

    return run_logged(args, "generator_infer", generate_events, ("generate_events.py", "vlm_io.py"))


def run_preprocess(args):
    from data_preprocess import prepare
    from run_logging import write_json

    def prepare_and_record(options, run):
        names = ("annotation_text.json", "annotation_image.json")
        inputs = {name: sha256_file(options.dataset_dir / name) for name in names}
        if options.split_file is not None:
            inputs[str(options.split_file)] = sha256_file(options.split_file)
        run.update_manifest(inputs=inputs, accelerator="cpu")
        audit = prepare(options.dataset_dir, options.output_dir, options.seed, options.split_file)
        write_json(run.path / "audit.json", audit)
        run.include_in_archive("audit.json")
        run.update_manifest(outputs={path.name: sha256_file(path) for path in sorted(options.output_dir.glob("*.json*"))})
        for split, counts in audit["split_counts"].items():
            run.metric(split=split, **counts)
            run.logger.info("%s documents=%d multi_event=%d", split, counts["documents"],
                            counts["multi_event_documents"])
        run.summary.update(output_dir=str(options.output_dir), split_counts=audit["split_counts"],
                           excluded=sum(len(items) for items in audit["exclusions"].values()))
        run.logger.info("Saved prepared dataset to %s", options.output_dir)

    return run_logged(args, "preprocess", prepare_and_record, ("data_preprocess.py",))


def run_extraction(args):
    from extraction_eval import evaluate_extraction

    return run_logged(args, "extraction_eval", evaluate_extraction, ("extraction_eval.py",))


def run_verify(args):
    from verify_generated import verify_generated

    return run_logged(args, "verify_generated", verify_generated, (
        "verify_generated.py", "evaluate.py", "train_verifier_sft.py", "vlm_io.py",
    ))


def run_logged(args, stage, action, code):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with RunLog(stage, args.runs_dir, args.archives_dir, config) as run:
        run.update_manifest(code={name: sha256_file(PROJECT / name) for name in (
            "main.py", "cmmevent_io.py", "run_logging.py", *code
        )})
        action(args, run)
    return 0


def run_verifier(args):
    from evaluate import main as evaluate

    keys = ("spp_dir", "adapter_dir", "dataset_dir", "model_dir", "runs_dir", "archives_dir", "device",
            "max_image_pixels", "max_new_tokens", "limit")
    options = [item for key in keys if getattr(args, key) is not None
               for item in (f"--{key.replace('_', '-')}", str(getattr(args, key)))]
    return evaluate(options)


def run_dry(args):
    from reward_dry_run import reward_dry_run

    return run_logged(args, "reward_dry_run", reward_dry_run,
                      ("reward_dry_run.py", "reward_routing.py", "evidence_scoring.py", "extraction_eval.py"))


def run_nli(args):
    from train_event_nli import run_nli as train

    return run_logged(args, "event_nli", train, ("train_event_nli.py", "evidence_scoring.py"))


def run_sft(args):
    from train_verifier_sft import train

    return run_logged(args, "verifier_sft", train, ("train_verifier_sft.py", "grpo.py", "vlm_io.py"))


def run_rl(args):
    from train_coevolution import train_coevolution

    return run_logged(args, "coevolution", train_coevolution, (
        "train_coevolution.py", "grpo.py", "reward_routing.py", "evidence_scoring.py", "generate_events.py",
        "extraction_eval.py", "evaluate.py", "train_verifier_sft.py", "vlm_io.py",
    ))


def main(argv=None):
    cli = build_parser().parse_args(argv)
    try:
        args = resolve_config(cli)
        commands = {
            "preprocess": run_preprocess,
            "generate": run_generator, "evaluate-extraction": run_extraction,
            "verifier-eval": run_verifier, "verify": run_verify,
            "reward-dry-run": run_dry, "train-nli": run_nli,
            "train-verifier-sft": run_sft, "train-rl": run_rl,
        }
        return commands[cli.stage](args)
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"{cli.stage} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
