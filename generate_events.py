"""Generate structured CMMEvent candidates with a local Qwen3-VL checkpoint."""

from collections import Counter, defaultdict
from importlib.metadata import version
import json
import re
from time import perf_counter

import torch
import transformers
from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

from cmmevent_io import read_split
from run_logging import sha256_file, write_json
from vlm_io import compact_json, encode_image_prompt, generate_completion, strict_json_loads, vision_token_ids


TEXT_FIELDS = ("title", "text", "description")
JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", re.IGNORECASE | re.DOTALL)
MIB = 1024 ** 2


def event_schema(train_rows):
    roles = defaultdict(set)
    for row in train_rows:
        for event in row["event_list"]:
            roles[event["event_type"]].update(argument["role"] for argument in event["arguments"])
    return {name: sorted(values) for name, values in sorted(roles.items())}


def generator_prompt(row, schema):
    evidence = {key: row[key] for key in TEXT_FIELDS}
    return (
        "你是中文图文事件抽取器。结合图片、标题、正文和背景，抽取有证据的事件，包括可辨认的次要事件。"
        "event_type 和 role 使用下列事件类型及角色名。trigger 是文中的触发词，argument 是有依据的论元。"
        "只输出一个 JSON 对象，格式为 "
        '{"events":[{"event_type":"事件类型","trigger":"触发词","arguments":[{"role":"角色","argument":"论元"}]}]}。'
        '可缺省无证据的角色；没有可提取事件时输出 {"events":[]}。\n'
        f"事件类型及允许角色\n{compact_json(schema)}\n"
        f"待抽取文档\n{compact_json(evidence)}"
    )


def encode_generator(row, schema, processor, dataset_dir, max_image_pixels):
    return encode_image_prompt(processor, dataset_dir, row["image"]["path"],
                               generator_prompt(row, schema), max_image_pixels)


def inspect_events(raw, schema):
    text = raw.strip()
    fence = JSON_FENCE.fullmatch(text)
    style = "fenced_json" if fence else "json"
    events = []
    try:
        value = strict_json_loads(fence[1] if fence else text)
        if not isinstance(value, dict) or set(value) != {"events"} or not isinstance(value["events"], list):
            raise ValueError("Expected a JSON object with an events list.")
        for index, event in enumerate(value["events"], 1):
            if not isinstance(event, dict) or set(event) != {"event_type", "trigger", "arguments"}:
                raise ValueError(f"Unexpected fields in event {index}.")
            kind, trigger, arguments = event["event_type"], event["trigger"], event["arguments"]
            if not isinstance(kind, str) or kind not in schema:
                raise ValueError(f"Unknown event type in event {index}.")
            if not isinstance(trigger, str) or not trigger.strip() or not isinstance(arguments, list):
                raise ValueError(f"Invalid trigger or arguments in event {index}.")
            numbered = []
            for arg_index, arg in enumerate(arguments, 1):
                if not isinstance(arg, dict) or set(arg) != {"role", "argument"}:
                    raise ValueError(f"Unexpected argument fields in event {index}.")
                role, argument = arg["role"], arg["argument"]
                if not isinstance(role, str) or role not in schema[kind]:
                    raise ValueError(f"Role {role!r} is outside the schema for {kind!r}.")
                if not isinstance(argument, str) or not argument.strip():
                    raise ValueError(f"Empty argument in event {index}.")
                numbered.append({"id": f"A_{index}_{arg_index}", "role": role, "argument": argument})
            events.append({"id": f"E_{index}", "event_type": kind, "trigger": trigger, "arguments": numbered})
    except (ValueError, RecursionError) as exc:
        return {"format_valid": False, "format": style, "format_error": str(exc), "events": []}
    return {"format_valid": True, "format": style, "format_error": None, "events": events}


def select_documents(rows, limit):
    if limit is None or limit >= len(rows):
        return rows
    first = {}
    for row in rows:
        first.setdefault(len(row["event_list"]) > 1, row)
    leading = list(first.values())
    selected = {row["id"] for row in leading}
    return (leading + [row for row in rows if row["id"] not in selected])[:limit]


def load_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Generator inference requires CUDA.")
    device = torch.device("cuda", torch.cuda.current_device()) if args.device == "auto" else torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Use a CUDA device such as cuda:0, or auto.")
    with torch.cuda.device(device):
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{device} does not support BF16. Set dtype to float16 in the configuration.")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, use_safetensors=True,
        dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
        device_map="auto" if args.device == "auto" else {"": str(device)},
    ).eval()
    adapter_dir = getattr(args, "generator_adapter_dir", None)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False, local_files_only=True).eval()
    input_device = model.get_input_embeddings().weight.device
    if input_device.type == "meta":
        input_device = torch.device("cpu")
    return model, processor, input_device


def generate_events(args, run):
    train_rows, train_hash = read_split(args.processed_dir / "train.jsonl", "train")
    rows, split_hash = (
        (train_rows, train_hash) if args.split == "train"
        else read_split(args.processed_dir / f"{args.split}.jsonl", args.split)
    )
    schema = event_schema(train_rows)
    if not schema:
        raise ValueError("No event types found in the training schema.")
    selected = select_documents(rows, args.limit)
    run.update_manifest(
        inputs={"train.jsonl": train_hash, f"{args.split}.jsonl": split_hash},
        model_files={path.name: sha256_file(path) for path in sorted(args.model_dir.iterdir())
                     if path.is_file() and path.suffix in (".json", ".safetensors")},
        packages={"torch": torch.__version__, "transformers": transformers.__version__,
                  "accelerate": version("accelerate")},
        selected_doc_ids=[row["id"] for row in selected],
    )
    adapter_dir = getattr(args, "generator_adapter_dir", None)
    if adapter_dir is not None:
        run.update_manifest(adapter_files={name: sha256_file(adapter_dir / name)
                                           for name in ("adapter_config.json", "adapter_model.safetensors")})
    write_json(run.path / "event_schema.json", schema)
    run.include_in_archive("event_schema.json")
    torch.manual_seed(args.seed)
    model, processor, device = load_model(args)
    run.update_manifest(device={"model_map": model.hf_device_map, "input_device": str(device), "cuda": torch.version.cuda})
    decoding = {"temperature": args.temperature, "top_p": args.top_p} if args.do_sample else {}
    generation = GenerationConfig(
        do_sample=args.do_sample, max_new_tokens=args.max_new_tokens, use_cache=True,
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
        suppress_tokens=vision_token_ids(model), **decoding,
    )
    run.logger.info("Generating %d/%d documents from %s with %s", len(selected), len(rows), args.split, args.model_dir)
    run.summary["predictions_written"] = 0
    counts = Counter()
    with (run.path / "predictions.jsonl").open(
        "w", encoding="utf-8", errors="backslashreplace", newline="\n"
    ) as file:
        run.include_in_archive("predictions.jsonl")
        for row in selected:
            start = perf_counter()
            inputs = encode_generator(row, schema, processor, args.dataset_dir, args.max_image_pixels)
            raw, stats = generate_completion(model, processor, inputs, device, generation, include_token_ids=True)
            prediction = inspect_events(raw, schema)
            events = prediction.pop("events")
            record = {
                "doc_id": row["id"], "split": args.split,
                "input": {**{key: row[key] for key in TEXT_FIELDS},
                          "image_path": row["image"]["path"], "events": events},
                "raw_output": raw, "eos_token_ids": generation.eos_token_id, **prediction, **stats,
                "seconds": round(perf_counter() - start, 3),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            run.summary["predictions_written"] += 1
            counts["valid"] += int(record["format_valid"])
            counts["events"] += len(events)
            counts["token_limit_hits"] += int(stats["hit_token_limit"])
            run.logger.info("doc=%s valid=%s events=%d tokens=%d seconds=%.2f", row["id"],
                            record["format_valid"], len(events), stats["generated_tokens"], record["seconds"])
    report = {
        "split": args.split, "documents": len(selected), "valid_outputs": counts["valid"],
        "invalid_outputs": len(selected) - counts["valid"], "format_pass_rate": counts["valid"] / len(selected),
        "generated_events": counts["events"], "token_limit_hits": counts["token_limit_hits"],
    }
    if device.type == "cuda":
        report["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / MIB, 1)
    run.metric(**report)
    write_json(run.path / "report.json", report)
    run.include_in_archive("report.json")
    run.update_manifest(outputs={name: sha256_file(run.path / name)
                                 for name in ("event_schema.json", "predictions.jsonl", "report.json")})
    run.summary.update(report)
