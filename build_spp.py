"""Build paired CMMEvent SPP data and archive the run logs.

Build the prompt from title, text, description and candidate events.
Use `input.image_path` only to load the image relative to --dataset-dir.
`target` contains HSFP labels. `provenance` stores the zero-based inverse edit.
The test split is held out.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
import sys

from run_logging import RunLog, sha256_file, write_json
from spp_operators import EXTENDED_OPERATORS, TEXT_FIELDS, extended_options, image_scene, same_sentence, unique_span


PROJECT = Path(__file__).resolve().parent
PILOT_SPLITS = ("train", "dev")
PILOT_OPERATORS = ("A1", "A2")
OPERATORS = (*PILOT_OPERATORS, *EXTENDED_OPERATORS)
SCHEMA_VERSION = 1
ORIGIN_MARKERS = ("从", "由", "自")
LOCATION_MARKERS = {"出发地": ORIGIN_MARKERS, "出发点": ORIGIN_MARKERS, "目的地": ("到", "向", "往", "至", "抵达")}


def load_inputs(data_dir, run):
    snapshots = {name: (data_dir / name).read_bytes() for name in ("splits.json", "train.jsonl", "dev.jsonl")}
    run.update_manifest(inputs={name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()})
    split_ids = json.loads(snapshots["splits.json"])
    if not isinstance(split_ids, dict) or set(split_ids) != {"train", "dev", "test"} or any(
        not isinstance(ids, list) for ids in split_ids.values()
    ):
        raise ValueError("Expected a split manifest with train, dev, and test ID lists.")
    ids = [doc_id for split in split_ids.values() for doc_id in split]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate document IDs in splits.json.")

    owners = {}
    documents = {}
    for split in PILOT_SPLITS:
        with io.StringIO(snapshots[f"{split}.jsonl"].decode("utf-8-sig"), newline=None) as file:
            rows = [json.loads(line) for line in file]
        if [row["id"] for row in rows] != split_ids[split]:
            raise ValueError(f"{split}.jsonl disagrees with splits.json.")
        for row in rows:
            if row["split"] != split:
                raise ValueError(f"Unexpected split for document {row['id']!r}.")
            for key in (("group", row["group_id"]), ("image", row["image"]["sha256"])):
                if owners.setdefault(key, split) != split:
                    raise ValueError(f"Cross-split {key[0]} for document {row['id']!r}.")
        documents[split] = rows
    return documents


def load_a2_reviews(path, documents, run):
    hashes = run.manifest["inputs"]
    snapshot = path.read_bytes()
    run.update_manifest(inputs={**hashes, "a2_review": hashlib.sha256(snapshot).hexdigest()})
    review = json.loads(snapshot)
    if not isinstance(review, dict) or review.get("schema_version") != 1:
        raise ValueError("Expected an A2 review with schema_version=1.")
    expected = {f"{split}.jsonl": hashes[f"{split}.jsonl"] for split in PILOT_SPLITS}
    if review.get("inputs") != expected:
        raise ValueError("A2 review does not match the source train/dev SHA-256 hashes.")
    records = review.get("records")
    if not isinstance(records, dict) or set(records) != set(PILOT_SPLITS):
        raise ValueError("A2 review must contain train and dev records.")

    result = {}
    for split in PILOT_SPLITS:
        eligible = {row["id"] for row in documents[split] if len(row["event_list"]) > 1}
        by_id = {}
        if not isinstance(records[split], list):
            raise ValueError(f"A2 review {split} records must be a list.")
        for record in records[split]:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid A2 review record in {split}.")
            doc_id = record.get("doc_id")
            if not isinstance(doc_id, str) or doc_id not in eligible or doc_id in by_id:
                raise ValueError(f"Unexpected or repeated A2 review document in {split}: {doc_id!r}")
            if record.get("decision") not in ("approve", "exclude"):
                raise ValueError(f"Invalid A2 review decision for {doc_id!r}.")
            reason = record.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"Missing A2 review rationale for {doc_id!r}.")
            by_id[doc_id] = record
        if split == "dev" and set(by_id) != eligible:
            raise ValueError("A2 review must cover every multi-event dev document.")
        result[split] = by_id
    return result


def argument_evidence(row, argument):
    source = argument["source"]
    if source not in TEXT_FIELDS:
        return None
    value = argument["argument"]
    text = row[source]
    span = unique_span(text, value)
    if span is None or span != argument["evidence_span"]:
        return None
    role = argument["role"]
    # A literal place name does not establish whether it is an origin or destination.
    markers = LOCATION_MARKERS.get(role)
    if markers and not text[:span[0]].rstrip().endswith(markers):
        return None
    return {"source": source, "span": span, "text": value, "role": role}


def grounded_arguments(row, event, value_counts):
    trigger = event["trigger"]
    trigger_span = unique_span(row["text"], trigger)
    if trigger_span is None or trigger_span != event["trigger_span"]:
        return []
    roles = Counter(argument["role"] for argument in event["arguments"])
    options = []
    for index, argument in enumerate(event["arguments"]):
        if roles[argument["role"]] != 1 or value_counts[argument["argument"]] != 1:
            continue
        evidence = argument_evidence(row, argument)
        if evidence is None:
            continue
        source = row[evidence["source"]]
        anchor = unique_span(source, trigger)
        if anchor is not None and same_sentence(source, evidence["span"], anchor):
            options.append((index, {**evidence, "trigger_span": anchor}))
    return options


def secondary_evidence(row, index, grounded, *, reviewed=False):
    event = row["event_list"][index]
    if sum(other["event_type"] == event["event_type"] for other in row["event_list"]) != 1:
        return None
    supports = [argument_evidence(row, arg) for arg in event["arguments"]]
    if any(evidence is None for evidence in supports):
        return None
    if reviewed:
        # Review can disambiguate distinct events that share a participant or
        # place. Their spans and a same-sentence body anchor still have to match.
        anchor = unique_span(row["text"], event["trigger"])
        if anchor is None or anchor != event["trigger_span"] or not any(
            evidence["source"] == "text" and same_sentence(row["text"], evidence["span"], anchor)
            for evidence in supports
        ):
            return None
    elif not any(evidence["source"] == "text" for _, evidence in grounded[index]):
        return None
    return [{"source": "text", "span": event["trigger_span"], "text": event["trigger"]}, *supports]


def reviewed_a2(row, grounded, review):
    if not review:
        return [], "not_reviewed"
    if review["decision"] == "exclude":
        return [], "excluded_by_review"
    events = row["event_list"]
    primary, secondary = (review.get(key) for key in ("primary_event_index", "secondary_event_index"))
    if (any(type(index) is not int or not 0 <= index < len(events) for index in (primary, secondary))
            or primary == secondary):
        raise ValueError(f"Invalid reviewed A2 event indices for {row['id']!r}.")
    cue = review.get("primary_evidence")
    if not isinstance(cue, dict) or cue.get("source") not in TEXT_FIELDS or not isinstance(cue.get("text"), str):
        raise ValueError(f"Invalid reviewed primary evidence for {row['id']!r}.")
    span = unique_span(row[cue["source"]], cue["text"])
    evidence = secondary_evidence(row, secondary, grounded, reviewed=True)
    if span is None or evidence is None:
        raise ValueError(f"Reviewed A2 evidence no longer passes checks for {row['id']!r}.")
    evidence.insert(1, {**cue, "span": span, "kind": "reviewed_primary", "event_index": primary})
    return [(secondary, None, evidence)], None


def deletion_options(row, review=None):
    events = row["event_list"]
    values = Counter(arg["argument"] for event in events for arg in event["arguments"])
    grounded = [grounded_arguments(row, event, values) for event in events]
    a1 = [
        (index, arg_index, [evidence])
        for index, options in enumerate(grounded)
        for arg_index, evidence in options
    ]
    if len(events) < 2:
        return a1, [], "single_event"
    if review is not None:
        a2, reason = reviewed_a2(row, grounded, review)
        return a1, a2, reason

    title = row["title"]
    primary = [index for index, event in enumerate(events) if event["trigger"] and event["trigger"] in title]
    if len(primary) != 1:
        return a1, [], "no_unique_primary_title_anchor"
    primary_index = primary[0]
    primary_trigger = events[primary_index]["trigger"]
    title_anchor = unique_span(title, primary_trigger)
    anchor = unique_span(row["text"], primary_trigger)
    if title_anchor is None or anchor is None or anchor != events[primary_index]["trigger_span"]:
        return a1, [], "unverified_primary_trigger"

    a2 = []
    for index in range(len(events)):
        if index == primary_index:
            continue
        evidence = secondary_evidence(row, index, grounded)
        if evidence is not None:
            evidence.insert(1, {"source": "title", "span": title_anchor, "text": primary_trigger,
                                "kind": "primary_trigger"})
            a2.append((index, None, evidence))
    if len(a2) > 1:
        return a1, [], "multiple_secondary_targets"
    return a1, a2, None if a2 else "no_grounded_secondary_event"


def take_options(pool, limit, groups, rng):
    selected = []
    for row, options in pool:
        if len(selected) >= limit:
            break
        if row["group_id"] not in groups:
            selected.append((row, rng.choice(options)))
            groups.add(row["group_id"])
    return selected


def select_image_swaps(pool, limit, groups, rng):
    selected = []
    donor_groups, donor_images = set(), set()
    for row, (option,) in pool:
        if len(selected) >= limit:
            break
        if row["group_id"] in groups:
            continue
        available = [(donor, evidence) for donor, evidence in option["donors"]
                     if donor["group_id"] not in donor_groups and donor["image"]["sha256"] not in donor_images]
        if not available:
            continue
        donor, evidence = rng.choice(available)
        donor_groups.add(donor["group_id"])
        donor_images.add(donor["image"]["sha256"])
        groups.add(row["group_id"])
        selected.append((row, {"event_index": option["event_index"], "donor": donor, "evidence": [
            *option["evidence"], {**evidence, "kind": "replacement_scene", "sha256": donor["image"]["sha256"],
                                  "event_type": donor["event_list"][0]["event_type"]},
        ]}))
    return selected


def select_balanced_a1(pool, limit, groups, rng):
    options = []
    for row, deletions in pool:
        for deletion in deletions:
            event_type = row["event_list"][deletion[0]]["event_type"]
            evidence = deletion[2][0]
            options.append((row, deletion, event_type, evidence["role"], evidence["source"]))
    rng.shuffle(options)
    background = [option for option in options if option[4] == "description"]
    other = [option for option in options if option[4] != "description"]
    other_groups = {option[0]["group_id"] for option in other} - groups
    type_counts, role_counts, pair_counts = Counter(), Counter(), Counter()
    selected = []

    def pick(pool, until, reserve_other=0):
        while len(selected) < until:
            available = (
                option for option in pool if option[0]["group_id"] not in groups
                and (len(other_groups) > reserve_other or option[0]["group_id"] not in other_groups)
            )
            choice = min(available, key=lambda option: (
                type_counts[option[2]], role_counts[option[3]], pair_counts[option[2:4]]
            ), default=None)
            if choice is None:
                break
            row, deletion, event_type, role, _ = choice
            selected.append((row, deletion))
            groups.add(row["group_id"])
            other_groups.discard(row["group_id"])
            type_counts[event_type] += 1
            role_counts[role] += 1
            pair_counts[event_type, role] += 1

    # Reserve roughly half the budget for background, balance types/roles
    # across both sources, and fill any unmet quota from the remaining groups.
    background_target = (limit + 1) // 2
    pick(background, background_target, min(limit - background_target, len(other_groups)))
    pick(other, limit)
    pick(options, limit)
    return selected


def candidate_events(row):
    return [
        {"event_type": event["event_type"], "trigger": event["trigger"],
         "arguments": [{"role": arg["role"], "argument": arg["argument"]} for arg in event["arguments"]]}
        for event in row["event_list"]
    ]


def hsfp(events, edit):
    event_level = [
        {"event_id": event["id"], "status": "correct", "error_type": None, "feedback": "保留该事件。"}
        for event in events
    ]
    argument_level = [
        {"event_id": event["id"], "argument_id": arg["id"], "role": arg["role"],
         "status": "correct", "error_type": None, "feedback": "保留该论元。"}
        for event in events for arg in event["arguments"]
    ]
    if edit is not None:
        value = edit["value"]
        if "argument_index" in edit:
            argument_level.append({
                "event_id": events[edit["event_index"]]["id"], "argument_id": None, "role": value["role"],
                "status": "missing", "error_type": "missing_argument",
                "feedback": f"补充论元「{value['role']}」为「{value['argument']}」。",
            })
        else:
            args = "、".join(f"「{arg['role']}」为「{arg['argument']}」" for arg in value["arguments"])
            event_level.append({
                "event_id": None, "status": "missing", "error_type": "missing_event",
                "feedback": f"补充「{value['event_type']}」事件，触发词为「{value['trigger']}」，论元包括{args}。",
            })
    return {
        "event_level": event_level, "argument_level": argument_level,
        "final_verification": "partial" if edit is not None else "correct",
    }


def sample(row, operator, deletion=None):
    events = candidate_events(row)
    edit, evidence = None, []
    if deletion is not None:
        event_index, arg_index, evidence = deletion
        edit = {"event_index": event_index}
        if arg_index is None:
            edit["value"] = events.pop(event_index)
        else:
            edit["argument_index"] = arg_index
            edit["value"] = events[event_index]["arguments"].pop(arg_index)

    # Number only what the verifier sees, so missing units leave no gaps.
    for event_index, event in enumerate(events, 1):
        event["id"] = f"E_{event_index}"
        for arg_index, argument in enumerate(event["arguments"], 1):
            argument["id"] = f"A_{event_index}_{arg_index}"
    return {
        "schema_version": SCHEMA_VERSION, "sample_id": f"{row['split']}:{row['id']}:{operator}",
        "doc_id": row["id"], "split": row["split"],
        "input": {**{field: row[field] for field in TEXT_FIELDS},
                  "image_path": row["image"]["path"], "events": events},
        "target": hsfp(events, edit),
        "provenance": {"operator": operator, "edit": edit, "evidence": evidence},
    }


def image_file(dataset_dir, row, *, verify_hash=False):
    relative = Path(row["image"]["path"])
    image = (dataset_dir / relative).resolve()
    if relative.is_absolute() or not image.is_relative_to(dataset_dir) or not image.is_file():
        raise ValueError(f"Missing or invalid image path for {row['id']!r}: {relative}")
    if verify_hash and sha256_file(image) != row["image"]["sha256"]:
        raise ValueError(f"Image content does not match the prepared hash for {row['id']!r}.")


def flag(unit, error_type, feedback):
    unit.update(status="incorrect", error_type=error_type, feedback=feedback)


def extended_pair(row, operator, option, dataset_dir):
    clean = sample(row, "clean")
    if operator == "C1":
        argument = clean["input"]["events"][option["event_index"]]["arguments"][option["argument_index"]]
        argument["bbox"] = option["original"]
        argument["bbox_image_size"] = [row["image"]["width"], row["image"]["height"]]
    perturbed = deepcopy(clean)
    perturbed["sample_id"] = f"{row['split']}:{row['id']}:{operator}"
    events = perturbed["input"]["events"]
    target = perturbed["target"]
    event_index = option.get("event_index", len(events))
    edit = {"event_index": event_index}

    def arg_target(argument):
        return next(unit for unit in target["argument_level"] if unit["argument_id"] == argument["id"])

    if operator == "B1":
        first, second = option["argument_indices"]
        arguments = events[event_index]["arguments"]
        left, right = arguments[first], arguments[second]
        left["argument"], right["argument"] = right["argument"], left["argument"]
        for argument, correct in ((left, right["argument"]), (right, left["argument"])):
            flag(arg_target(argument), "wrong_role", f"将「{argument['role']}」的论元值改为「{correct}」。")
        edit.update(operation="swap_argument_values", argument_indices=[first, second])
    elif operator == "B2":
        event = events[event_index]
        original = event["event_type"]
        event["event_type"] = option["alternative"]
        flag(target["event_level"][event_index], "wrong_type", f"将事件类型改为「{original}」。")
        edit.update(operation="restore_event_type", value=original)
    elif operator == "C1":
        arg_index = option["argument_index"]
        argument = events[event_index]["arguments"][arg_index]
        argument["bbox"] = option["alternative"]
        flag(arg_target(argument), "ungrounded_arg",
             f"将「{argument['argument']}」的图像框改为像素坐标 {option['original']}。")
        edit.update(operation="restore_bbox", argument_index=arg_index, value=option["original"])
    elif operator == "C2":
        donor = option["donor"]
        image_file(dataset_dir, donor, verify_hash=True)
        perturbed["input"]["image_path"] = donor["image"]["path"]
        flag(target["event_level"][event_index], "modal_conflict",
             "正文支持该事件，但图像展示的是另一事件。换用与正文事件相符的图像。")
        edit = {"operation": "restore_image_path", "value": clean["input"]["image_path"]}
    elif operator == "D1":
        event = events[event_index]
        arg_index = len(event["arguments"])
        argument = {"id": f"A_{event_index + 1}_{arg_index + 1}", "role": option["role"],
                    "argument": option["alternative"]}
        event["arguments"].append(argument)
        feedback = {"event_id": event["id"], "argument_id": argument["id"], "role": argument["role"]}
        flag(feedback, "redundant_arg", f"删除缺乏证据的论元「{argument['argument']}」。")
        target["argument_level"].append(feedback)
        edit.update(operation="delete_argument", argument_index=arg_index)
    else:
        event = deepcopy(option["alternative"])
        event["id"] = f"E_{event_index + 1}"
        events.append(event)
        feedback = {"event_id": event["id"]}
        flag(feedback, "redundant_event", f"删除缺乏证据的「{event['event_type']}」事件。")
        target["event_level"].append(feedback)
        for arg_index, argument in enumerate(event["arguments"], 1):
            argument["id"] = f"A_{event_index + 1}_{arg_index}"
            target["argument_level"].append({
                "event_id": event["id"], "argument_id": argument["id"], "role": argument["role"],
                "status": "correct", "error_type": None,
                "feedback": "论元值有文本依据。删除该事件时一并移除此论元。",
            })
        edit.update(operation="delete_event")

    target["final_verification"] = "partial"
    perturbed["provenance"] = {"operator": operator, "edit": edit, "evidence": option["evidence"]}
    return clean, perturbed


def build_split(rows, split, limit, seed, dataset_dir, reviews=None, operators=PILOT_OPERATORS):
    rng = random.Random(f"{seed}:{split}")
    shuffled = list(rows)
    rng.shuffle(shuffled)
    pools = {operator: [] for operator in operators}
    rejected = {operator: Counter() for operator in operators}
    extended = [operator for operator in operators if operator in EXTENDED_OPERATORS]
    scenes = {}
    if "C2" in extended:
        for row in shuffled:
            scene = image_scene(row)
            if scene is not None:
                scenes[row["id"]] = (row, scene)
    for row in shuffled:
        if "A1" in operators or "A2" in operators:
            review = None if reviews is None or "A2" not in operators else reviews.get(row["id"], {})
            a1, a2, reason = deletion_options(row, review)
            for operator, options, why in (("A1", a1, "no_unique_textual_argument"), ("A2", a2, reason)):
                if operator in operators:
                    if options:
                        pools[operator].append((row, options))
                    else:
                        rejected[operator][why] += 1
        for operator in extended:
            options, reason = extended_options(operator, row, scenes, argument_evidence)
            if options:
                pools[operator].append((row, options))
            else:
                rejected[operator][reason] += 1

    groups = set()
    a2_selected = take_options(pools.get("A2", []), limit, groups, rng)
    a1_selected = select_balanced_a1(pools.get("A1", []), limit, groups, rng)
    eligible_groups = {operator: {row["group_id"] for row, _ in pool} for operator, pool in pools.items()}

    examples = []
    for operator, selected in (("A1", a1_selected), ("A2", a2_selected)):
        for row, deletion in selected:
            image_file(dataset_dir, row)
            examples.extend((sample(row, "clean"), sample(row, operator, deletion)))
    for operator in sorted(extended, key=lambda name: len(eligible_groups[name] - groups)):
        select = select_image_swaps if operator == "C2" else take_options
        for row, option in select(pools[operator], limit, groups, rng):
            image_file(dataset_dir, row, verify_hash=operator in ("C1", "C2", "D1", "D2"))
            examples.extend(extended_pair(row, operator, option, dataset_dir))
    rng.shuffle(examples)
    if not examples and operators == PILOT_OPERATORS:
        raise ValueError(f"No eligible pilot examples in {split}.")

    selected_counts = Counter(example["provenance"]["operator"] for example in examples)
    audit = {
        "source_documents": len(rows), "requested_per_operator": limit, "samples": len(examples),
        "clean": selected_counts["clean"],
        "operators": {
            operator: {
                "attempted_documents": len(rows), "eligible_documents": len(pools[operator]),
                "eligible_groups": len(eligible_groups[operator]),
                "selected": selected_counts[operator], "rejected": dict(sorted(rejected[operator].items())),
            }
            for operator in operators
        },
        "a1_evidence_sources": dict(Counter(deletion[2][0]["source"] for _, deletion in a1_selected)),
        "a1_event_types": dict(Counter(row["event_list"][deletion[0]]["event_type"]
                                       for row, deletion in a1_selected)),
        "a1_roles": dict(Counter(deletion[2][0]["role"] for _, deletion in a1_selected)),
    }
    return examples, audit


def generate(args, run):
    documents = load_inputs(args.data_dir, run)
    reviews = load_a2_reviews(args.a2_review, documents, run) if args.a2_review else None
    splits, report = {}, {}
    limits = {"train": args.train_per_operator, "dev": args.dev_per_operator}
    for split in PILOT_SPLITS:
        splits[split], report[split] = build_split(
            documents[split], split, limits[split], args.seed, args.dataset_dir,
            None if reviews is None else reviews[split], args.operators,
        )
        for operator, stats in report[split]["operators"].items():
            run.metric(split=split, operator=operator, **stats)
            run.logger.info("%s %s attempted=%d eligible=%d selected=%d rejected=%s",
                            split, operator, stats["attempted_documents"], stats["eligible_documents"],
                            stats["selected"], stats["rejected"])
            if stats["selected"] < limits[split]:
                run.logger.warning("%s %s selected %d of %d requested", split, operator,
                                   stats["selected"], limits[split])
        run.metric(split=split, operator="clean", selected=report[split]["clean"])
        run.logger.info("%s samples=%d clean=%d", split, report[split]["samples"], report[split]["clean"])

    write_json(run.path / "spp_audit.json", {
        "schema_version": SCHEMA_VERSION, "seed": args.seed, "scope": args.profile,
        "evidence_policy": ("heuristic_text_spans_and_role_checks" if args.operators == PILOT_OPERATORS
                            else "text_spans_role_cues_visual_annotations_and_same_split_image_donors"),
        "a2_primary_policy": "reviewed_only" if reviews is not None else "literal_title_anchor",
        "splits": report,
    })
    run.include_in_archive("spp_audit.json")
    for split, examples in splits.items():
        if not examples:
            raise ValueError(f"No eligible SPP examples in {split}; see spp_audit.json.")
    for split, examples in splits.items():
        with (run.path / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as file:
            for example in examples:
                file.write(json.dumps(example, ensure_ascii=False) + "\n")
    run.update_manifest(outputs={
        name: sha256_file(run.path / name) for name in ("train.jsonl", "dev.jsonl", "spp_audit.json")
    })
    run.summary["samples_by_split"] = {split: len(examples) for split, examples in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "datasets" / "processed" / "cmmevent")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT / "datasets" / "CMMEvent")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT / "runs")
    parser.add_argument("--archives-dir", type=Path, default=PROJECT / "archives")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot", help="Enable A1/A2 or all eight operators.")
    parser.add_argument("--operators", choices=OPERATORS, nargs="+", help="Explicit operator list overriding the profile.")
    parser.add_argument("--train-per-operator", type=int, default=50, help="Train cap per operator (default 50).")
    parser.add_argument("--dev-per-operator", type=int, default=15, help="Dev cap per operator (default 15).")
    parser.add_argument("--a2-review", type=Path, help="SHA-256-bound A2 review; use only approved documents for A2.")
    args = parser.parse_args()
    if args.operators:
        args.operators = tuple(operator for operator in OPERATORS if operator in args.operators)
        args.profile = "pilot" if args.operators == PILOT_OPERATORS else "custom"
    else:
        args.operators = OPERATORS if args.profile == "full" else PILOT_OPERATORS
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    try:
        with RunLog("spp", args.runs_dir, args.archives_dir, config, manifest={"accelerator": "cpu"}) as run:
            if min(args.train_per_operator, args.dev_per_operator) < 1:
                raise ValueError("Operator limits must be positive.")
            run.update_manifest(code={name: sha256_file(PROJECT / name)
                                     for name in ("build_spp.py", "spp_operators.py", "run_logging.py")})
            generate(args, run)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"SPP build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
