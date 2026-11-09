"""Prepare CMMEvent document manifests on CPU with Pillow.

Run `python data_preprocess.py --help` for input and output options.
Image paths in JSONL are relative to --dataset-dir. Event and argument fields
come from the unified event table; visual annotations retain their own roles.
Character spans are half-open Python string offsets in the named source field.
In visual annotations, raw_bb preserves source boxes and bb exposes only boxes
usable on the local image, in pixel xyxy coordinates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
import unicodedata

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_FIELDS = {"des": "description", "tit": "title", "title": "title", "img": "image"}


def read_json(path):
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stem(name):
    return Path(name).stem.casefold()


def image_key(name, verbs):
    name_stem = stem(name)
    prefix, _, rest = name_stem.partition("_")
    if prefix in verbs and rest:
        name_stem = rest
    # CMMEvent spells some version suffixes as "zv1" in the document ID.
    return re.sub(r"(?:z|_)(v[0-9]+)$", r"_\1", name_stem)


def image_index(image_root, verbs):
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")
    index = defaultdict(list)
    for path in sorted(image_root.rglob("*")):
        if path.suffix.casefold() in IMAGE_SUFFIXES and path.is_file():
            index[image_key(path.name, verbs)].append(path)
    return index


def join_image(doc_id, paths, dataset_dir, exclusions, decoded):
    relative_paths = [path.relative_to(dataset_dir).as_posix() for path in paths]
    if not paths:
        exclusions["missing_image"].append({"id": doc_id})
        return None

    try:
        digests = {file_hash(path) for path in paths}
        if len(digests) != 1:
            exclusions["ambiguous_image"].append({"id": doc_id, "paths": relative_paths})
            return None
        digest = digests.pop()
        # Prefer the original filename when available, then a stable relative path.
        selected = min(paths, key=lambda path: (stem(path.name) != stem(doc_id), path.as_posix()))
        if digest not in decoded:
            with Image.open(selected) as image:
                image.load()
                decoded[digest] = image.size
        width, height = decoded[digest]
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        exclusions["unreadable_image"].append(
            {"id": doc_id, "paths": relative_paths, "error": str(exc)}
        )
        return None

    return {
        "path": selected.relative_to(dataset_dir).as_posix(),
        "aliases": relative_paths,
        "sha256": digest,
        "width": width,
        "height": height,
    }


def span_at(text, value, start):
    end = start + len(value)
    if value and 0 <= start < end <= len(text) and text[start:end] == value:
        return [start, end]
    return None


def unique_span(text, value):
    if not value:
        return None
    start = text.find(value)
    if start < 0 or text.find(value, start + 1) >= 0:
        return None
    return [start, start + len(value)]


def normalize_events(row, issues):
    events = []
    for event_index, event in enumerate(row["event_list"]):
        location = {"id": row["id"], "event_index": event_index}
        trigger_span = span_at(row["text"], event["trigger"], event["trigger_start_index"])
        if trigger_span is None:
            issues["invalid_trigger_span"].append(location)

        arguments = []
        for argument_index, argument in enumerate(event["arguments"]):
            marker = argument["argument_start_index"]
            value = argument["argument"]
            source = SOURCE_FIELDS.get(marker, "unknown")
            evidence_span = None
            if type(marker) is int or (isinstance(marker, str) and re.fullmatch(r"[0-9]+", marker)):
                source = "text"
                evidence_span = span_at(row["text"], value, int(marker))
            elif source in ("title", "description"):
                evidence_span = unique_span(row[source], value)

            if source == "unknown" or (source != "image" and evidence_span is None):
                issues["invalid_argument_span"].append(
                    {**location, "argument_index": argument_index, "source": source}
                )
            arguments.append({**argument, "source": source, "evidence_span": evidence_span})

        events.append({**event, "trigger_span": trigger_span, "arguments": arguments})
    return events


def valid_box(box, width, height):
    if not isinstance(box, list) or len(box) != 4:
        return False
    if any(type(coordinate) is not int for coordinate in box):
        return False
    left, top, right, bottom = box
    return 0 <= left < right <= width and 0 <= top < bottom <= height


def normalize_visual(doc_id, image, visual_by_stem, issues):
    names = sorted({stem(path) for path in image["aliases"]})
    annotations = []
    for name in names:
        for annotation_id, visual in visual_by_stem.get(name, ()):
            location = {"id": doc_id, "annotation_id": annotation_id}
            label_size = (visual["width"], visual["height"])
            image_size = (image["width"], image["height"])
            # A one-pixel metadata difference is common. Retain only boxes that
            # fit both sizes without shifting or rescaling the coordinates.
            compatible = max(abs(a - b) for a, b in zip(label_size, image_size)) <= 1
            if label_size != image_size:
                issue = "visual_size_tolerance" if compatible else "visual_size_mismatch"
                issues[issue].append({**location, "annotation_size": label_size, "image_size": image_size})
            boxes = {}
            for role, box in visual["bb"].items():
                boxes[role] = None
                if box == [-1, -1, -1, -1]:
                    continue
                if not valid_box(box, *label_size) or (compatible and not valid_box(box, *image_size)):
                    issues["invalid_box"].append({**location, "role": role, "box": box})
                elif compatible:
                    boxes[role] = box
            annotations.append({**visual, "annotation_id": annotation_id, "raw_bb": visual["bb"], "bb": boxes})
    if not annotations:
        issues["missing_visual_annotation"].append({"id": doc_id})
    return annotations


def group_documents(records):
    """Group by normalized title or identical image, including transitive links."""
    parents = {record["id"]: record["id"] for record in records}
    owners = {}

    def root(doc_id):
        while parents[doc_id] != doc_id:
            parents[doc_id] = parents[parents[doc_id]]
            doc_id = parents[doc_id]
        return doc_id

    for record in records:
        doc_id = record["id"]
        title = " ".join(unicodedata.normalize("NFKC", record["title"]).casefold().split())
        keys = [("image", record["image"]["sha256"])]
        if title:
            keys.append(("title", title))
        for key in keys:
            first = root(owners.setdefault(key, doc_id))
            current = root(doc_id)
            parents[max(first, current)] = min(first, current)

    groups = defaultdict(list)
    for record in records:
        group_id = root(record["id"])
        record["group_id"] = group_id
        groups[group_id].append(record)
    return groups


def split_features(group):
    features = Counter()
    for record in group:
        events = record["event_list"]
        features["multi" if len(events) > 1 else "single"] += 1
        for event_type in {event["event_type"] for event in events}:
            features[f"type:{event_type}"] += 1
    return features


def generate_split(groups, seed):
    """Greedily balance document counts and labels while keeping groups intact."""
    candidates = [(group, split_features(group)) for group in groups.values()]
    totals = Counter()
    for _, features in candidates:
        totals.update(features)
    total_docs = sum(len(group) for group, _ in candidates)
    sizes = Counter()
    label_counts = {split: Counter() for split in SPLIT_RATIOS}
    targets = {split: total_docs * ratio for split, ratio in SPLIT_RATIOS.items()}
    limits = {split: int(target) for split, target in targets.items()}
    by_remainder = sorted(targets, key=lambda split: targets[split] - limits[split], reverse=True)
    for split in by_remainder[:total_docs - sum(limits.values())]:
        limits[split] += 1

    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda item: (min(totals[f] for f in item[1]), -len(item[0])))
    assignment = {}
    for group, features in candidates:
        size = len(group)
        feasible = [split for split in targets if sizes[split] + size <= limits[split]]

        def cost(split):
            ratio = SPLIT_RATIOS[split]
            target = targets[split]
            # Change in squared imbalance, normalized by the expected count.
            size_cost = size * (2 * (sizes[split] - target) + size) / (target + 1)
            label_cost = 0.0
            for feature, count in sorted(features.items()):
                expected = totals[feature] * ratio
                deficit = label_counts[split][feature] - expected
                label_cost += count * (2 * deficit + count) / (expected + 1)
            return size_cost + label_cost / len(features)

        chosen = min(feasible or targets, key=cost)
        sizes[chosen] += size
        label_counts[chosen].update(features)
        for record in group:
            assignment[record["id"]] = chosen
    return assignment


def supplied_split(path, all_ids, records):
    data = read_json(path)
    if not isinstance(data, dict) or set(data) != set(SPLIT_RATIOS):
        raise ValueError("Split file must contain exactly train, dev, and test.")
    assignment = {}
    for split, doc_ids in data.items():
        if not isinstance(doc_ids, list):
            raise ValueError(f"Split {split} must be a list of document IDs.")
        for doc_id in doc_ids:
            if not isinstance(doc_id, str) or doc_id not in all_ids:
                raise ValueError(f"Unknown document ID in split file: {doc_id!r}")
            if doc_id in assignment:
                raise ValueError(f"Duplicate document ID in split file: {doc_id!r}")
            assignment[doc_id] = split
    missing = [record["id"] for record in records if record["id"] not in assignment]
    if missing:
        raise ValueError(f"Split file omits {len(missing)} usable documents, including {missing[0]!r}.")
    return assignment


def counts(records):
    events = [event for record in records for event in record["event_list"]]
    types = Counter(event["event_type"] for event in events)
    return {
        "documents": len(records),
        "single_event_documents": sum(len(record["event_list"]) == 1 for record in records),
        "multi_event_documents": sum(len(record["event_list"]) > 1 for record in records),
        "events": len(events),
        "arguments": sum(len(event["arguments"]) for event in events),
        "event_types": dict(sorted(types.items())),
    }


def write_outputs(output_dir, splits, audit):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cmmevent-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        for split, records in splits.items():
            with (bundle / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as file:
                for record in records:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
        split_ids = {split: [record["id"] for record in records] for split, records in splits.items()}
        for name, data in (("splits.json", split_ids), ("audit.json", audit)):
            with (bundle / name).open("w", encoding="utf-8", newline="\n") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
                file.write("\n")
        bundle.rename(output_dir)


def prepare(dataset_dir, output_dir, seed, split_file):
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new --output-dir: {output_dir}")

    text_rows = read_json(dataset_dir / "annotation_text.json")
    visual_rows = read_json(dataset_dir / "annotation_image.json")
    ids = [row["id"] for row in text_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate document IDs in annotation_text.json.")
    visual_by_stem = defaultdict(list)
    for name, visual in sorted(visual_rows.items()):
        visual_by_stem[stem(name)].append((name, visual))
    verbs = {visual["verb"].casefold() for visual in visual_rows.values()}
    images = image_index(dataset_dir / "image" / "img", verbs)

    exclusions, issues = defaultdict(list), defaultdict(list)
    records = []
    decoded = {}
    for row in sorted(text_rows, key=lambda row: row["id"]):
        doc_id = row["id"]
        image = join_image(doc_id, images.get(image_key(doc_id, verbs), []), dataset_dir, exclusions, decoded)
        if image is None:
            continue
        records.append({
            "id": doc_id,
            "title": row["title"],
            "text": row["text"],
            "description": row["description"],
            "image": image,
            "event_list": normalize_events(row, issues),
            "visual_annotations": normalize_visual(doc_id, image, visual_by_stem, issues),
        })

    groups = group_documents(records)
    assignment = supplied_split(split_file, set(ids), records) if split_file else generate_split(groups, seed)
    for group_id, group in groups.items():
        memberships = sorted({assignment[record["id"]] for record in group})
        if len(memberships) > 1:
            issues["group_crosses_supplied_splits"].append({"group_id": group_id, "splits": memberships})
    splits = {split: [] for split in SPLIT_RATIOS}
    for record in records:
        split = assignment[record["id"]]
        record["split"] = split
        splits[split].append(record)

    audit = {
        "schema_version": 1,
        "split_source": "supplied" if split_file else "generated",
        "seed": seed if not split_file else None,
        "target_ratios": SPLIT_RATIOS if not split_file else None,
        "raw": counts(text_rows),
        "usable": counts(records),
        "split_counts": {split: counts(rows) for split, rows in splits.items()},
        "exclusions": dict(exclusions),
        "issues": dict(issues),
    }
    write_outputs(output_dir, splits, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(
        description="Align, audit, and split CMMEvent on CPU. Requires Pillow.",
        epilog="Outputs train/dev/test JSONL, splits.json, and audit.json in a new directory.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_DIR / "datasets" / "CMMEvent")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "datasets" / "processed" / "cmmevent")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the grouped 8:1:1 split (default: 42).")
    parser.add_argument("--split-file", type=Path, help="JSON mapping train/dev/test to lists of original document IDs.")
    args = parser.parse_args()

    try:
        audit = prepare(args.dataset_dir.resolve(), args.output_dir.resolve(), args.seed, args.split_file)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Preprocessing failed: {exc}\n")

    for split, stats in audit["split_counts"].items():
        print(f"{split}: {stats['documents']} documents, {stats['multi_event_documents']} multi-event")
    excluded = sum(len(items) for items in audit["exclusions"].values())
    print(f"Excluded {excluded} documents. Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
