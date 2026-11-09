"""Prepare WildMMEvent text and image manifests on CPU with Pillow.

The source filenames are document IDs and visual annotation keys. The text
offsets are half-open Python string spans. Visual annotations keep their
original boxes in raw_bb and expose usable pixel xyxy boxes in bb.
"""

from collections import defaultdict
from pathlib import PurePosixPath

from data_preprocess import (
    IMAGE_SUFFIXES,
    SPLIT_RATIOS,
    counts,
    generate_split,
    group_documents,
    join_image,
    normalize_events,
    normalize_visual,
    read_json,
    stem,
    supplied_split,
    write_outputs,
)


def resolve_image(row, candidates, dataset_dir, exclusions, issues, decoded):
    doc_id = row["id"]
    img_url = row.get("img_url")
    source = PurePosixPath(img_url) if isinstance(img_url, str) else PurePosixPath()
    if source.parts[:1] != ("Image",) or source.name != doc_id or ".." in source.parts:
        exclusions["image_source_mismatch"].append({"id": doc_id, "img_url": img_url})
        return None

    exact_paths = [path for path in candidates if path.relative_to(dataset_dir).as_posix() == source.as_posix()]
    image = join_image(doc_id, exact_paths or candidates, dataset_dir, exclusions, decoded)
    if image is not None and not exact_paths:
        issues["relocated_image"].append({"id": doc_id, "img_url": img_url, "path": image["path"]})
    return image


def prepare(dataset_dir, output_dir, seed, split_file):
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new --output-dir: {output_dir}")

    text_rows = read_json(dataset_dir / "annotation_text.json", reject_duplicates=True)
    visual_rows = read_json(dataset_dir / "annotation_image.json", reject_duplicates=True)
    ids = {row["id"] for row in text_rows}
    if len(ids) != len(text_rows):
        raise ValueError("Duplicate document IDs in annotation_text.json.")

    image_root = dataset_dir / "Image"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")
    images = defaultdict(list)
    for path in sorted(image_root.rglob("*")):
        if path.suffix.casefold() in IMAGE_SUFFIXES and path.is_file():
            images[path.name].append(path)

    exclusions, issues = defaultdict(list), defaultdict(list)
    for name in sorted(visual_rows.keys() - ids):
        issues["unmatched_visual_annotation"].append({"annotation_id": name})
    for name in sorted(images.keys() - ids):
        for path in images[name]:
            issues["unmatched_image_file"].append({"path": path.relative_to(dataset_dir).as_posix()})

    records = []
    decoded = {}
    for row in sorted(text_rows, key=lambda row: row["id"]):
        doc_id = row["id"]
        image = resolve_image(row, images.get(doc_id, []), dataset_dir, exclusions, issues, decoded)
        if image is None:
            continue
        visual = {stem(doc_id): [(doc_id, visual_rows[doc_id])]} if doc_id in visual_rows else {}
        records.append({
            **row,
            "image": image,
            "event_list": normalize_events(row, issues),
            "visual_annotations": normalize_visual(doc_id, image, visual, issues),
        })

    groups = group_documents(records)
    if split_file:
        assignment = supplied_split(split_file, ids, records, reject_duplicates=True)
        for group_id, group in groups.items():
            memberships = sorted({assignment[record["id"]] for record in group})
            if len(memberships) > 1:
                splits = ", ".join(memberships)
                raise ValueError(f"Supplied split puts duplicate group {group_id!r} across splits: {splits}.")
    else:
        assignment = generate_split(groups, seed)

    splits = {split: [] for split in SPLIT_RATIOS}
    for record in records:
        split = assignment[record["id"]]
        record["split"] = split
        splits[split].append(record)

    audit = {
        "schema_version": 1,
        "dataset": "wildmmevent",
        "split_source": "supplied" if split_file else "generated",
        "seed": seed if not split_file else None,
        "target_ratios": SPLIT_RATIOS if not split_file else None,
        "raw": counts(text_rows),
        "usable": counts(records),
        "groups": {
            "total": len(groups),
            "duplicate_groups": sum(len(group) > 1 for group in groups.values()),
            "largest_group": max(map(len, groups.values()), default=0),
        },
        "split_counts": {split: counts(rows) for split, rows in splits.items()},
        "exclusions": dict(exclusions),
        "issues": dict(issues),
    }
    write_outputs(output_dir, splits, audit)
    return audit


if __name__ == "__main__":
    import sys

    from main import main

    raise SystemExit(main(["preprocess", "--dataset", "wildmmevent", *sys.argv[1:]]))
