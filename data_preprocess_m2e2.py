"""Prepare the released M2E2 annotations as a held-out document manifest.

Each sentence groups its native annotation rows by sentence ID. Its events
combine identical targets while retaining their source memberships. Text
annotation offsets refer to the native annotation rendition, which can differ
from the downloaded article in `text`.

Image and article paths are relative to the dataset directory. Unavailable
assets have null metadata. Records with only raw assets have has_annotations
set to false. Visual `role` fields preserve the native tuples.
`boxes_xyxy` parallels each role's tuples and exposes usable pixel boxes.
Cross-media links preserve the release's sentence ID, image filename, and
event type, with no implied correspondence to a particular event mention.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from data_preprocess import IMAGE_SUFFIXES, file_hash, join_image, read_json, valid_box, write_outputs


TEXT_FILES = ("text_only_event.json", "text_multimedia_event.json")
IMAGE_FILES = ("image_only_event.json", "image_multimedia_event.json")
COREF_FILE = "crossmedia_coref.txt"
ANNOTATION_FILES = (*TEXT_FILES, *IMAGE_FILES, COREF_FILE)
ARTICLE_SUFFIX = ".rsd.txt"
MEMBER_ID = re.compile(r"([^/\\]+)_([0-9]+)\Z")


def resolve_annotations_dir(dataset_dir, annotations_dir):
    if annotations_dir is not None:
        directory = Path(annotations_dir).resolve()
        missing = [name for name in ANNOTATION_FILES if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing M2E2 annotation files in {directory}: {', '.join(missing)}")
        return directory

    candidates = (
        dataset_dir / "m2e2_annotations",
        dataset_dir / "data" / "m2e2_annotations",
        dataset_dir / "m2e2-master" / "data" / "m2e2_annotations",
        dataset_dir / "m2e2-master" / "m2e2-master" / "data" / "m2e2_annotations",
    )
    complete = [path for path in candidates if all((path / name).is_file() for name in ANNOTATION_FILES)]
    if len(complete) > 1:
        raise ValueError("Multiple complete M2E2 annotation directories found. Specify --annotations-dir.")
    if not complete:
        raise FileNotFoundError(f"No complete M2E2 annotation directory found under {dataset_dir}. Specify --annotations-dir.")
    return complete[0]


def load_annotations(path, expected_type):
    data = read_json(path, reject_duplicates=True)
    if not isinstance(data, expected_type):
        raise ValueError(f"Expected {expected_type.__name__} in {path}.")
    return data


def member_document(identifier):
    match = MEMBER_ID.fullmatch(identifier)
    return match[1] if match else None


def image_document(filename):
    name = Path(filename)
    if name.suffix.casefold() in IMAGE_SUFFIXES and name.name == filename:
        return member_document(name.stem)
    return None


def asset_index(root, accepts, key):
    index = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and accepts(path):
            index[key(path)].append(path)
    return index


def read_article(doc_id, paths, dataset_dir, issues):
    aliases = [path.relative_to(dataset_dir).as_posix() for path in paths]
    location = {"id": doc_id, "paths": aliases}
    if not paths:
        issues["missing_article"].append({"id": doc_id})
        return None, None
    if len(paths) > 1:
        issues["duplicate_article_files"].append(location)
    try:
        contents = paths[0].read_bytes()
        if any(path.read_bytes() != contents for path in paths[1:]):
            issues["ambiguous_article"].append(location)
            return None, None
        text = contents.decode("utf-8")
        if not text.strip():
            raise ValueError("Article is empty.")
    except (OSError, ValueError) as exc:
        issues["unreadable_article"].append({**location, "error": str(exc)})
        return None, None
    return {"path": aliases[0], "aliases": aliases, "sha256": hashlib.sha256(contents).hexdigest()}, text


def check_native_offsets(row, article, mismatches, issues):
    location = {"sentence_id": row["sentence_id"], "source": row["source"]}
    start, end = row["sentence_start"], row["sentence_end"]
    words, offsets = row["words"], row["index"]
    if not (type(start) is type(end) is int and 0 <= start <= end):
        issues["invalid_sentence_offsets"].append(location)
        return
    if article is not None and article[start:end + 1] != row["sentence"]:
        mismatches["sentence_rows"] += 1

    if len(words) != len(offsets):
        issues["token_offset_count_mismatch"].append(location)
    invalid_tokens = []
    for token, (word, offset) in enumerate(zip(words, offsets)):
        if (not isinstance(offset, list) or len(offset) != 2 or
                not all(type(value) is int or isinstance(value, str) and value.isdecimal() for value in offset)):
            invalid_tokens.append(token)
            continue
        left, right = map(int, offset)
        if not start <= left <= right <= end:
            invalid_tokens.append(token)
        elif article is not None and article[left:right + 1] != word:
            mismatches["tokens"] += 1
    if invalid_tokens:
        issues["invalid_token_offsets"].append({**location, "token_indices": invalid_tokens})

    mentions = list(row["golden-entity-mentions"])
    for event in row["golden-event-mentions"]:
        mentions.extend([event["trigger"], *event["arguments"]])
    for mention in mentions:
        left, right = mention["start"], mention["end"]
        if not (type(left) is type(right) is int and 0 <= left < right <= len(words)):
            issues["invalid_mention_span"].append({**location, "mention": mention})


def sentence_record(sentence_id, annotations, issues):
    if len(annotations) > 1:
        sources = [row["source"] for row in annotations]
        issues["duplicate_sentence_ids"].append({"sentence_id": sentence_id, "sources": sources})
        if any(row["words"] != annotations[0]["words"] or row["index"] != annotations[0]["index"]
               for row in annotations[1:]):
            issues["conflicting_sentence_tokenization"].append({"sentence_id": sentence_id, "sources": sources})
            # Token-indexed targets cannot be combined across tokenizations.
            return {"sentence_id": sentence_id, "annotations": annotations, "events": None}

    events = {}
    for row in annotations:
        for event in row["golden-event-mentions"]:
            key = json.dumps(event, sort_keys=True, ensure_ascii=False)
            combined = events.setdefault(key, {**event, "sources": []})
            if row["source"] not in combined["sources"]:
                combined["sources"].append(row["source"])
    return {"sentence_id": sentence_id, "annotations": annotations, "events": list(events.values())}


def visual_record(image_id, annotation, asset, issues):
    boxes = {}
    for role, native_boxes in annotation["role"].items():
        boxes[role] = []
        for box in native_boxes:
            xyxy = box[1:] if isinstance(box, list) and len(box) == 5 else []
            # Without an image, check only the geometry of the source tuple.
            size = (asset["width"], asset["height"]) if asset else xyxy[2:]
            usable = len(size) == 2 and valid_box(xyxy, *size)
            if not usable:
                issues["invalid_box"].append({"image_id": image_id, "source": annotation["source"], "role": role, "box": box})
            boxes[role].append(xyxy if usable and asset else None)
    return {**annotation, "boxes_xyxy": boxes}


def load_coref(path, issues):
    links = {}
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            values = raw.split("\t")
            if len(values) != 3 or not all(values) or member_document(values[0]) is None:
                issues["invalid_coref_row"].append({"line": line_number, "raw": raw})
                continue
            sentence_id, image_id, event_type = values
            link = links.setdefault(tuple(values), {
                "sentence_id": sentence_id, "image_id": image_id, "event_type": event_type, "source_lines": [],
            })
            link["source_lines"].append(line_number)
    return list(links.values())


def validate_coref(link, text_by_id, visual_by_id, issues):
    sentence_id, image_id, event_type = link["sentence_id"], link["image_id"], link["event_type"]
    reasons = []
    if not text_by_id.get(sentence_id):
        reasons.append("missing_sentence_annotation")
    elif not any(event["event_type"] == event_type for row in text_by_id[sentence_id]
                 for event in row["golden-event-mentions"]):
        reasons.append("sentence_event_type_mismatch")
    if image_document(image_id) != member_document(sentence_id):
        reasons.append("image_document_mismatch")
    if not visual_by_id.get(image_id):
        reasons.append("missing_visual_annotation")
    elif not any(row["event_type"] == event_type for row in visual_by_id[image_id]):
        reasons.append("image_event_type_mismatch")
    if reasons:
        issues["unresolved_coref"].append({**link, "reasons": reasons})
    if len(link["source_lines"]) > 1:
        issues["duplicate_coref"].append(link)
    return {**link, "resolved": not reasons}


def counts(records):
    sentences = [sentence for record in records for sentence in record["sentences"]]
    images = [image for record in records for image in record["images"]]
    text_events = [event for sentence in sentences for event in sentence["events"] or ()]
    visual_events = [event for image in images for event in image["visual_annotations"]]
    return {
        "documents": len(records),
        "documents_with_annotations": sum(record["has_annotations"] for record in records),
        "documents_with_articles": sum(record["article"] is not None for record in records),
        "multi_event_documents": sum(
            sum(len(sentence["events"] or ()) for sentence in record["sentences"]) > 1 for record in records
        ),
        "sentences": len(sentences),
        "negative_sentences": sum(sentence["events"] == [] for sentence in sentences),
        "text_events": len(text_events),
        "text_arguments": sum(len(event["arguments"]) for event in text_events),
        "images": len(images),
        "available_images": sum(image["asset"] is not None for image in images),
        "visual_events": len(visual_events),
        "visual_boxes": sum(len(boxes) for event in visual_events for boxes in event["role"].values()),
        "crossmedia_links": sum(len(record["crossmedia_coref"]) for record in records),
    }


def prepare(dataset_dir, output_dir, *, annotations_dir=None):
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new --output-dir: {output_dir}")
    annotations_dir = resolve_annotations_dir(dataset_dir, annotations_dir)
    issues = defaultdict(list)
    inputs = {str(annotations_dir / name): file_hash(annotations_dir / name) for name in ANNOTATION_FILES}

    articles = asset_index(dataset_dir / "article", lambda path: path.name.endswith(ARTICLE_SUFFIX),
                           lambda path: path.name.removesuffix(ARTICLE_SUFFIX))
    files = asset_index(dataset_dir / "image", lambda path: path.suffix.casefold() in IMAGE_SUFFIXES,
                        lambda path: path.name)
    for image_id, paths in sorted(files.items()):
        if image_document(image_id) is None:
            issues["invalid_image_filename"].append({"image_id": image_id})
        if len(paths) > 1:
            issues["duplicate_image_files"].append({
                "image_id": image_id, "paths": [path.relative_to(dataset_dir).as_posix() for path in paths],
            })

    captions = {}
    caption_path = dataset_dir / "image" / "image_url_caption.json"
    if caption_path.is_file():
        inputs[str(caption_path)] = file_hash(caption_path)
        for doc_id, doc_captions in load_annotations(caption_path, dict).items():
            for number, caption in doc_captions.items():
                image_id = f"{doc_id}_{number}.jpg"
                if image_document(image_id) != doc_id:
                    raise ValueError(f"Invalid M2E2 caption image ID {image_id!r}.")
                captions[image_id] = caption

    text_by_id, visual_by_id = defaultdict(list), defaultdict(list)
    sentences_by_doc, images_by_doc, links_by_doc = defaultdict(set), defaultdict(set), defaultdict(list)
    for name in TEXT_FILES:
        for row in load_annotations(annotations_dir / name, list):
            sentence_id = row["sentence_id"]
            doc_id = member_document(sentence_id)
            if doc_id is None:
                raise ValueError(f"Invalid M2E2 sentence ID {sentence_id!r} in {name}.")
            sentences_by_doc[doc_id].add(sentence_id)
            text_by_id[sentence_id].append({**row, "source": Path(name).stem})
            for image_id in set(row["image"]):
                if image_document(image_id) == doc_id:
                    images_by_doc[doc_id].add(image_id)
                else:
                    issue = {"sentence_id": sentence_id, "image_id": image_id, "source": Path(name).stem}
                    issues["unresolved_sentence_image"].append(issue)

    annotated_docs = set(sentences_by_doc)
    for name in IMAGE_FILES:
        for annotation_id, row in load_annotations(annotations_dir / name, dict).items():
            doc_id = member_document(annotation_id)
            if doc_id is None:
                raise ValueError(f"Invalid M2E2 image annotation ID {annotation_id!r} in {name}.")
            image_id = f"{annotation_id}.jpg"
            annotated_docs.add(doc_id)
            images_by_doc[doc_id].add(image_id)
            visual_by_id[image_id].append({**row, "source": Path(name).stem, "annotation_id": annotation_id})

    for image_id, annotations in sorted(visual_by_id.items()):
        if len(annotations) > 1:
            issues["duplicate_visual_ids"].append({"image_id": image_id, "sources": [row["source"] for row in annotations]})

    for link in load_coref(annotations_dir / COREF_FILE, issues):
        doc_id = member_document(link["sentence_id"])
        annotated_docs.add(doc_id)
        if image_document(link["image_id"]) == doc_id:
            images_by_doc[doc_id].add(link["image_id"])
        links_by_doc[doc_id].append(validate_coref(link, text_by_id, visual_by_id, issues))

    for image_id in files.keys() | captions.keys():
        doc_id = image_document(image_id)
        if doc_id is not None:
            images_by_doc[doc_id].add(image_id)
    all_docs = annotated_docs | articles.keys() | images_by_doc.keys()
    for doc_id in sorted(all_docs - annotated_docs):
        issues["unannotated_documents"].append({
            "id": doc_id,
            "article_paths": [path.relative_to(dataset_dir).as_posix() for path in articles.get(doc_id, ())],
            "image_ids": sorted(images_by_doc.get(doc_id, ())),
        })

    records, decoded = [], {}
    for doc_id in sorted(all_docs):
        article, text = read_article(doc_id, articles.get(doc_id, ()), dataset_dir, issues)
        mismatches = Counter()
        sentences = []
        for sentence_id in sorted(sentences_by_doc[doc_id], key=lambda value: int(value.rsplit("_", 1)[1])):
            annotations = text_by_id[sentence_id]
            for row in annotations:
                check_native_offsets(row, text, mismatches, issues)
            sentences.append(sentence_record(sentence_id, annotations, issues))
        if mismatches:
            issues["raw_article_offset_mismatch"].append({"id": doc_id, **mismatches})

        images = []
        for image_id in sorted(images_by_doc[doc_id]):
            asset = join_image(image_id, files.get(image_id, ()), dataset_dir, issues, decoded)
            annotations = [visual_record(image_id, row, asset, issues) for row in visual_by_id.get(image_id, ())]
            images.append({
                **captions.get(image_id, {}),
                "image_id": image_id,
                "asset": asset,
                "visual_annotations": annotations,
            })
        records.append({
            "id": doc_id, "split": "test", "has_annotations": doc_id in annotated_docs, "article": article, "text": text,
            "sentences": sentences, "images": images, "crossmedia_coref": links_by_doc[doc_id],
        })

    audit = {
        "schema_version": 1,
        "dataset": "m2e2",
        "split_source": "released_m2e2_annotations_and_assets",
        "inputs": inputs,
        "coordinates": {
            "sentence_start_end_and_index": "Inclusive character offsets in the native annotation rendition",
            "text_mention_spans": "Half-open token spans in the row's words",
            "boxes_xyxy": "Native pixel x1, y1, x2, y2 on the decoded image",
        },
        "raw": {
            "article_files": sum(map(len, articles.values())),
            "image_files": sum(map(len, files.values())),
            "image_captions": len(captions),
            "text_annotation_rows": sum(map(len, text_by_id.values())),
            "visual_annotation_rows": sum(map(len, visual_by_id.values())),
            "crossmedia_annotation_rows": sum(len(link["source_lines"]) for links in links_by_doc.values() for link in links),
        },
        "split_counts": {"test": counts(records)},
        "exclusions": {},
        "issues": dict(issues),
    }
    write_outputs(output_dir, {"test": records}, audit)
    return audit


if __name__ == "__main__":
    import sys

    from main import main

    raise SystemExit(main(["preprocess", "--dataset", "m2e2", *sys.argv[1:]]))
