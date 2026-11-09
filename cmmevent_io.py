"""Read versioned inputs for CMMEvent runs without importing model packages."""

import hashlib
import io
import json


SPLITS = ("train", "dev", "test")


def read_jsonl(path):
    snapshot = path.read_bytes()
    with io.StringIO(snapshot.decode("utf-8-sig"), newline=None) as file:
        rows = [json.loads(line) for line in file]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected nonempty JSONL records in {path}.")
    return rows, hashlib.sha256(snapshot).hexdigest()


def read_split(path, split):
    rows, digest = read_jsonl(path)
    if any(row["split"] != split for row in rows):
        raise ValueError(f"Expected {split} documents in {path}.")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate document IDs in {path}.")
    return rows, digest


def read_generator_predictions(path, split=None):
    rows, digest = read_jsonl(path)
    split = rows[0].get("split") if split is None else split
    if split not in SPLITS or any(row.get("split") != split for row in rows):
        raise ValueError(f"Expected predictions from a single CMMEvent split in {path}.")
    seen = set()
    for row in rows:
        doc_id = row.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id or doc_id in seen:
            raise ValueError(f"Invalid or duplicate prediction document ID {doc_id!r}.")
        seen.add(doc_id)
        valid = row.get("format_valid")
        if type(valid) is not bool:
            raise ValueError(f"Document {doc_id!r} needs a boolean format_valid flag.")
        if valid and (not isinstance(row.get("input"), dict) or not isinstance(row["input"].get("events"), list)):
            raise ValueError(f"Document {doc_id!r} needs an input.events list.")
    return rows, digest, split
