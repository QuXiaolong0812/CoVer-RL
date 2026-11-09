"""Fine-tune mDeBERTa on event entailment pairs and calibrate on held-out CMMEvent dev."""

from collections import Counter, defaultdict
from importlib.metadata import version
import json
import math
from pathlib import Path
import random

import torch

from cmmevent_io import read_split
from evidence_scoring import CALIBRATION_FILE, TEXT_FIELDS, sigmoid, text_premise, verbalize
from run_logging import sha256_file, write_json


def train_inventory(train_rows):
    """Build negative donors exclusively from the training split."""
    if not train_rows or any(row["split"] != "train" for row in train_rows):
        raise ValueError("Event-NLI negative donors must come from the training split.")
    roles, fillers = defaultdict(set), defaultdict(set)
    for row in train_rows:
        for event in row["event_list"]:
            kind = event["event_type"]
            roles[kind].update(arg["role"] for arg in event["arguments"])
            for arg in event["arguments"]:
                role = arg["role"]
                if textual_filler(row, arg):
                    fillers[kind, role].add((row["id"], arg["argument"]))
    return {
        "types": sorted(roles),
        "roles": {kind: sorted(values) for kind, values in roles.items()},
        "fillers": {key: sorted(values) for key, values in fillers.items()},
    }


def textual_filler(row, arg):
    source = arg.get("source")
    return source in TEXT_FIELDS and bool(arg["argument"]) and arg["argument"] in row.get(source, "")


def build_pairs(rows, inventory, *, split, seed):
    """Create aligned positive claims and hard, one-field counterfactuals."""
    if split not in {"train", "dev"} or any(row["split"] != split for row in rows):
        raise ValueError("Event-NLI pairs may only be built from the named train or dev split.")
    pairs = []
    # Different images can share the same article and caption. For text NLI, protect
    # all annotated claims for an identical premise and train on each pair once.
    gold_by_premise, seen_by_premise = defaultdict(set), defaultdict(set)
    for row in rows:
        gold = gold_by_premise[text_premise(row)]
        for event in row["event_list"]:
            kind, trigger = event["event_type"], event["trigger"]
            gold.add(verbalize(kind, trigger))
            gold.update(verbalize(kind, trigger, arg["role"], arg["argument"]) for arg in event["arguments"])
    for row in rows:
        rng = random.Random(f"{seed}:{split}:{row['id']}")
        premise = text_premise(row)
        gold, seen = gold_by_premise[premise], seen_by_premise[premise]

        def add(hypothesis, label, kind, donor_doc_id=None):
            if hypothesis in seen or (label == 0 and hypothesis in gold):
                return False
            seen.add(hypothesis)
            pairs.append({"doc_id": row["id"], "split": split, "premise": premise,
                          "hypothesis": hypothesis, "label": label, "kind": kind, "donor_doc_id": donor_doc_id})
            return True

        for event in row["event_list"]:
            kind, trigger = event["event_type"], event["trigger"]
            if not trigger or trigger not in premise:
                continue
            if add(verbalize(kind, trigger), 1, "header"):
                alternatives = [candidate for candidate in inventory["types"] if candidate != kind]
                rng.shuffle(alternatives)
                alternatives.sort(key=lambda candidate: candidate.split("-")[0] != kind.split("-")[0])
                for other_type in alternatives:
                    if add(verbalize(other_type, trigger), 0, "wrong_type"):
                        break
            local_roles = {arg["role"] for arg in event["arguments"]}
            for arg in event["arguments"]:
                if not textual_filler(row, arg):
                    continue
                role, filler = arg["role"], arg["argument"]
                if not add(verbalize(kind, trigger, role, filler), 1, "argument"):
                    continue
                alternatives = [candidate for candidate in inventory["roles"].get(kind, []) if candidate != role]
                rng.shuffle(alternatives)
                alternatives.sort(key=lambda candidate: candidate not in local_roles)
                for other_role in alternatives:
                    if add(verbalize(kind, trigger, other_role, filler), 0, "wrong_role"):
                        break
                # Reuse a different same-role mention from this premise where possible.
                local_fillers = [
                    other_arg["argument"]
                    for other_event in row["event_list"] if other_event is not event
                    for other_arg in other_event["arguments"]
                    if other_arg["role"] == role and textual_filler(row, other_arg)
                ]
                rng.shuffle(local_fillers)
                if any(add(verbalize(kind, trigger, role, value), 0, "wrong_filler")
                       for value in local_fillers if value != filler):
                    continue
                donors = inventory["fillers"].get((kind, role), [])
                if donors:
                    offset = rng.randrange(len(donors))
                    for index in range(len(donors)):
                        donor_id, value = donors[(offset + index) % len(donors)]
                        if donor_id != row["id"] and value != filler and add(
                            verbalize(kind, trigger, role, value), 0, "wrong_filler", donor_id
                        ):
                            break
    return pairs


def nli_metrics(logits, labels, temperature):
    probabilities, correct, nll = [], [], 0.0
    true_by_class, hits_by_class, pred_by_class = Counter(labels), Counter(), Counter()
    for logit, target in zip(logits, labels):
        scaled = logit / temperature
        nll += max(scaled, 0) - scaled * target + math.log1p(math.exp(-abs(scaled)))
        probability = sigmoid(scaled)
        prediction = int(probability >= 0.5)
        pred_by_class[prediction] += 1
        hits_by_class[target] += int(prediction == target)
        probabilities.append(probability)
        correct.append(prediction == target)
    brier = sum((probability - target) ** 2 for probability, target in zip(probabilities, labels)) / len(labels)
    count, confidence, outcomes = Counter(), Counter(), Counter()
    for probability, target in zip(probabilities, labels):
        index = min(9, int(10 * probability))
        count[index] += 1
        confidence[index] += probability
        outcomes[index] += target
    ece = sum(abs(confidence[index] - outcomes[index]) for index in count) / len(labels)
    macro_f1 = sum(2 * hits_by_class[label] / (true_by_class[label] + pred_by_class[label])
                   for label in (0, 1) if true_by_class[label] + pred_by_class[label]) / 2
    return {"nll": nll / len(labels), "brier": brier, "ece_10": ece,
            "accuracy": sum(correct) / len(labels), "macro_f1": macro_f1}


def fit_temperature(logits, labels):
    if len(logits) != len(labels) or set(labels) != {0, 1} or not all(math.isfinite(x) for x in logits):
        raise ValueError("Dev calibration needs finite logits with both binary event-NLI labels.")
    # Includes temperature 1 and both predeclared bounds. Tie-break towards 1.
    temperatures = [2 ** (index / 20) for index in range(-40, 41)]
    candidates = [(nli_metrics(logits, labels, temperature), temperature) for temperature in temperatures]
    calibrated, temperature = min(candidates, key=lambda item: (item[0]["nll"], abs(math.log(item[1]))))
    return temperature, {"uncalibrated": nli_metrics(logits, labels, 1.0), "calibrated": calibrated}


def _collate(pairs, tokenizer, max_length):
    encoded = tokenizer(
        [pair["premise"] for pair in pairs], [pair["hypothesis"] for pair in pairs],
        padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    )
    labels = torch.tensor([pair["label"] for pair in pairs], dtype=torch.float32)
    return encoded, labels


@torch.inference_mode()
def _dev_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for encoded, targets in loader:
        logits.extend(model(**encoded.to(device)).logits.flatten().float().cpu().tolist())
        labels.extend(targets.int().tolist())
    if not all(math.isfinite(logit) for logit in logits):
        raise ValueError("Dev event-NLI logits were nonfinite.")
    return logits, labels


def run_nli(args, run):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    model_dir, processed_dir = Path(args.model_dir), Path(args.processed_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing local mDeBERTa-v3 base: {model_dir}")
    seed = getattr(args, "seed", 42)
    epochs = getattr(args, "epochs", 3)
    batch_size = getattr(args, "batch_size", 8)
    max_length = getattr(args, "max_length", 384)
    learning_rate = getattr(args, "learning_rate", 2e-5)
    train_limit = getattr(args, "train_limit", None)
    dev_limit = getattr(args, "dev_limit", None)
    if (min(epochs, batch_size) < 1 or not 8 <= max_length <= 512 or not 0 < learning_rate < 1
            or any(limit is not None and limit < 1 for limit in (train_limit, dev_limit))):
        raise ValueError("Invalid event-NLI training settings.")
    device = torch.device(getattr(args, "device", "cpu"))
    torch.manual_seed(seed)
    rows, digests = {}, {}
    for split in ("train", "dev"):
        path = processed_dir / f"{split}.jsonl"
        rows[split], digests[str(path)] = read_split(path, split)
    for extract in (lambda row: row["id"], lambda row: row.get("group_id"),
                    lambda row: row.get("image", {}).get("sha256"), text_premise):
        if ({extract(row) for row in rows["train"]} & {extract(row) for row in rows["dev"]}) - {None, ""}:
            raise ValueError("Event-NLI train and dev share a document, group, image or text premise.")
    run.update_manifest(
        inputs=digests, base_model_dir=str(model_dir.resolve()), device=str(device),
        base_model_files={path.name: sha256_file(path) for path in sorted(model_dir.iterdir())
                          if path.is_file() and path.suffix in (".json", ".model", ".bin", ".safetensors")},
        packages={"torch": torch.__version__,
                  **{name: version(name) for name in ("transformers", "sentencepiece", "huggingface-hub")}},
        train_donor_documents=len(rows["train"]),
    )
    inventory = train_inventory(rows["train"])
    pairs = {}
    for split, limit in (("train", train_limit), ("dev", dev_limit)):
        selected = rows[split][:limit]
        pairs[split] = build_pairs(selected, inventory, split=split, seed=seed)
        if {pair["label"] for pair in pairs[split]} != {0, 1}:
            raise ValueError(f"Event-NLI {split} needs both entailment and non-entailment pairs.")
        path = run.path / f"{split}_pairs.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as file:
            for pair in pairs[split]:
                file.write(json.dumps(pair, ensure_ascii=False, allow_nan=False) + "\n")
        run.metric(split=split, documents=len(selected), pairs=len(pairs[split]),
                   pair_kinds=dict(Counter(pair["kind"] for pair in pairs[split])))
        run.logger.info("Event-NLI %s: %d pairs from %d documents", split, len(pairs[split]), len(selected))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=1, id2label={0: "ENTAILMENT"}, label2id={"ENTAILMENT": 0},
        local_files_only=True,
    ).to(device)
    if model.config.model_type != "deberta-v2":
        raise ValueError("Event-NLI expects an mDeBERTa-v3 base checkpoint.")
    loaders = {
        split: torch.utils.data.DataLoader(
            pairs[split], batch_size=batch_size, shuffle=split == "train",
            collate_fn=lambda batch: _collate(batch, tokenizer, max_length),
            generator=torch.Generator().manual_seed(seed),
        ) for split in ("train", "dev")
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(loaders["train"]) * epochs
    schedule = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
    checkpoint = run.path / "nli_checkpoint"
    best_nll = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for step, (encoded, targets) in enumerate(loaders["train"], 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded.to(device)).logits.flatten()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite event-NLI train loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            loss_sum += loss.item() * len(targets)
            seen += len(targets)
            if step % 100 == 0:
                run.metric(split="train", epoch=epoch, step=step, mean_loss=loss_sum / seen)
        logits, labels = _dev_logits(model, loaders["dev"], device)
        metrics = nli_metrics(logits, labels, 1.0)
        run.metric(split="dev", epoch=epoch, train_loss=loss_sum / seen, **metrics)
        run.logger.info("Event-NLI epoch %d train_loss=%.4f dev_nll=%.4f dev_macro_f1=%.4f",
                        epoch, loss_sum / seen, metrics["nll"], metrics["macro_f1"])
        if metrics["nll"] < best_nll:
            best_epoch, best_nll, best_logits, best_labels = epoch, metrics["nll"], logits, labels
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

    temperature, calibration_metrics = fit_temperature(best_logits, best_labels)
    calibration = {
        "task": "cmmevent_event_nli", "train_split": "train", "calibration_split": "dev",
        "checkpoint_selection": "minimum_uncalibrated_dev_nll", "best_epoch": best_epoch,
        "temperature": temperature, "max_length": max_length, "seed": seed,
        "train_pairs": len(pairs["train"]), "dev_pairs": len(pairs["dev"]), **calibration_metrics,
    }
    write_json(checkpoint / CALIBRATION_FILE, calibration)
    write_json(run.path / CALIBRATION_FILE, calibration)
    run.include_in_archive(CALIBRATION_FILE)
    output_files = [run.path / name for name in ("train_pairs.jsonl", "dev_pairs.jsonl", CALIBRATION_FILE)]
    output_files.extend(path for path in sorted(checkpoint.iterdir()) if path.is_file())
    run.update_manifest(outputs={path.relative_to(run.path).as_posix(): sha256_file(path) for path in output_files})
    run.summary.update(nli_checkpoint=str(checkpoint), **calibration)
    run.metric(split="dev", event="calibration", **calibration)
    return checkpoint
