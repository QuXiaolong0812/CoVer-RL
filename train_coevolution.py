"""Interleave CMMEvent generator and verifier updates and select a pair on dev."""

from collections import Counter
import hashlib
import json
import random
from time import perf_counter

import peft
import torch
import transformers

from cmmevent_io import read_split
from evaluate import LABELS, inspect_prediction
from extraction_eval import metrics, record_table_counts
from generate_events import encode_generator, event_schema, inspect_events, select_documents
from grpo import clipped_loss, load_policies, token_terms
from run_logging import sha256_file, write_json
from train_verifier_sft import encode_prompt


def verifier_input(doc, events):
    return {**{key: doc[key] for key in ("title", "text", "description")},
            "image_path": doc["image"]["path"], "events": events}


def parse_rollout(rollout, schema):
    parsed = inspect_events(rollout.raw, schema)
    return parsed["events"] if parsed["format_valid"] else None


def write_trace(trace, **record):
    if trace is not None:
        trace.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        trace.flush()


def generator_rollouts(args, docs, schema, generator, verifier, evidence, progress, alpha, trace, cycle):
    from reward_routing import generator_advantages, local_units, route_local_rewards, structural_score

    batch, counts, outcome_total = [], Counter(), 0.0
    for doc in docs:
        inputs = encode_generator(doc, schema, generator.processor, args.dataset_dir, args.max_image_pixels)
        group, outcomes, returns, local_mask = [], [], [], []
        for _ in range(args.generator_group_size):
            rollout = generator.sample(inputs)
            group.append(rollout)
            events = parse_rollout(rollout, schema)
            token_ids = rollout.stats["completion_token_ids"]
            outcome, local, aligned = args.failure_reward, [0.0] * len(token_ids), False
            alignment, checked, evidence_result = None, None, None
            counts["generator_valid"] += int(events is not None)
            counts["generator_token_limit"] += int(rollout.stats["hit_token_limit"])
            complete = doc.get("reference_complete", True)
            score = structural_score(events, doc["event_list"]) if complete else None
            if events is not None:
                if alpha or evidence is not None:
                    candidate = verifier_input(doc, events)
                    verification = verifier.sample(
                        encode_prompt(candidate, verifier.processor, args.dataset_dir, args.max_image_pixels)
                    )
                    checked = inspect_prediction(verification.raw, events)
                    counts["feedback_valid"] += int(checked["format_valid"])
                    counts["feedback_token_limit"] += int(verification.stats["hit_token_limit"])
                    if checked["format_valid"] and alpha:
                        units = local_units(checked["parsed"])
                        local, alignment = route_local_rewards(
                            rollout.raw, token_ids, generator.processor.tokenizer, units,
                            generator.generation.eos_token_id,
                        )
                        aligned = alignment["ok"]
                        counts["aligned"] += int(aligned)
                        counts["alignment_errors"] += int(not aligned)
                if evidence is not None:
                    evidence_result = evidence.score(doc, events, checked["parsed"] if checked else None, progress)
                    counts["evidence_scored"] += 1
                outcome = score["f1"] if complete else evidence_result["tcr"]
            outcomes.append(outcome)
            outcome_total += outcome
            returns.append(local)
            local_mask.append(aligned)
            write_trace(trace, phase="generator", cycle=cycle, doc_id=doc["id"], raw_output=rollout.raw,
                        **rollout.stats, structural=score, outcome=outcome, alignment=alignment,
                        hsfp=checked, evidence=evidence_result)
        for rollout, advantage in zip(group, generator_advantages(outcomes, returns, alpha, local_mask=local_mask)):
            rollout.advantages = advantage
        counts["outcome_variable_groups"] += int(max(outcomes) > min(outcomes))
        counts["advantage_groups"] += int(any(any(rollout.advantages) for rollout in group))
        batch.extend(group)
    return batch, {**counts, "outcome_mean": outcome_total / len(batch)}


def verifier_rollouts(args, train, rng, schema, generator, verifier, class_counts, trace, cycle):
    from reward_routing import class_weight, standardized, structural_score, verifier_reward

    groups, totals = [], Counter()
    for _ in range(args.groups_per_update):
        for _ in range(args.candidate_attempts):
            doc = rng.choice(train)
            if not doc.get("reference_complete", True):
                continue
            proposal = generator.sample(
                encode_generator(doc, schema, generator.processor, args.dataset_dir, args.max_image_pixels)
            )
            events = parse_rollout(proposal, schema)
            if events is not None:
                break
            totals["malformed_candidates"] += 1
        else:
            totals["skipped_groups"] += 1
            continue
        label = structural_score(events, doc["event_list"])["label"]
        class_counts[label] += 1
        inputs = encode_prompt(verifier_input(doc, events), verifier.processor, args.dataset_dir, args.max_image_pixels)
        group = [verifier.sample(inputs) for _ in range(args.verifier_group_size)]
        rewards = []
        for rollout in group:
            checked = inspect_prediction(rollout.raw, events)
            reward = verifier_reward(checked, label, args.gamma)
            rewards.append(reward)
            totals["feedback_valid"] += int(checked["format_valid"])
            totals["feedback_token_limit"] += int(rollout.stats["hit_token_limit"])
            totals["reward_sum"] += reward
            write_trace(trace, phase="verifier", cycle=cycle, doc_id=doc["id"], candidate=events,
                        true_label=label, reward=reward, raw_output=rollout.raw, **rollout.stats, hsfp=checked)
        totals["advantage_groups"] += int(max(rewards) > min(rewards))
        groups.append((group, label, rewards))
    batch = []
    for group, label, rewards in groups:
        weight = class_weight(class_counts, label)
        for rollout, advantage in zip(group, standardized(rewards)):
            rollout.advantages = [weight * advantage] * rollout.stats["generated_tokens"]
        batch.extend(group)
    reward_sum = totals.pop("reward_sum", 0.0)
    return batch, {**totals, "reward_mean": reward_sum / len(batch) if batch else None}


def validation(args, docs, schema, generator, run, cycle):
    counts = Counter()
    for doc in docs:
        rollout = generator.sample(
            encode_generator(doc, schema, generator.processor, args.dataset_dir, args.max_image_pixels),
            greedy=True,
        )
        events = parse_rollout(rollout, schema)
        counts["format_valid"] += int(events is not None)
        tp, fp, fn = record_table_counts(events or [], doc["event_list"])
        counts.update(tp=tp, fp=fp, fn=fn)
    result = {"documents": len(docs), "format_valid": counts["format_valid"],
              **metrics(counts["tp"], counts["fp"], counts["fn"])}
    run.metric(phase="dev", cycle=cycle, **result)
    run.logger.info("cycle=%d dev_docs=%d format_valid=%d f1=%.4f", cycle, len(docs),
                    counts["format_valid"], result["f1"])
    return result


def checkpoint_files(path):
    return {file.relative_to(path).as_posix(): sha256_file(file) for file in sorted(path.rglob("*"))
            if file.is_file() and (file.suffix in (".safetensors", ".bin", ".json", ".model") or file.name == "vocab.txt")}


def validate_training_splits(train, dev, audit_path):
    fields = {
        "document IDs": lambda row: row["id"],
        "document groups": lambda row: row.get("group_id"),
        "image hashes": lambda row: row["image"].get("sha256"),
    }
    for name, key in fields.items():
        left = {key(row) for row in train} - {None}
        right = {key(row) for row in dev} - {None}
        if left & right:
            raise ValueError(f"Training and validation share {name}.")
    if audit_path.is_file():
        audit = json.loads(audit_path.read_bytes())
        if audit.get("issues", {}).get("group_crosses_supplied_splits"):
            raise ValueError("The preprocessing audit reports document groups crossing splits.")


def save_cycle(path, generator, verifier, devices, rng, state):
    path.mkdir()
    if generator.model is verifier.model:
        generator.model.save_pretrained(path, selected_adapters=["generator", "verifier"], safe_serialization=True)
    else:
        generator.model.save_pretrained(path, selected_adapters=["generator"], safe_serialization=True)
        verifier.model.save_pretrained(path, selected_adapters=["verifier"], safe_serialization=True)
    state.update(adapters=checkpoint_files(path), random=rng.getstate(), torch_random=torch.get_rng_state(),
                 cuda_random={str(device): torch.cuda.get_rng_state(device) for device in devices},
                 generator_optimizer=generator.optimizer.state_dict(), verifier_optimizer=verifier.optimizer.state_dict())
    partial = path / "training_state.tmp"
    torch.save(state, partial)
    partial.replace(path / "training_state.pt")


def train_coevolution(args, run):
    if not torch.cuda.is_available():
        raise RuntimeError("Co-evolution requires a BF16 CUDA GPU.")
    torch.cuda.init()
    devices = sorted({args.generator_device, args.verifier_device})
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    train, train_hash = read_split(args.processed_dir / "train.jsonl", "train")
    dev, dev_hash = read_split(args.processed_dir / "dev.jsonl", "dev")
    validate_training_splits(train, dev, args.processed_dir / "audit.json")
    schema = event_schema(train)
    train = train[:args.train_limit]
    dev_docs = select_documents(dev, args.dev_limit)
    if args.nli_dir is None and any(not doc.get("reference_complete", True) for doc in train):
        raise ValueError("Incomplete training references require the TCR scorers via --nli-dir.")
    directories = {args.generator_model_dir, args.verifier_model_dir, args.adapter_dir}
    if args.nli_dir is not None:
        directories.update((args.nli_dir, args.siglip_dir, args.dino_dir))
    weights = {str(path): checkpoint_files(path) for path in sorted(directories)}
    if any(not files for files in weights.values()):
        raise ValueError("One or more model or adapter directories have no checkpoint files.")
    data = {"train.jsonl": train_hash, "dev.jsonl": dev_hash}
    fingerprint = hashlib.sha256(json.dumps(
        {"data": data, "weights": weights, "configuration": {
            k: v for k, v in run.config.items()
            if k not in {"resume", "stop_after_cycle", "runs_dir", "archives_dir", "config_path", "config_sha256",
                         "trace_rollouts"}
        }}, sort_keys=True,
    ).encode()).hexdigest()
    run.update_manifest(inputs=data, checkpoints=weights, fingerprint=fingerprint,
                        packages={"torch": torch.__version__, "peft": peft.__version__,
                                  "transformers": transformers.__version__},
                        devices={device: torch.cuda.get_device_name(device) for device in devices})
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    state = None
    if args.resume:
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
        if state["fingerprint"] != fingerprint or state["adapters"] != checkpoint_files(args.resume):
            raise ValueError("Resume requires the original configuration, data, initial weights and intact adapters.")
        if state["cycle"] >= args.cycles:
            raise ValueError("The resume checkpoint already reached the configured cycle count.")
        if args.stop_after_cycle is not None and args.stop_after_cycle <= state["cycle"]:
            raise ValueError("stop_after_cycle must be after the resumed cycle.")
    generator, verifier = load_policies(args, args.resume)
    evidence = None
    if args.nli_dir is not None:
        from evidence_scoring import EvidenceScorer

        evidence = EvidenceScorer(args.nli_dir, args.siglip_dir, args.dino_dir, args.dataset_dir,
                                  device=args.evidence_device)
    first, g_step, best, best_pair, stale = 1, 0, -1.0, None, 0
    class_counts = Counter({label: 0 for label in LABELS})
    if state is not None:
        first, g_step = state["cycle"] + 1, state["generator_step"]
        best, best_pair, stale = state["best_f1"], state["best_pair"], state["stale"]
        class_counts.update(state["class_counts"])
        generator.optimizer.load_state_dict(state["generator_optimizer"])
        verifier.optimizer.load_state_dict(state["verifier_optimizer"])
        rng.setstate(state["random"])
        torch.set_rng_state(state["torch_random"])
        for device, rng_state in state["cuda_random"].items():
            torch.cuda.set_rng_state(rng_state, device)
    run.logger.info("GRPO train_docs=%d dev_docs=%d cycles=%d starting_at=%d groups_per_update=%d",
                    len(train), len(dev_docs), args.cycles, first, args.groups_per_update)
    trace = None
    if args.trace_rollouts:
        trace_path = run.path / "rollouts.jsonl"
        trace = trace_path.open("w", encoding="utf-8", errors="backslashreplace", newline="\n")
        run.include_in_archive(trace_path.name)
    options = dict(epochs=args.ppo_epochs, epsilon=args.epsilon, beta=args.beta,
                   token_chunk=args.token_chunk, grad_clip=args.grad_clip)
    total_generator_steps = args.cycles * args.generator_updates
    try:
        for cycle in range(first, args.cycles + 1):
            for index in range(args.generator_updates):
                start = perf_counter()
                progress = g_step / total_generator_steps
                alpha = args.alpha_start * (args.alpha_end / args.alpha_start) ** progress if args.alpha_start else 0.0
                docs = rng.choices(train, k=args.groups_per_update)
                run.logger.info("cycle=%d generator=%d/%d sampling", cycle, index + 1, args.generator_updates)
                batch, details = generator_rollouts(args, docs, schema, generator, verifier, evidence,
                                                     progress, alpha, trace, cycle)
                result = generator.update(batch, **options)
                g_step += 1
                run.metric(phase="generator", cycle=cycle, step=g_step, alpha=alpha, rollouts=len(batch),
                           seconds=round(perf_counter() - start, 3), **details, **result)
                run.logger.info("generator step=%d outcome=%.4f grad_norm=%.5f variable_groups=%d",
                                g_step, details["outcome_mean"], result["grad_norm"],
                                details.get("advantage_groups", 0))
                del batch
            for index in range(args.verifier_updates):
                start = perf_counter()
                run.logger.info("cycle=%d verifier=%d/%d sampling", cycle, index + 1, args.verifier_updates)
                batch, details = verifier_rollouts(args, train, rng, schema, generator, verifier, class_counts,
                                                  trace, cycle)
                result = verifier.update(batch, **options) if batch else {}
                run.metric(phase="verifier", cycle=cycle, update=index + 1, rollouts=len(batch),
                           class_counts=dict(class_counts), seconds=round(perf_counter() - start, 3),
                           **details, **result)
                if batch:
                    run.logger.info("verifier reward=%.4f grad_norm=%.5f variable_groups=%d",
                                    details["reward_mean"], result["grad_norm"], details.get("advantage_groups", 0))
                else:
                    run.logger.warning("Verifier update skipped after exhausting valid-candidate retries.")
                del batch
            cycle_path = run.path / f"cycle-{cycle:04d}"
            if cycle % args.dev_every == 0 or cycle == args.cycles:
                report = validation(args, dev_docs, schema, generator, run, cycle)
                if report["f1"] > best:
                    best, best_pair, stale = report["f1"], str(cycle_path), 0
                else:
                    stale += 1
            save_cycle(cycle_path, generator, verifier, devices, rng,
                       dict(cycle=cycle, generator_step=g_step, best_f1=best, best_pair=best_pair,
                            stale=stale, class_counts=dict(class_counts), fingerprint=fingerprint))
            run.summary.update(cycles_completed=cycle, best_dev_f1=best if best_pair else None, best_pair=best_pair,
                               last_checkpoint=str(cycle_path), class_counts=dict(class_counts))
            run.update_manifest(last_checkpoint={"path": str(cycle_path), **checkpoint_files(cycle_path),
                                                 "training_state.pt": sha256_file(cycle_path / "training_state.pt")})
            run.logger.info("Saved cycle %d checkpoint %s", cycle, cycle_path)
            if args.patience and stale >= args.patience:
                run.logger.info("Dev F1 has plateaued for %d validation checks.", stale)
                break
            if cycle == args.stop_after_cycle:
                run.logger.info("Stopped after the requested cycle %d of %d.", cycle, args.cycles)
                break
    finally:
        if trace is not None:
            trace.close()
    run.summary["peak_allocated_mib"] = {device: round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                                        for device in devices}
    write_json(run.path / "best_checkpoint.json", {"pair": best_pair, "dev_f1": best if best_pair else None})
    run.include_in_archive("best_checkpoint.json")
