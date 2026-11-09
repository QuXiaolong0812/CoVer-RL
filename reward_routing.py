"""Labeled CoVer-RL rewards and token credit assignment (Appendices A.2 and A.4)."""

from collections import Counter, defaultdict
import json
from math import fsum, inf, sqrt
import re
from unicodedata import normalize


LABELS = ("correct", "partial", "incorrect")
DELTA = 1e-8
# Keep the accepted envelope in sync with generate_events.inspect_events.
JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", re.IGNORECASE | re.DOTALL)
DECODER = json.JSONDecoder()


def _canonical(text):
    """Preserve case and punctuation after NFC and Unicode whitespace collapsing."""
    return " ".join(normalize("NFC", text).split())


def _group_atoms(events):
    by_type = defaultdict(list)
    count = 0
    for event in events:
        fillers = Counter(
            (_canonical(arg["role"]), _canonical(arg["argument"])) for arg in event["arguments"]
        )
        by_type[_canonical(event["event_type"])].append((_canonical(event["trigger"]), fillers))
        count += 1 + len(event["arguments"])
    return by_type, count


def _maximum_weight(weights):
    """Rectangular Hungarian assignment, with no more rows than columns."""
    rows, columns = len(weights), len(weights[0])
    if rows == 1:
        return max(weights[0])
    row_potential, col_potential = [0] * (rows + 1), [0] * (columns + 1)
    row_at_col, previous_col = [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        row_at_col[0] = row
        distance, visited = [inf] * (columns + 1), [False] * (columns + 1)
        column = 0
        while True:
            visited[column] = True
            current_row = row_at_col[column]
            step, next_column = inf, 0
            for other in range(1, columns + 1):
                if visited[other]:
                    continue
                cost = -weights[current_row - 1][other - 1] - row_potential[current_row] - col_potential[other]
                if cost < distance[other]:
                    distance[other], previous_col[other] = cost, column
                if distance[other] < step:
                    step, next_column = distance[other], other
            for other in range(columns + 1):
                if visited[other]:
                    row_potential[row_at_col[other]] += step
                    col_potential[other] -= step
                else:
                    distance[other] -= step
            column = next_column
            if not row_at_col[column]:
                break
        while column:
            previous = previous_col[column]
            row_at_col[column] = row_at_col[previous]
            column = previous
    return sum(weights[row - 1][column - 1] for column, row in enumerate(row_at_col[1:], 1) if row)


def structural_score(events, gold):
    """Match events within type, maximizing header and filler matches without replacement.

    A header matches on type and trigger. An aligned pair can share fillers even if its
    triggers differ. Fillers retain role, value, and multiplicity. IDs and order do not
    contribute to the score. Pass None for a malformed candidate.
    """
    if events is None:
        return {"f1": None, "label": "malformed", "tp": None, "fp": None, "fn": None}
    predicted, predicted_count = _group_atoms(events)
    reference, reference_count = _group_atoms(gold)
    tp = 0
    for kind in predicted.keys() & reference.keys():
        left, right = predicted[kind], reference[kind]
        if len(left) > len(right):
            left, right = right, left
        tp += _maximum_weight([
            [int(trigger == target) + sum((fillers & expected).values()) for target, expected in right]
            for trigger, fillers in left
        ])
    fp, fn = predicted_count - tp, reference_count - tp
    f1 = 2 * tp / (predicted_count + reference_count) if predicted_count + reference_count else 1.0
    label = ("partial" if tp else "incorrect") if fp or fn else "correct"
    return {"f1": f1, "label": label, "tp": tp, "fp": fp, "fn": fn}


def verifier_reward(checked, true_label, gamma=0.8):
    """Eq. (33), with complete HSFP validation performed by the caller."""
    if not checked["format_valid"]:
        return 0.0
    return gamma + int(checked["parsed"]["final_verification"] == true_label)


def local_units(parsed):
    """Eq. (19), with both omissions and unresolved judgments in the level budgets."""
    if parsed is None:
        return []
    units = []
    for level, budget in (("event", 0.4), ("argument", 0.6)):
        records = parsed[f"{level}_level"]
        for record in records:
            event_id, argument_id = record["event_id"], record.get("argument_id")
            omission = event_id is None if level == "event" else argument_id is None
            status = record["status"].strip().lower()
            if omission or record["error_type"] is not None or status in {"incorrect", "erroneous", "wrong"}:
                judgment = -1
            else:
                judgment = int(status in {"correct", "supported"})
            units.append({
                "level": level, "event_id": event_id, "argument_id": argument_id,
                "omission": omission, "reward": budget * judgment / len(records),
            })
    return units


def _skip_space(text, start):
    while start < len(text) and text[start] in " \t\r\n":
        start += 1
    return start


def _members(text, start, opener):
    """Yield each immediate JSON member's key/index and raw half-open value span."""
    if text[start:start + 1] != opener:
        raise ValueError(f"Expected {opener!r} at character {start}.")
    closer = "}" if opener == "{" else "]"
    start = _skip_space(text, start + 1)
    index = 1
    while text[start:start + 1] != closer:
        if opener == "{":
            key, end = DECODER.raw_decode(text, start)
            colon = _skip_space(text, end)
            if text[colon:colon + 1] != ":":
                raise ValueError(f"Expected ':' at character {colon}.")
            start = _skip_space(text, colon + 1)
        else:
            key = index
        _, end = DECODER.raw_decode(text, start)
        yield key, (start, end)
        start = _skip_space(text, end)
        if text[start:start + 1] == closer:
            break
        if text[start:start + 1] != ",":
            raise ValueError(f"Expected ',' at character {start}.")
        start = _skip_space(text, start + 1)
        index += 1


def _unit_ends(raw):
    """Get exclusive raw character ends for headers and arguments of a validated candidate."""
    stripped = raw.strip()
    start = len(raw) - len(raw.lstrip())
    fence = JSON_FENCE.fullmatch(stripped)
    if fence:
        start += fence.start(1)
    root = dict(_members(raw, _skip_space(raw, start), "{"))
    ends = {}
    for event_index, (event_start, _) in _members(raw, root["events"][0], "["):
        fields = dict(_members(raw, event_start, "{"))
        event_id = f"E_{event_index}"
        ends[event_id, None] = max(fields["event_type"][1], fields["trigger"][1])
        for arg_index, (_, end) in _members(raw, fields["arguments"][0], "["):
            ends[event_id, f"A_{event_index}_{arg_index}"] = end
    return ends


def _decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _token_indices(raw, ids, tokenizer, ends):
    if not ends:
        return {}, "omissions_only"
    try:
        encoded = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) == ids:
            offsets = encoded["offset_mapping"]
            positions = {}
            for index, (start, end) in enumerate(offsets):
                for target in ends:
                    if start < target <= end:
                        positions.setdefault(target, index)
            if len(positions) == len(ends):
                return positions, "verified_reencoding"
    except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
        # Slow tokenizers may not provide offsets. Decode the sampled prefixes instead.
        pass

    # Sampling can produce a different BPE segmentation of the exact same string.
    # Ignore incomplete UTF-8 prefixes, which can decode with replacement characters.
    pending = sorted(ends, reverse=True)
    positions = {}
    for index in range(len(ids)):
        prefix = _decode(tokenizer, ids[:index + 1])
        if raw.startswith(prefix):
            while pending and pending[-1] <= len(prefix):
                positions[pending.pop()] = index
            if not pending:
                return positions, "sampled_prefixes"
    raise ValueError("Could not align a semantic unit to an exact sampled prefix.")


def route_local_rewards(raw, token_ids, tokenizer, units, eos_ids):
    """Route Eq. (19) rewards and compute Eq. (20) causal returns on sampled tokens.

    Pass validated generator text and the original completion IDs, including a
    sampled EOS. An exclusive character end maps to the token containing the final
    header field's closing quote or the argument object's closing brace. If generation
    hit a length limit, omissions use the last observed token. Failure returns zero
    rewards and a diagnostic so callers can mask this channel.
    """
    ids = list(token_ids)
    if eos_ids is None or isinstance(eos_ids, int):
        eos_ids = (eos_ids,)
    ended_by_eos = bool(ids and ids[-1] in eos_ids)
    content = ids[:-1] if ended_by_eos else ids
    terminal = "eos" if ended_by_eos else "last_observed_token" if ids else None
    sparse = [0.0] * len(ids)
    details = {"ok": True, "error": None, "ended_by_eos": ended_by_eos, "terminal_placement": terminal,
               "placements": [], "method": None}
    if not units:
        return sparse, {**details, "method": "no_units"}
    try:
        if not ids or _decode(tokenizer, content) != raw:
            raise ValueError("Sampled content IDs do not decode to the supplied raw completion.")
        present = [unit for unit in units if not unit["omission"]]
        ends = _unit_ends(raw) if present else {}
        boundaries = {ends[unit["event_id"], unit["argument_id"]] for unit in present}
        positions, method = _token_indices(raw, content, tokenizer, boundaries)
        placements = []
        for unit in units:
            char_end = None if unit["omission"] else ends[unit["event_id"], unit["argument_id"]]
            index = len(ids) - 1 if char_end is None else positions[char_end]
            sparse[index] += unit["reward"]
            placements.append({**unit, "char_end": char_end, "token_index": index})
    except (KeyError, TypeError, ValueError) as exc:
        return [0.0] * len(ids), {**details, "ok": False, "error": str(exc)}
    returns = sparse.copy()
    for index in range(len(returns) - 2, -1, -1):
        returns[index] += returns[index + 1]
    return returns, {**details, "method": method, "placements": placements, "sparse_rewards": sparse}


def standardized(values, delta=DELTA):
    """Eq. (21) and (27) use population variance with delta inside the square root."""
    if not values:
        return []
    shifted = [value - values[0] for value in values]
    mean = fsum(shifted) / len(shifted)
    centered = [value - mean for value in shifted]
    denominator = sqrt(fsum(value ** 2 for value in centered) / len(centered) + delta)
    return [value / denominator for value in centered]


def generator_advantages(outcomes, returns, alpha, *, local_mask=None, delta=DELTA):
    """Mix outcome and sequence-balanced local advantages as in Eq. (21)-(23)."""
    if len(outcomes) != len(returns) or (local_mask is not None and len(local_mask) != len(returns)):
        raise ValueError("Expected one outcome, local return sequence and optional mask per rollout.")
    mask = [True] * len(returns) if local_mask is None else local_mask
    active = [sequence for sequence, enabled in zip(returns, mask) if enabled and sequence]
    origin = active[0][0] if active else 0.0
    mean = variance = 0.0
    if active:
        mean = fsum(fsum(value - origin for value in seq) / len(seq) for seq in active) / len(active)
        variance = fsum(
            fsum((value - origin - mean) ** 2 for value in seq) / len(seq) for seq in active
        ) / len(active)
    denominator = sqrt(variance) + delta
    outcome_advantages = standardized(outcomes, delta)
    return [
        [(1 - alpha) * outcome + alpha * ((value - origin - mean) / denominator if enabled else 0.0)
         for value in sequence]
        for outcome, sequence, enabled in zip(outcome_advantages, returns, mask)
    ]


def class_weight(counts, label, delta=DELTA):
    """Eq. (34), normalized to unit expectation under the observed class frequencies."""
    count = sum(counts.get(kind, 0) for kind in LABELS)
    if not count:
        return 1.0
    frequencies = {kind: counts.get(kind, 0) / count for kind in LABELS}
    inverse = {kind: 1 / sqrt(frequency + delta) for kind, frequency in frequencies.items()}
    return inverse[label] / fsum(frequencies[kind] * inverse[kind] for kind in LABELS)
