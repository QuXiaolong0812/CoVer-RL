"""Frozen local evidence scorers and Appendix A.1 Triangular Consistency Reward."""

from functools import lru_cache
import json
import math
from pathlib import Path

from PIL import Image
import torch


TEXT_FIELDS = ("title", "description", "text")
CALIBRATION_FILE = "event_nli_calibration.json"
# Time, intent, and semantic content are not visually verifiable roles in CMMEvent.
# Physical participants, objects, and scene locations retain their visual masks.
TEXT_ONLY_ROLE_MARKERS = ("时间", "目的", "内容", "项目", "新年")
DINO_PROMPT = (
    "person. group of people. vehicle. animal. object. food. plant. building. street. "
    "landscape. sign. furniture. tool. smoke. fire."
)


def text_premise(document):
    return "\n".join(document[field].strip() for field in TEXT_FIELDS if document.get(field, "").strip())


def verbalize(event_type, trigger, role=None, filler=None):
    header = f"发生了「{event_type}」事件，触发词是「{trigger}」"
    if role is None:
        return header + "。"
    return f"{header}，该事件的「{role}」是「{filler}」。"


def applicability(document, role):
    """Determine modality applicability with fixed input and schema rules."""
    textual = bool(text_premise(document))
    visual = bool(document.get("image", {}).get("path")) and (
        role is None or not any(marker in role for marker in TEXT_ONLY_ROLE_MARKERS)
    )
    return textual, visual, textual and visual


def sigmoid(logit):
    # Stable for both large positive and large negative logits.
    return math.exp(min(logit, 0.0)) / (1.0 + math.exp(-abs(logit)))


def claim_score(text, visual, masks, contradiction, progress):
    """Calculate the support factor, consistency score, and scheduled Eq. (14)."""
    a_t, a_v, a_x = map(int, masks)
    support = (a_t * text + a_v * visual) / (a_t + a_v) if a_t + a_v else 0.0
    cross = support * (1.0 - contradiction)
    text_weight = visual_weight = 0.4 + 0.1 * progress
    cross_weight = 0.2 * (1.0 - progress)
    denominator = text_weight * a_t + visual_weight * a_v + cross_weight * a_x
    numerator = text_weight * a_t * text + visual_weight * a_v * visual + cross_weight * a_x * cross
    return {"a_t": a_t, "a_v": a_v, "a_x": a_x, "s_t": text, "s_v": visual, "c": contradiction,
            "g": support, "s_x": cross, "rho": numerator / denominator if denominator else 0.0}


def load_calibration(checkpoint):
    path = Path(checkpoint) / CALIBRATION_FILE
    if not path.is_file():
        raise ValueError(f"Expected a trained event-NLI checkpoint with dev calibration at {path}.")
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if (not isinstance(data, dict) or data.get("task") != "cmmevent_event_nli"
            or data.get("train_split") != "train" or data.get("calibration_split") != "dev"
            or type(data.get("temperature")) not in (int, float) or not 0.25 <= data["temperature"] <= 4.0
            or type(data.get("max_length")) is not int or not 8 <= data["max_length"] <= 512):
        raise ValueError(f"Invalid trained event-NLI calibration at {path}.")
    return data


def valid_crop(box, size, *, clip=False):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)):
        return None
    width, height = size
    left, top, right, bottom = box
    if not clip and not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None
    if left >= right or top >= bottom:
        return None
    # Outward integer rounding preserves the original detector region.
    left, top = max(0, math.floor(left)), max(0, math.floor(top))
    right, bottom = min(width, math.ceil(right)), min(height, math.ceil(bottom))
    return (left, top, right, bottom) if left < right and top < bottom else None


def proposal_boxes(boxes, scores, labels, *, size, limit):
    proposals, seen = [], set()
    for raw_box, score, label in sorted(zip(boxes, scores, labels), key=lambda row: -row[1]):
        box = valid_crop(raw_box, size, clip=True)
        if box is None or box in seen or not math.isfinite(score) or not label.strip():
            continue
        proposals.append({"box": box, "detector_score": score, "label": label})
        seen.add(box)
        if len(proposals) == limit:
            break
    return proposals


class EvidenceScorer:
    """Evaluate atomic event claims using calibrated DeBERTa, Grounding DINO and SigLIP2."""

    def __init__(self, nli_dir, siglip_dir, dino_dir, dataset_dir, device="cpu", *,
                 box_threshold=0.25, text_threshold=0.2, max_proposals=12, visual_bias=0.10,
                 visual_temperature=0.10):
        from transformers import AutoModelForSequenceClassification, AutoModelForZeroShotObjectDetection
        from transformers import AutoProcessor, AutoTokenizer, SiglipModel

        self.calibration = load_calibration(nli_dir)
        if not 0 < box_threshold < 1 or not 0 < text_threshold < 1 or max_proposals < 1:
            raise ValueError("Invalid Grounding DINO proposal settings.")
        if not -1 <= visual_bias <= 1 or not 0.01 <= visual_temperature <= 2.0:
            raise ValueError("Invalid SigLIP2 cosine calibration settings.")
        self.dataset_dir = Path(dataset_dir).resolve()
        self.device = torch.device(device)
        self.box_threshold, self.text_threshold, self.max_proposals = box_threshold, text_threshold, max_proposals
        self.visual_bias, self.visual_temperature = visual_bias, visual_temperature
        local = {"local_files_only": True}
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_dir, use_fast=False, **local)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_dir, **local)
        if (self.nli_model.config.model_type != "deberta-v2" or self.nli_model.config.num_labels != 1
                or self.nli_model.config.id2label != {0: "ENTAILMENT"}):
            raise ValueError("Expected the trained single-logit event-NLI DeBERTa checkpoint.")
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_dir, use_fast=False, **local)
        self.siglip_model = SiglipModel.from_pretrained(siglip_dir, **local)
        self.dino_processor = AutoProcessor.from_pretrained(dino_dir, use_fast=False, **local)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_dir, disable_custom_kernels=True, **local
        )
        for model in (self.nli_model, self.siglip_model, self.dino_model):
            model.to(self.device).eval().requires_grad_(False)

        # Samples within and across rollout groups often repeat the same image or claim.
        self.text_support = lru_cache(maxsize=4096)(self.text_support)
        self._text_embedding = lru_cache(maxsize=1024)(self._text_embedding)
        self._region_embeddings = lru_cache(maxsize=48)(self._region_embeddings)

    @torch.inference_mode()
    def text_support(self, premise, hypothesis):
        inputs = self.nli_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt",
                                    max_length=self.calibration["max_length"]).to(self.device)
        logit = self.nli_model(**inputs).logits.item()
        if not math.isfinite(logit):
            raise ValueError("Nonfinite event-NLI logit.")
        return sigmoid(logit / self.calibration["temperature"])

    def _proposals(self, image):
        inputs = self.dino_processor(images=image, text=DINO_PROMPT, return_tensors="pt").to(self.device)
        result = self.dino_processor.post_process_grounded_object_detection(
            self.dino_model(**inputs), inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[(image.height, image.width)],
        )[0]
        return proposal_boxes(result["boxes"].tolist(), result["scores"].tolist(), result["text_labels"],
                              size=image.size, limit=self.max_proposals)

    @torch.inference_mode()
    def _region_embeddings(self, image_path, specified_box=None):
        relative = Path(image_path)
        path = (self.dataset_dir / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.dataset_dir):
            raise ValueError(f"Image path must stay inside the dataset directory: {relative}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        if specified_box is None:
            proposals = self._proposals(image)
        else:
            box = valid_crop(specified_box, image.size)
            proposals = [{"box": box, "detector_score": None, "label": "specified"}] if box is not None else []
        if not proposals:
            return proposals, None
        crops = [image.crop(p["box"]) for p in proposals]
        inputs = self.siglip_processor(images=crops, return_tensors="pt").to(self.device)
        features = self.siglip_model.get_image_features(**inputs)
        return proposals, torch.nn.functional.normalize(features.float(), dim=-1).cpu()

    @torch.inference_mode()
    def _text_embedding(self, hypothesis):
        inputs = self.siglip_processor(
            text=[f"照片中，{hypothesis}"], return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.siglip_model.config.text_config.max_position_embeddings,
        ).to(self.device)
        features = self.siglip_model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features.float(), dim=-1)[0].cpu()

    def visual_support(self, image_path, hypothesis, box):
        proposals, features = self._region_embeddings(image_path, box)
        if features is None:
            return {"s_v": 0.0, "status": "no_eligible_proposal" if box is None else "invalid_box",
                    "selected_box": None, "proposals": []}
        similarities = (features @ self._text_embedding(hypothesis)).clamp(-1, 1).tolist()
        if not all(math.isfinite(similarity) for similarity in similarities):
            raise ValueError("Nonfinite SigLIP2 cosine similarity.")
        scored = [
            {**proposal, "cosine": similarity,
             "support": sigmoid((similarity - self.visual_bias) / self.visual_temperature)}
            for proposal, similarity in zip(proposals, similarities)
        ]
        best = max(scored, key=lambda proposal: proposal["support"])
        return {"s_v": best["support"], "status": "scored", "selected_box": best["box"], "proposals": scored}

    def score(self, document, events, feedback=None, progress=0.0):
        if not 0.0 <= progress <= 1.0:
            raise ValueError("TCR progress must be in [0, 1].")
        premise = text_premise(document)
        image_path = document.get("image", {}).get("path")
        feedback = feedback or {}
        diagnoses = {
            (record["event_id"], record.get("argument_id")): record["error_type"]
            for level in ("event_level", "argument_level") for record in feedback.get(level, [])
            if record["event_id"] is not None and (level == "event_level" or record.get("argument_id") is not None)
        }
        event_scores = []
        for i, event in enumerate(events, 1):
            event_id = event.get("id", f"E_{i}")
            claims = []
            for j, unit in enumerate([event, *event["arguments"]]):
                arg_id = unit.get("id", f"A_{i}_{j}") if j else None
                role, filler = (unit["role"], unit["argument"]) if j else (None, None)
                hypothesis = verbalize(event["event_type"], event["trigger"], role, filler)
                masks = applicability(document, role)
                text = self.text_support(premise, hypothesis) if masks[0] else 0.0
                raw_box = unit.get("bbox")
                box = None if raw_box is None else valid_crop(raw_box, (math.inf, math.inf))
                # An explicitly invalid box cannot fall back to unrelated proposals.
                visual = self.visual_support(image_path, hypothesis, box if raw_box is None or box else ()) if masks[1] else {
                    "s_v": 0.0, "status": "inapplicable", "selected_box": None, "proposals": [],
                }
                key = (event_id, arg_id)
                error = diagnoses.get(key)
                contradiction = float(masks[2] and error == "modal_conflict")
                claims.append({
                    "event_id": event_id, "argument_id": arg_id, "role": role, "filler": filler,
                    "hypothesis": hypothesis, "specified_box": raw_box, "visual": visual,
                    "hsfp_present": key in diagnoses, "hsfp_error": error,
                    **claim_score(text, visual["s_v"], masks, contradiction, progress),
                })
            event_scores.append({"event_id": event_id, "claims": claims,
                                 "score": sum(claim["rho"] for claim in claims) / len(claims)})
        tcr = sum(event["score"] for event in event_scores) / len(event_scores) if event_scores else 0.0
        return {"doc_id": document.get("id"), "progress": progress, "tcr": tcr, "events": event_scores}
