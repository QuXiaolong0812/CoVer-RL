"""Evidence checks for the additional CMMEvent SPP operators."""

from collections import Counter
import re


EXTENDED_OPERATORS = ("B1", "B2", "C1", "C2", "D1", "D2")
TEXT_FIELDS = ("text", "title", "description")
SENTENCE_END = re.compile(r"[。！？；!?;\n]")
NEGATION = re.compile(r"(?:未曾|并未|没有|无法|未能|拒绝|禁止|从未|尚未|未|不)(?:成功|进行|实施|参与|前往|实际|发生)?$")
VISUAL_VERBS = {
    "交流类-介绍类": "instructing", "交流类-读类": "reading", "交流类-谈话类": "communicating",
    "写类-签署类": "signing", "写类-绘画类": "drawing",
    "农业类-种植类": "planting", "农业类-采摘类": "mowing",
    "展出类-展示类": "displaying", "展出类-绽开类": "blossoming",
    "救援类-救助类": "bandaging", "救援类-灭火类": "extinguishing",
    "旅游类-参观类": "admiring", "旅游类-拍照类": "photographing",
    "活动类-比赛类": "competing", "活动类-表演类": "performing", "活动类-阅兵类": "parading",
    "生活类-买卖类": "shopping", "生活类-吃类": "eating",
    "移动类-驾驶类": "driving", "移动类-运输类": "carting",
    "聚集类-排队类": "queuing", "聚集类-等待类": "waiting",
}

# Distinctive trigger cues for the selected CMMEvent subtypes.
TYPE_CUES = {
    "农业类-种植类": ("种植", "插秧", "播种", "移栽", "栽种"),
    "农业类-采摘类": ("采收", "采摘", "收割", "收获", "摘取", "摘下", "摘菜"),
    "写类-签署类": ("签署", "签字", "签名"),
    "写类-绘画类": ("绘画", "绘制", "画画", "涂鸦", "速写"),
    "旅游类-拍照类": ("拍照", "合影", "合照", "自拍", "留影"),
    "旅游类-参观类": ("参观",),
    "救援类-灭火类": ("灭火", "扑灭"),
    "救援类-救助类": ("救助", "搜救", "解救", "救出", "救援"),
    "展出类-绽开类": ("绽放", "盛开"),
    "生活类-吃类": ("进食", "吃饭", "用餐", "品尝", "进餐"),
    "移动类-驾驶类": ("驾驶", "行驶", "骑车", "驶入", "驶出"),
    "活动类-比赛类": ("拼抢", "比赛", "跳水", "扣篮"),
    "活动类-阅兵类": ("阅兵", "检阅"),
    "活动类-表演类": ("表演", "演出", "演唱", "演奏"),
    "交流类-介绍类": ("介绍", "讲解"),
    "交流类-读类": ("阅读", "读书"),
}
TYPE_PAIRS = (
    ("农业类-种植类", "农业类-采摘类"),
    ("写类-签署类", "写类-绘画类"),
    ("旅游类-参观类", "旅游类-拍照类"),
    ("救援类-救助类", "救援类-灭火类"),
)
CONFUSABLE = {source: target for first, second in TYPE_PAIRS
              for source, target in ((first, second), (second, first))}
CONTEXT_ROLES = {
    "农业类-种植类": ("种植者", "种植地点", "种植植物"),
    "农业类-采摘类": ("采摘者", "采摘地点", "采摘植物"),
    "写类-签署类": ("写者", "写作地点"),
    "写类-绘画类": ("绘画者", "绘画地点"),
}
IMAGE_CONFLICT_TYPES = frozenset({
    "交流类-读类", "写类-签署类", "写类-绘画类", "农业类-种植类", "农业类-采摘类",
    "展出类-绽开类", "旅游类-拍照类", "活动类-比赛类", "活动类-表演类",
    "活动类-阅兵类", "生活类-吃类", "移动类-驾驶类", "救援类-灭火类",
    "交流类-介绍类",
})

# Exact Chinese labels whose visual frame classes can be matched without guessing
# which occurrence or object the textual role refers to.
VISUAL_ROLES = {
    ("旅游类-拍照类", "拍照工具"): ("tool", {"手机": "n04401088", "相机": "n02942699"}),
    ("移动类-驾驶类", "交通工具"): ("item", {
        "自行车": "n02834778", "电动摩托车": "n03790512", "出租车": "n02958343",
    }),
    ("写类-签署类", "写作工具"): ("tool", {"笔": "n03906997", "手": "n05564590"}),
    ("写类-签署类", "写作对象"): ("item", {"纸": "n14974264"}),
    ("农业类-种植类", "种植植物"): ("planted", {"植物": "n00017222"}),
    ("旅游类-参观类", "参观内容"): ("admired", {"花": "n11669921"}),
    ("生活类-买卖类", "物品"): ("goods", {"冰淇淋": "n07614500"}),
    ("移动类-运输类", "运输工具"): ("tool", {"手": "n05564590"}),
}
VISUAL_CONTRAST = {
    "手机": ("相机", ("相机", "照相机", "单反", "微单", "摄影机", "摄像机")),
    "相机": ("手机", ("手机", "电话", "智能机")),
    "自行车": ("出租车", ("出租车", "的士", "计程车", "汽车")),
    "电动摩托车": ("出租车", ("出租车", "的士", "计程车", "汽车")),
    "出租车": ("自行车", ("自行车", "单车", "脚踏车")),
}


def unique_span(text, value):
    if not value:
        return None
    start = text.find(value)
    if start < 0 or text.find(value, start + 1) >= 0:
        return None
    return [start, start + len(value)]


def same_sentence(text, first, second):
    left, right = sorted((first, second))
    return left[1] <= right[0] and not SENTENCE_END.search(text, left[1], right[0])


def trigger_evidence(row, event):
    text = row["text"]
    trigger = event["trigger"]
    span = unique_span(text, trigger)
    if span is None or span != event["trigger_span"] or NEGATION.search(text[:span[0]].rstrip()):
        return None
    return {"source": "text", "kind": "trigger", "span": span, "text": trigger, "event_type": event["event_type"]}


def role_swap(row, argument_evidence):
    rules = {
        "交流类-介绍类": (("介绍者", "介绍对象"), ("介绍", "讲解", "指导")),
        "救援类-救助类": (("救助者", "受助者"), ("救助", "解救", "协助", "救出", "帮助")),
    }
    values = Counter(arg["argument"] for event in row["event_list"] for arg in event["arguments"])
    text = row["text"]
    options = []
    for event_index, event in enumerate(row["event_list"]):
        rule = rules.get(event["event_type"])
        anchor = trigger_evidence(row, event)
        if rule is None or anchor is None or event["trigger"] not in rule[1]:
            continue
        slots = [[(i, arg) for i, arg in enumerate(event["arguments"]) if arg["role"] == role]
                 for role in rule[0]]
        if any(len(slot) != 1 for slot in slots):
            continue
        (left_index, left), (right_index, right) = (slot[0] for slot in slots)
        support = [argument_evidence(row, arg) for arg in (left, right)]
        if any(evidence is None or evidence["source"] != "text" for evidence in support):
            continue
        if any(values[arg["argument"]] != 1 for arg in (left, right)):
            continue
        (a, b), (c, d), (e, f) = support[0]["span"], support[1]["span"], anchor["span"]
        if rule[0][0] == "介绍者":
            directed = b <= c and d <= e and text[b:c].strip() in ("向", "给", "为", "对") and not text[d:e].strip()
        else:
            directed = (b <= e and f <= c and text[b:e].strip() in ("", "在", "正在", "及时")
                        and text[f:c].strip() in ("", "转移", "受灾的", "被洪水围困的", "受洪灾影响的"))
        if not directed or not same_sentence(text, [a, b], [c, d]):
            continue
        end = max(d, f)
        options.append({"event_index": event_index, "argument_indices": [left_index, right_index],
                        "evidence": [anchor, *support, {"source": "text", "kind": "direction",
                                                       "span": [a, end], "text": text[a:end]}]})
    return options, "no_unambiguous_directed_role_pair"


def type_confusion(row):
    options = []
    for index, event in enumerate(row["event_list"]):
        alternative = CONFUSABLE.get(event["event_type"])
        anchor = trigger_evidence(row, event)
        if (alternative is None or anchor is None or event["trigger"] not in TYPE_CUES[event["event_type"]]
                or sum(e["trigger"] == event["trigger"] for e in row["event_list"]) != 1
                or any(e["event_type"] == alternative for e in row["event_list"])):
            continue
        options.append({"event_index": index, "alternative": alternative, "evidence": [anchor]})
    return options, "no_distinctive_trigger_with_confusable_type"


def absent_from_text(row, cues):
    return not any(cue in row[field] for cue in cues for field in TEXT_FIELDS)


def valid_box(row, box):
    width, height = row["image"].get("width"), row["image"].get("height")
    if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
        return False
    return bool(width and height and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def frame_class(annotation, role):
    classes = {frame.get(role) for frame in annotation.get("frames", [])}
    if len(classes) == 1:
        return classes.pop()
    return None


def unique_visual(row, event_type, *, exclusive=False):
    verb = VISUAL_VERBS.get(event_type)
    annotations = row.get("visual_annotations", [])
    matches = [annotation for annotation in annotations if annotation["verb"] == verb]
    if len(matches) != 1 or (exclusive and len(annotations) != 1):
        return None
    annotation = matches[0]
    if not any(valid_box(row, box) for box in annotation["bb"].values()):
        return None
    return annotation


def visual_evidence(row, annotation, **fields):
    return {"source": "image", "doc_id": row["id"], "annotation_id": annotation["annotation_id"],
            "verb": annotation["verb"], **fields}


def grounded_visual_arguments(row):
    types = Counter(event["event_type"] for event in row["event_list"])
    for event_index, event in enumerate(row["event_list"]):
        event_type = event["event_type"]
        if types[event_type] != 1:
            continue
        annotation = unique_visual(row, event_type)
        if annotation is None:
            continue
        roles = Counter(arg["role"] for arg in event["arguments"])
        for arg_index, arg in enumerate(event["arguments"]):
            rule = VISUAL_ROLES.get((event_type, arg["role"]))
            if arg["source"] != "image" or roles[arg["role"]] != 1 or rule is None:
                continue
            role, classes = rule
            synset = classes.get(arg["argument"])
            box = annotation["bb"].get(role)
            if synset and frame_class(annotation, role) == synset and valid_box(row, box):
                evidence = visual_evidence(row, annotation, kind="grounded_argument",
                                           role=role, synset=synset, bbox=box, text=arg["argument"])
                yield event_index, arg_index, arg, annotation, evidence


def overlap(first, second):
    return (max(0, min(first[2], second[2]) - max(first[0], second[0]))
            * max(0, min(first[3], second[3]) - max(first[1], second[1])))


def displace_box(row):
    options = []
    for event_index, arg_index, _, annotation, support in grounded_visual_arguments(row):
        matching_boxes = [box for entry in row["visual_annotations"] for role, box in entry["bb"].items()
                          if valid_box(row, box) and frame_class(entry, role) == support["synset"]]
        for role, box in annotation["bb"].items():
            synset = frame_class(annotation, role)
            if (role == "place" or not synset or synset == support["synset"] or not valid_box(row, box)
                    or any(overlap(box, original) for original in matching_boxes)):
                continue
            distractor = visual_evidence(row, annotation, kind="other_entity", role=role, synset=synset, bbox=box)
            options.append({"event_index": event_index, "argument_index": arg_index,
                            "original": support["bbox"], "alternative": box, "evidence": [support, distractor]})
    return options, "no_grounded_argument_with_disjoint_annotated_entity"


def image_scene(row):
    if len(row["event_list"]) != 1:
        return None
    event = row["event_list"][0]
    event_type = event["event_type"]
    if event_type not in IMAGE_CONFLICT_TYPES or event["trigger"] not in TYPE_CUES[event_type]:
        return None
    anchor = trigger_evidence(row, event)
    annotation = unique_visual(row, event_type, exclusive=True)
    if anchor is not None and annotation is not None:
        return event_type.partition("-")[0], [anchor, visual_evidence(row, annotation, kind="annotated_scene")]
    return None


def image_mismatch(row, scenes, argument_evidence):
    scene = scenes.get(row["id"])
    if scene is None:
        return [], "no_single_confirmed_image_event"
    if not row["event_list"][0]["arguments"] or any(
        argument_evidence(row, arg) is None for arg in row["event_list"][0]["arguments"]
    ):
        return [], "image_swap_would_invalidate_nontextual_arguments"
    family, source = scene
    mentioned = {kind for kind in IMAGE_CONFLICT_TYPES if not absent_from_text(row, TYPE_CUES[kind])}
    donors = {}
    for donor, (other_family, evidence) in scenes.values():
        if (other_family == family or donor["split"] != row["split"] or donor["group_id"] == row["group_id"]
                or donor["image"]["sha256"] == row["image"]["sha256"]
                or donor["image"]["path"] == row["image"]["path"]):
            continue
        donor_event = donor["event_list"][0]
        if donor_event["event_type"] not in mentioned:
            donors.setdefault(donor["group_id"], (donor, evidence[1]))
    if donors:
        return [{"event_index": 0, "donors": list(donors.values()), "evidence": source}], None
    return [], "no_distinct_same_split_image_donor"


def redundant_argument(row):
    options = []
    for event_index, _, arg, annotation, support in grounded_visual_arguments(row):
        contrast = VISUAL_CONTRAST.get(arg["argument"])
        if contrast is None:
            continue
        alternative, cues = contrast
        if not absent_from_text(row, cues) or any(
            any(cue in other["argument"] for cue in cues)
            for event in row["event_list"] for other in event["arguments"]
        ):
            continue
        _, classes = VISUAL_ROLES[(row["event_list"][event_index]["event_type"], arg["role"])]
        if any(frame_class(entry, role) == classes[alternative] for entry in row["visual_annotations"]
               for role in entry["bb"]):
            continue
        options.append({"event_index": event_index, "role": arg["role"], "alternative": alternative,
                        "evidence": [support, visual_evidence(row, annotation, kind="incompatible_visual_class",
                                                             proposed_synset=classes[alternative])]})
    return options, "no_annotated_role_with_unmentioned_incompatible_class"


def redundant_event(row, argument_evidence):
    if len(row["event_list"]) != 1:
        return [], "no_single_confirmed_event"
    event = row["event_list"][0]
    event_type = event["event_type"]
    alternative = CONFUSABLE.get(event_type)
    if event_type not in CONTEXT_ROLES:
        return [], "no_confusable_event_with_shared_context_roles"
    anchor = trigger_evidence(row, event)
    annotation = unique_visual(row, event_type, exclusive=True)
    if anchor is None or event["trigger"] not in TYPE_CUES[event_type] or annotation is None:
        return [], "no_single_confirmed_event"
    if not absent_from_text(row, TYPE_CUES[alternative]):
        return [], "competing_event_cue_present"
    roles = dict(zip(CONTEXT_ROLES[event_type], CONTEXT_ROLES[alternative]))
    roles["时间"] = "时间"
    counts = Counter(arg["role"] for arg in event["arguments"])
    transferred = []
    support = []
    for arg in event["arguments"]:
        role = roles.get(arg["role"])
        if role and counts[arg["role"]] == 1:
            evidence = argument_evidence(row, arg)
            if evidence is None:
                continue
            if arg["role"] == CONTEXT_ROLES[event_type][0] and (
                evidence["source"] != "text" or not same_sentence(row["text"], evidence["span"], anchor["span"])
            ):
                continue
            transferred.append({"role": role, "argument": arg["argument"]})
            support.append(evidence)
    if not any(arg["role"] == CONTEXT_ROLES[alternative][0] for arg in transferred):
        return [], "no_grounded_shared_participant"
    injected = {"event_type": alternative, "trigger": TYPE_CUES[alternative][0], "arguments": transferred}
    return [{"alternative": injected, "evidence": [
        anchor, visual_evidence(row, annotation, kind="annotated_scene"),
        {"kind": "competing_cues_absent", "fields": list(TEXT_FIELDS), "cues": list(TYPE_CUES[alternative])},
        *support,
    ]}], None


def extended_options(operator, row, scenes, argument_evidence):
    if operator == "B1":
        return role_swap(row, argument_evidence)
    if operator == "B2":
        return type_confusion(row)
    if operator == "C1":
        return displace_box(row)
    if operator == "C2":
        return image_mismatch(row, scenes, argument_evidence)
    if operator == "D1":
        return redundant_argument(row)
    return redundant_event(row, argument_evidence)
