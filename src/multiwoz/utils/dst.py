#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json

all_domain = [
    "[taxi]",
    "[police]",
    "[hospital]",
    "[hotel]",
    "[attraction]",
    "[train]",
    "[restaurant]",
]
GENERAL_TYPO = {
    # type
    "guesthouse": "guest house",
    "guesthouses": "guest house",
    "guest": "guest house",
    "mutiple sports": "multiple sports",
    "sports": "multiple sports",
    "mutliple sports": "multiple sports",
    "swimmingpool": "swimming pool",
    "concerthall": "concert hall",
    "concert": "concert hall",
    "pool": "swimming pool",
    "night club": "nightclub",
    "mus": "museum",
    "ol": "architecture",
    "colleges": "college",
    "coll": "college",
    "architectural": "architecture",
    "musuem": "museum",
    "churches": "church",
    # area
    "center": "centre",
    "center of town": "centre",
    "near city center": "centre",
    "in the north": "north",
    "cen": "centre",
    "east side": "east",
    "east area": "east",
    "west part of town": "west",
    "ce": "centre",
    "town center": "centre",
    "centre of cambridge": "centre",
    "city center": "centre",
    "the south": "south",
    "scentre": "centre",
    "town centre": "centre",
    "in town": "centre",
    "north part of town": "north",
    "centre of town": "centre",
    "cb30aq": "none",
    # price
    "mode": "moderate",
    "moderate -ly": "moderate",
    "mo": "moderate",
    # day
    "next friday": "friday",
    "monda": "monday",
    # parking
    "free parking": "free",
    # internet
    "free internet": "yes",
    # star
    "4 star": "4",
    "4 stars": "4",
    "0 star rarting": "none",
    # others
    "y": "yes",
    "any": "dontcare",
    "n": "no",
    "does not care": "dontcare",
    "not men": "none",
    "not": "none",
    "not mentioned": "none",
    "not present": "none",
    "": "none",
    "not mendtioned": "none",
    "3 .": "3",
    "does not": "no",
    "fun": "none",
    "art": "none",
    # "dontcare": "none"
}


IGNORE_TURNS_TYPE2 = {
    "PMUL1812": [1, 2],
    "MUL2177": [9, 10, 11, 12, 13, 14],
    "PMUL0182": [1],
    "PMUL0095": [1],
    "MUL1883": [1],
    "PMUL2869": [9, 11],
    "SNG0433": [0, 1],
    "PMUL4880": [2],
    "PMUL2452": [2],
    "PMUL2882": [2],
    "SNG01391": [1],
    "MUL0803": [7],
    "MUL1560": [4, 5],
    "PMUL4964": [6, 7],
    "MUL1753": [8, 9],
    "PMUL3921": [4],
    "PMUL3403": [0, 4],
    "SNG0933": [3],
    "SNG0296": [1],
    "SNG0477": [1],
    "MUL0814": [1],
    "SNG0078": [1],
    "PMUL1036": [6],
    "PMUL4840": [2],
    "PMUL3423": [6],
    "MUL2284": [2],
    "PMUL1373": [1],
    "SNG01538": [1],
    "MUL0011": [2],
    "PMUL4326": [4],
    "MUL1697": [10],
    "MUL0014": [5],
    "PMUL1370": [1],
    "PMUL1801": [7],
    "MUL0466": [2],
    "PMUL0506": [1, 2],
    "SNG1036": [2],
    "MUL1575": [2],
}

requestable_slots = {
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"],
    "hotel": [
        "address",
        "postcode",
        "internet",
        "phone",
        "parking",
        "type",
        "pricerange",
        "stars",
        "area",
        "reference",
    ],
    "attraction": [
        "price",
        "type",
        "address",
        "postcode",
        "phone",
        "area",
        "reference",
    ],
    "train": ["time", "leave", "price", "arrive", "id", "reference"],
    "restaurant": [
        "phone",
        "postcode",
        "address",
        "pricerange",
        "food",
        "area",
        "reference",
    ],
}
all_reqslot = [
    "car",
    "address",
    "postcode",
    "phone",
    "internet",
    "parking",
    "type",
    "pricerange",
    "food",
    "stars",
    "area",
    "reference",
    "time",
    "leave",
    "price",
    "arrive",
    "id",
]
# count: 17

informable_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": [
        "type",
        "parking",
        "pricerange",
        "internet",
        "stay",
        "day",
        "people",
        "area",
        "stars",
        "name",
    ],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"],
}
all_infslot = [
    "type",
    "parking",
    "pricerange",
    "internet",
    "stay",
    "day",
    "people",
    "area",
    "stars",
    "name",
    "leave",
    "destination",
    "departure",
    "arrive",
    "department",
    "food",
    "time",
]
# count: 17

all_slots = (
    all_reqslot
    + all_infslot
    + ["stay", "day", "people", "name", "destination", "departure", "department"]
)
all_slots = set(all_slots)


def paser_bs(sent):
    """Convert compacted bs span to triple list
    Ex:
    """
    # sent=sent.strip('<sos_b>').strip('<eos_b>')
    sent = sent.strip()
    sent = sent.split()
    belief_state = []
    domain_idx = [idx for idx, token in enumerate(sent) if token in all_domain]
    for i, d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i + 1 == len(domain_idx) else domain_idx[i + 1]
        domain = sent[d_idx]
        sub_span = sent[d_idx + 1 : next_d_idx]
        sub_s_idx = [idx for idx, token in enumerate(sub_span) if token in all_slots]
        for j, s_idx in enumerate(sub_s_idx):
            next_s_idx = len(sub_span) if j == len(sub_s_idx) - 1 else sub_s_idx[j + 1]
            slot = sub_span[s_idx]
            value = " ".join(sub_span[s_idx + 1 : next_s_idx])
            bs = "->".join([domain, slot, value])
            belief_state.append(bs)
    return list(set(belief_state))


import re


def replace_whitespace(s):
    return re.sub(r"(\w+)\s(\'s\s\w+)", r"\1\2", s)


def ignore_none(pred_belief, target_belief):
    for pred in pred_belief:
        if "catherine s" in pred:
            pred.replace("catherine s", "catherine 's")

    clean_target_belief = []
    clean_pred_belief = []
    for bs in target_belief:
        if "not mentioned" in bs or "none" in bs or "not present" in bs:
            continue
        clean_target_belief.append(bs)

    for bs in pred_belief:
        if "not mentioned" in bs or "none" in bs or "not present" in bs:
            continue
        clean_pred_belief.append(bs)

    # dontcare_slots = []
    # for bs in target_belief:
    #     if 'dontcare' in bs:
    #         domain = bs.split("-")[0]
    #         slot = bs.split("-")[1]
    #         dontcare_slots.append('{}_{}'.format(domain, slot))

    target_belief = clean_target_belief
    pred_belief = clean_pred_belief

    return pred_belief, target_belief


def ignore_dontcare(pred_belief):
    clean_pred_belief = []

    for bs in pred_belief:
        if "dontcare" in bs or "dont care" in bs or "do not care" in bs:
            continue
        clean_pred_belief.append(bs)

    pred_belief = clean_pred_belief

    return pred_belief


def fix_mismatch_jason(slot, value):
    # miss match slot and value
    if (
        slot == "type"
        and value
        in [
            "nigh",
            "moderate -ly priced",
            "bed and breakfast",
            "centre",
            "venetian",
            "intern",
            "a cheap -er hotel",
        ]
        or slot == "internet"
        and value == "4"
        or slot == "pricerange"
        and value == "2"
        or slot == "type"
        and value in ["gastropub", "la raza", "galleria", "gallery", "science", "m"]
        or "area" in slot
        and value in ["moderate"]
        or "day" in slot
        and value == "t"
    ):
        value = "none"
    elif slot == "type" and value in [
        "hotel with free parking and free wifi",
        "4",
        "3 star hotel",
    ]:
        value = "hotel"
    elif slot == "star" and value == "3 star hotel":
        value = "3"
    elif "area" in slot:
        if value == "no":
            value = "north"
        elif value == "we":
            value = "west"
        elif value == "cent":
            value = "centre"
    elif "day" in slot:
        if value == "we":
            value = "wednesday"
        elif value == "no":
            value = "none"
    elif "price" in slot and value == "ch":
        value = "cheap"
    elif "internet" in slot and value == "free":
        value = "yes"
    elif "parking" in slot and value == "free":
        value = "yes"
    elif value.startswith("the"):
        value = value[4:]
    # elif ("time" in slot or "arrive" in slot or "leave" in slot) and ":" not in value:
    #     value = value+":00"
    elif "[value_" in value:
        value = "none"
    value = replace_whitespace(value)

    # some out-of-define classification slot values
    if (
        slot == "area"
        and value in ["stansted airport", "cambridge", "silver street"]
        or slot == "area"
        and value in ["norwich", "ely", "museum", "same area as hotel"]
    ):
        value = "none"
    return slot, value


def fix_mismatch_jason_2020(slot, val):
    # Typo or naming
    if val == "cafe uno":
        val = "caffe uno"
    if val == "alpha milton guest house":
        val = "alpha-milton guest house"
    if val in [
        "churchills college",
        "churchhill college",
        "churchill",
        "the churchill college",
    ]:
        val = "churchill college"
    if val == "portugese":
        val = "portuguese"
    # if val == "pizza hut fenditton":
    #     val = "pizza hut fen ditton"
    if val == "restaurant 17":
        val = "restaurant one seven"
    if val == "restaurant 2 two":
        val = "restaurant two two"
    if val == "gallery at 12 a high street":
        val = "gallery at twelve a high street"
    if val == "museum of archaelogy":
        val = "museum of archaelogy and anthropology"
    if val in ["huntingdon marriot hotel", "marriot hotel"]:
        val = "huntingdon marriott hotel"
    if val in [
        "sheeps green and lammas land park fen causeway",
        "sheeps green and lammas land park",
    ]:
        val = "sheep's green and lammas land park fen causeway"
    if val in ["cambridge and country folk museum", "county folk museum"]:
        val = "cambridge and county folk museum"
    if val == "ambridge":
        val = "cambridge"
    if val == "cambridge contemporary art museum":
        val = "cambridge contemporary art"
    if val == "molecular gastonomy":
        val = "molecular gastronomy"
    if val == "2 two and cote":
        val = "two two and cote"
    if val == "caribbeanindian":
        val = "caribbean|indian"
    if val == "whipple museum":
        val = "whipple museum of the history of science"
    if val == "ian hong":
        val = "ian hong house"
    if val == "sundaymonday":
        val = "sunday|monday"
    if val == "mondaythursday":
        val = "monday|thursday"
    if val == "fridaytuesday":
        val = "friday|tuesday"
    if val == "cheapmoderate":
        val = "cheap|moderate"
    if val == "golden house                            golden house":
        val = "the golden house"
    if val == "golden house":
        val = "the golden house"
    if val == "sleeperz":
        val = "sleeperz hotel"
    if val == "jamaicanchinese":
        val = "jamaican|chinese"
    if val == "shiraz":
        val = "shiraz restaurant"
    if val == "museum of archaelogy and anthropogy":
        val = "museum of archaelogy and anthropology"
    if val == "yipee noodle bar":
        val = "yippee noodle bar"
    if val == "abc theatre":
        val = "adc theatre"
    if val == "wankworth house":
        val = "warkworth house"
    if val in ["cherry hinton water play park", "cherry hinton water park"]:
        val = "cherry hinton water play"
    if val == "the gallery at 12":
        val = "the gallery at twelve"
    if val == "barbequemodern european":
        val = "barbeque|modern european"
    if val == "north americanindian":
        val = "north american|indian"
    if val == "chiquito":
        val = "chiquito restaurant bar"
    if val == "museum of archaeology and anthropology":
        val = "museum of archaelogy and anthropology"
    if val == "saint catherine's college":
        val = "saint catharine's college"
    if val == "king's lynn":
        val == "kings lynn"

    # Abbreviation
    if val == "city centre north bed and breakfast":
        val = "city centre north b and b"
    if val == "north bed and breakfast":
        val = "north b and b"

    # Article and 's
    if val == "christ college":
        val = "christ's college"
    if val == "kings college":
        val = "king's college"
    if val == "saint johns college":
        val = "saint john's college"
    if val == "kettles yard":
        val = "kettle's yard"
    if val == "rosas bed and breakfast":
        val = "rosa's bed and breakfast"
    if val == "saint catharines college":
        val = "saint catharine's college"
    if val == "little saint marys church":
        val = "little saint mary's church"
    if val == "great saint marys church":
        val = "great saint mary's church"
    if val in ["queens college", "queens' college"]:
        val = "queen's college"
    if val == "peoples portraits exhibition at girton college":
        val = "people's portraits exhibition at girton college"
    if val == "st johns college":
        val = "saint john's college"
    if val == "whale of time":
        val = "whale of a time"
    if val in ["st catharines college", "saint catharines college"]:
        val = "saint catharine's college"
    if val in ["saint john 's college", "saint johns colleg", "saint johns college"]:
        val = "saint john's college"
    if val in ["pizza hut fen ditton", "pizza hit fen ditton"]:
        val = "pizza hut fenditton"
    if val in ["the copper kettle"]:
        val = "copper kettle"
    if val in ["the dojo noodle bar"]:
        val = "dojo noodle bar"

    # Time
    if val == "16,15":
        val = "16:15"
    if val == "1330":
        val = "13:30"
    if val == "1430":
        val = "14:30"
    if val == "1532":
        val = "15:32"
    if val == "845":
        val = "08:45"
    if val == "1145":
        val = "11:45"
    if val == "1545":
        val = "15:45"
    if val == "1329":
        val = "13:29"
    if val == "1345":
        val = "13:45"
    if val == "1715":
        val = "17:15"
    if val == "929":
        val = "09:29"
    return slot, val


def default_cleaning(pred_belief, target_belief):
    pred_belief_jason = []
    target_belief_jason = []
    for pred in pred_belief:
        if pred in ["", " "]:
            continue
        # domain = pred.split()[0]
        # if 'book' in pred:
        #     slot = ' '.join(pred.split()[1:3])
        #     val = ' '.join(pred.split()[3:])
        # else:
        #     slot = pred.split()[1]
        #     val = ' '.join(pred.split()[2:])
        domain, slot, val = pred.split("->")

        if val in GENERAL_TYPO:
            val = GENERAL_TYPO[val]

        slot, val = fix_mismatch_jason(slot, val)
        slot, val = fix_mismatch_jason_2020(slot, val)

        pred_belief_jason.append("->".join([domain, slot, val]))

    for tgt in target_belief:
        # domain = tgt.split()[0]
        # if 'book' in tgt:
        #     slot = ' '.join(tgt.split()[1:3])
        #     val = ' '.join(tgt.split()[3:])
        # else:
        #     slot = tgt.split()[1]
        #     val = ' '.join(tgt.split()[2:])
        domain, slot, val = tgt.split("->")

        if slot in GENERAL_TYPO:
            val = GENERAL_TYPO[slot]

        slot, val = fix_mismatch_jason(slot, val)
        slot, val = fix_mismatch_jason_2020(slot, val)

        target_belief_jason.append("->".join([domain, slot, val]))

    turn_pred = pred_belief_jason
    turn_target = target_belief_jason

    return turn_pred, turn_target
