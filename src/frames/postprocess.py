#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import pandas as pd
import json
import random
import copy
import yaml
import re
from tqdm import trange
from tqdm import tqdm
import argparse

from src.multiwoz.postprocess import compare_dict, unzip_session_data, sample_data_ids
from src.frames.preprocess import all_domain, all_acts, act_description_dict
from src.multiwoz.utils.utils import paser_dict_to_list

"""
Frames:
Domains: movie, restaurant, taxi

TODO
- reuse the schema of multiwoz
- add db exact information and delex response
- only examples for the restaurant domain
"""


def load_schema():
    processed_data_path = f"./data/pre-training_corpora/processed_data/Frames"
    schema_path = f"{processed_data_path}/normalized_schema.yml"
    ontology_path = f"{processed_data_path}/ontology.json"

    if not os.path.exists(schema_path):
        with open(ontology_path, "r") as file:
            ontology = json.load(file)

        schema = []
        for domain in all_domain:
            service = {}
            service["service_name"] = domain[1:-1]

            # slots
            service["slots"] = []
            for slot in ontology[domain]["slots"]:
                # append
                service["slots"].append(
                    {
                        "name": slot,
                        "description": " ".join(slot.split("_")),
                        "possible_values": [],
                        "is_categorical": False,
                    }
                )

            # actions
            service["actions"] = []
            for act in ontology[domain]["actions"]:
                # description
                description = (
                    act_description_dict[act] if act in act_description_dict else ""
                )
                # append
                service["actions"].append(
                    {"name": act[1:-1], "description": description}
                )

            schema.append(service)

        # save data
        with open(schema_path, "w") as yaml_file:
            yaml.dump(schema, yaml_file, sort_keys=False)

    else:
        with open(schema_path, "r") as yaml_file:
            schema = yaml.safe_load(yaml_file)

    return schema


def process_data(data, split, reader):
    # Start
    processed_data = {}

    for dial_idx, dial in enumerate(tqdm(data)):
        dial_id = f"{split}_{dial_idx}"
        turns = dial["dialogue_session"]
        processed_turns = []
        valid_turn = 0

        for turn_id, turn in enumerate(turns):
            processed_turn = {}

            for key in [
                "dial_id",
                "turn_num",
                "user",
                "resp",
                "nodelx_resp",
                "dspn",
                "bspn",
                "bspn_dict",
                "turn_bspn_dict",
                "bsdx",
                "aspn",
                "aspn_dict",
                "turn_domain",
                "all_domains",
            ]:
                if key in turn:
                    processed_turn[key] = turn[key]
                else:
                    processed_turn[key] = ""

            if not processed_turn["dial_id"]:
                processed_turn["dial_id"] = dial_id

            processed_turn["all_domains"] = ["[travel]"]

            # delexicalized response
            processed_turn["nodelx_resp"] = turn["resp"]

            # belief state update
            if turn_id == 0:
                turn_dict = processed_turn["bspn_dict"]
            else:
                turn_dict = compare_dict(
                    old_dict=processed_turns[-1]["bspn_dict"],
                    new_dict=processed_turn["bspn_dict"],
                )
            processed_turn["turn_bspn_dict"] = turn_dict

            if turn_dict:
                valid_turn += 1

            # save data
            processed_turns.append(processed_turn)

        if valid_turn / len(turns) >= 0.2:
            processed_data[dial_id] = processed_turns

    return processed_data


def get_data_split(reader, n_train=-1, n_test=-1, return_list=False):
    preprocessed_data_path = f"./data/pre-training_corpora/separate_datasets/Frames"
    processed_data_path = f"./data/pre-training_corpora/processed_data/Frames"
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    # training data
    if not os.path.exists(f"{processed_data_path}/train_raw_dials.json"):
        with open(f"{preprocessed_data_path}/frames_train.json", "r") as file:
            train_data = json.load(file)
            train_data = process_data(train_data, "train", reader)
        with open(f"{processed_data_path}/train_raw_dials.json", "w") as file:
            json.dump(train_data, file, indent=4)
    else:
        with open(f"{processed_data_path}/train_raw_dials.json", "r") as file:
            train_data = json.load(file)

    # test data
    if not os.path.exists(f"{processed_data_path}/test_raw_dials.json"):
        with open(f"{preprocessed_data_path}/frames_test.json", "r") as file:
            test_data = json.load(file)
            test_data = process_data(test_data, "test", reader)
        with open(f"{processed_data_path}/test_raw_dials.json", "w") as file:
            json.dump(test_data, file, indent=4)
    else:
        with open(f"{processed_data_path}/test_raw_dials.json", "r") as file:
            test_data = json.load(file)

    # randomly sampled data
    if n_train != -1:
        if not os.path.exists(f"{processed_data_path}/train_{n_train}_ids.json"):
            train_data_ids = sample_data_ids(train_data, n_train)
            if n_train < len(
                train_data
            ):  # only record the sampled ids is not the full set
                with open(
                    f"{processed_data_path}/train_{n_train}_ids.json", "w"
                ) as file:
                    json.dump(train_data_ids, file)
        else:
            with open(f"{processed_data_path}/train_{n_train}_ids.json", "r") as file:
                train_data_ids = json.load(file)

        sampled_train_data = {}
        for did in train_data_ids:
            sampled_train_data[did] = train_data[did]
        train_data = sampled_train_data

    if n_test != -1:
        if not os.path.exists(f"{processed_data_path}/test_{n_test}_ids.json"):
            test_data_ids = sample_data_ids(test_data, n_test)
            if n_test < len(
                test_data
            ):  # only record the sampled ids is not the full set
                with open(f"{processed_data_path}/test_{n_test}_ids.json", "w") as file:
                    json.dump(test_data_ids, file)
        else:
            with open(f"{processed_data_path}/test_{n_test}_ids.json", "r") as file:
                test_data_ids = json.load(file)

    print(f"Train dataset size: {len(train_data)}")
    print(f"Test dataset size: {len(test_data)}")

    if return_list:
        return unzip_session_data(train_data), unzip_session_data(test_data)
    else:
        return train_data, test_data


def retrieve_demo(
    dialogs, schema, domains, n=100, max_turns=3, bs_da_ratios=[0.8, 0.2]
):
    """
    Demo selection
    """
    all_slots, all_acts = [], []
    # select the required slots
    for domain in domains:
        for service in schema:
            if service["service_name"] == domain:
                break
        for slot in service["slots"]:
            all_slots.append(slot)
        for act in service["actions"]:
            all_acts.append(act)

    # select required dialog acts for the domains
    bs_ratio, da_ratio = bs_da_ratios
    covered_da_list, covered_bs_list = [], []

    # filtered demos with the same domains
    filtered_ids = []
    for dial_id, turns in dialogs.items():
        if set(turns[-1]["all_domains"]) == set(domains):
            filtered_ids.append(dial_id)

    # measure score
    demo_scores = []
    for dial_id in filtered_ids:
        covered_da, covered_bs = [], []

        turns = dialogs[dial_id]
        # mentioned belief states, dialog acts
        turn = turns[-1]

        da_dict = turn["aspn_dict"]
        da_list = paser_dict_to_list(da_dict, level=2)
        for da in da_list:
            if da not in covered_da_list:
                covered_da.append(da)

        bs_dict = turn["bspn_dict"]
        bs_list = paser_dict_to_list(bs_dict, level=2)
        for bs in bs_list:
            if bs not in covered_bs_list:
                covered_bs.append(bs)

        covered_da = list(set(covered_da))
        covered_bs = list(set(covered_bs))

        # diversity
        bs_score = len(covered_bs) / len(all_slots) if len(all_slots) > 0 else 0
        da_score = len(covered_da) / len(all_acts) if len(all_acts) > 0 else 0
        demo_score = da_ratio * da_score + bs_ratio * bs_score

        # length penality
        turn_num = len(turns)
        lp = (turn_num / max_turns) - 1
        lp = max(0, lp)
        lp = np.exp(-lp)
        demo_score *= lp

        demo_scores.append(demo_score)

    # rank the dialogues, select the top n
    sorted_pairs = sorted(
        zip(filtered_ids, demo_scores), key=lambda pair: pair[1], reverse=True
    )[:n]
    selected_demo_ids = [pair[0] for pair in sorted_pairs]

    return selected_demo_ids


def load_examples(data, ratio=0.1, max_turns=3, bs_da_ratios=[0.8, 0.2]):
    schema = load_schema()

    all_examples = {}
    all_example_ids = []
    data_size = len(data)  # number of dialogs
    example_size = min(
        int(ratio * data_size), 100
    )  # take 10% for demonstration examples

    for domain in all_domain:
        example_ids = retrieve_demo(
            data,
            schema,
            [domain],  # retrive single-domain examples
            n=int(example_size // len(all_domain)),  # size for each domain
            max_turns=max_turns,  # the max number of turns
            bs_da_ratios=bs_da_ratios,  # the ratio between bs and da diversity
        )
        domain_examples = [data[dial_id] for dial_id in example_ids]
        all_examples[domain] = domain_examples
        all_example_ids.extend(example_ids)

    # exclude the examples from the data
    new_data = {}
    for dial_id in data:
        if dial_id not in all_example_ids:
            new_data[dial_id] = data[dial_id]

    return all_examples, new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument("--dataset", type=str, default="frames")  #
    parser.add_argument("--split", type=str, default="test")  #

    args, unknown = parser.parse_known_args()
    reader = None

    # component 1: process the normalized_schema.yml
    normalized_schema = load_schema()

    # component 2: post-process the dialogue data
    train_data, test_data = get_data_split(reader)

    # component 3: retrieve examples for the domain combinations
    # examples = load_examples(train_data)
