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

from src.taskmaster.preprocess import (
    domain_term_dict,
    domain_mapping_dict,
    domain_list,
    all_domain,
)
from src.multiwoz.postprocess import (
    compare_dict,
    unzip_session_data,
    sample_data_ids,
    paser_dict_to_list,
)


"""
TaskMaster:
Domains: movie, restaurant, taxi

TODO
- reuse the schema of multiwoz
- add db exact information and delex response
- only examples for the restaurant domain
"""


def load_schema():
    preprocessed_data_path = f"./data/pre-training_corpora/separate_datasets/TaskMaster"
    processed_data_path = f"./data/pre-training_corpora/processed_data/TaskMaster"
    schema_path = f"{processed_data_path}/normalized_schema.yml"
    ontology_path = f"{preprocessed_data_path}/ontology.json"

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path, exist_ok=True)

    if not os.path.exists(schema_path):
        with open(ontology_path, "r") as file:
            ontology = json.load(file)

        schema = []
        for domain in domain_list:
            service = {}
            # rename domain
            # new_domain = domain_alias_dict[domain]
            # service["service_name"] = new_domain
            service["service_name"] = domain

            # rename slots
            service["slots"] = []
            included_slots = domain_mapping_dict[domain].keys()
            alias = domain_mapping_dict[domain]
            descriptions = domain_term_dict[domain]
            for slot in included_slots:
                # possible values
                possible_values = []
                for v in ontology[domain]["slots"][slot]:
                    # avoid values like {Burbank#Pasadena California}
                    if (
                        not (v.startswith("{") and v.endswith("}"))
                        and v
                        and v.lower() not in possible_values
                    ):
                        possible_values.append(v)

                # is categorical
                # if len(possible_values) <= 10:
                #     is_categorical = True
                # else:
                #     is_categorical = False
                #     possible_values = possible_values[:20]
                is_categorical = False
                possible_values = possible_values[:20]

                # description
                description = descriptions[slot] if slot in descriptions else ""
                # rename slots
                if slot in alias:
                    new_slot = alias[slot]
                else:
                    new_slot = descriptions[slot]
                new_slot = "_".join(new_slot.split())

                # append
                service["slots"].append(
                    {
                        "name": f"{domain}-{new_slot}",
                        "description": description,
                        "possible_values": possible_values,
                        "is_categorical": is_categorical,
                    }
                )

            # actions
            service["actions"] = []
            schema.append(service)

        # save data
        with open(schema_path, "w") as yaml_file:
            yaml.dump(schema, yaml_file, sort_keys=False)

    else:
        with open(schema_path, "r") as yaml_file:
            schema = yaml.safe_load(yaml_file)

    return schema


def process_data(data, split):
    # Start
    processed_data = {}

    for dial_idx, dial in enumerate(tqdm(data)):
        dial_id = f"{split}_{dial_idx}"
        turns = dial["dialogue_session"]
        processed_turns = []
        valid_turn = 0

        for turn_id, turn in enumerate(turns):
            dial_domains = []
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

            processed_turn["dial_id"] = dial_id
            processed_turn["nodelx_resp"] = processed_turn["resp"]

            # rename the domain name
            turn_domain = turn["turn_domain"][0]
            # new_turn_domain = "["+domain_alias_dict[turn_domain]+"]"
            new_turn_domain = "[" + turn_domain + "]"
            processed_turn["dspn"] = new_turn_domain
            processed_turn["turn_domain"] = [new_turn_domain]
            processed_turn["all_domains"] = [new_turn_domain]  # all single turn

            # clean belief state, dialog act
            bs_dict = turn["bspn_dict"]
            cleaned_bs_dict = {}
            cleaned_bs_dict[new_turn_domain] = {}
            for slot, value in bs_dict[turn_domain].items():
                try:
                    # slot = domain_mapping_dict[turn_domain][slot]
                    # cleaned_bs_dict[new_turn_domain][slot] = value
                    if slot in domain_mapping_dict[turn_domain]:
                        new_slot = domain_mapping_dict[turn_domain][slot]
                    else:
                        new_slot = domain_term_dict[turn_domain][slot]
                    new_slot = "_".join(new_slot.split())
                    value = value[:-1] if value.endswith(".") else value
                    cleaned_bs_dict[new_turn_domain][new_slot] = value
                except:
                    print(turn_domain, slot)
            processed_turn["bspn_dict"] = cleaned_bs_dict
            processed_turn["aspn_dict"] = {}

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

        if valid_turn / (len(turns)) >= 0.2:
            processed_data[dial_id] = processed_turns

    return processed_data


def get_data_split(n_train=-1, n_test=-1, return_list=False):
    preprocessed_data_path = f"./data/pre-training_corpora/separate_datasets/TaskMaster"
    processed_data_path = f"./data/pre-training_corpora/processed_data/TaskMaster"

    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    # training data
    if not os.path.exists(f"{processed_data_path}/train_raw_dials.json"):
        with open(f"{preprocessed_data_path}/taskmaster_train.json", "r") as file:
            train_data = json.load(file)
            train_data = process_data(train_data, split="train")
        with open(f"{processed_data_path}/train_raw_dials.json", "w") as file:
            json.dump(train_data, file, indent=4)
    else:
        with open(f"{processed_data_path}/train_raw_dials.json", "r") as file:
            train_data = json.load(file)

    # test data
    if not os.path.exists(f"{processed_data_path}/test_raw_dials.json"):
        with open(f"{preprocessed_data_path}/taskmaster_test.json", "r") as file:
            test_data = json.load(file)
            test_data = process_data(test_data, split="test")
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
    dialogs, schema, domains, n=100, max_turns=3, bs_da_ratios=[1.0, 0.0]
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

        # da_dict = turn["aspn_dict"]
        # da_list = paser_dict_to_list(da_dict, level=2)
        # for da in da_list:
        #     if da not in covered_da_list:
        #         covered_da.append(da)

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


def load_examples(data, ratio=0.1, max_turns=3, bs_da_ratios=[1.0, 0.0]):
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
    parser.add_argument("--dataset", type=str, default="mse2e")  #
    parser.add_argument("--split", type=str, default="test")  #

    args, unknown = parser.parse_known_args()

    # component 1: process the normalized_schema.yml
    normalized_schema = load_schema()

    # component 2: post-process the dialogue data
    train_data, test_data = get_data_split()

    # component 3: retrieve examples for the domain combinations
    examples = load_examples(train_data)
