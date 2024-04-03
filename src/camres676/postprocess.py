import os
import numpy as np
import pandas as pd
import json
import yaml
import re
from tqdm import trange
from tqdm import tqdm
import argparse

from src.multiwoz.utils.utils import (paser_aspn_to_dict, 
                                              paser_dict_to_bs,
                                              paser_bs_to_dict,
                                              paser_dict_to_list,
                                              )
from src.utils import add_bracket, remove_bracket
from src.multiwoz.postprocess import (compare_dict,
                                                unzip_session_data,
                                                zip_session_data,
                                                sample_data_ids)
"""
CamRest676 is a subset of MultiWOZ, with only restaurant domains.
Its ontology is a subset of those of MultiWOZ.

Ontology:
slots:
food, area, pricerange
actions:
request, inform, recommend, nooffer, general

TODO
- reuse the schema of multiwoz
- add db exact information and delex response
- only examples for the restaurant domain
"""

class Reader(object):
    def __init__(self, data_prefix):
        # load db
        db_path = f"{data_prefix}/CamRestDB.json"
        assert os.path.exists(db_path)

        with open(db_path, "r") as file:
            self.database = json.load(file)

    def get_match_num(self, constrains):

        if not constrains:
            return []

        matched_results = []
        for d in self.database:
            satisfied = True
            for slot, slot_value in constrains["restaurant"].items():
                if slot_value and slot in d:
                    if d[slot] != slot_value:
                        satisfied = False
                        break
                elif slot not in d:
                    satisfied = False
                    break

            if satisfied:
                matched_results.append(d)
        return matched_results


def find_substring(x, y):
    x_lower = x.lower()
    y_lower = y.lower()

    if x_lower in y_lower:
        return y_lower.find(x_lower)
    else:
        return -1

def find_phone_number(text):
    # Regular expression pattern to match the format 5 digits + 6 digits, len=12
    pattern = r'\d{5} \d{6}'
    match = re.search(pattern, text)

    if match:
        return match.start()  # Returns the start index of the first match
    else:
        return -1

def find_postcode(text):
    # Regular expression pattern to match the format
    pattern = r'[A-Z]\.[A-Z] \d, \d [A-Z]\.[A-Z]'
    match = re.search(pattern, text)

    if match:
        return match.start()  # Returns the start index of the first match
    else:
        return -1

def get_delex_text(text, reader, num_db):
    
    # name & address
    for d in reader.database:
        name = d["name"]
        address = d["address"]

        ind = find_substring(name, text)
        if ind >= 0:
            text = text[:ind] + "[value_name]" + text[ind+len(name):]

        ind = find_substring(address, text)
        if ind >= 0:
            text = text[:ind] + "[value_address]" + text[ind+len(address):]

    # phone
    ind = find_phone_number(text)
    if ind >= 0: # total 12 digits
        text = text[:ind] + "[value_phone]" + text[ind+12:]
    
    # postcode
    ind = find_postcode(text)
    if ind >= 0: # total 12 digits, e.g., C.B 2, 1 A.B
        text = text[:ind] + "[value_postcode]" + text[ind+12:]

    # number of choice
    db_str = f" {num_db} "
    ind = find_substring(db_str, text)
    if num_db > 0 and ind >= 0:
        text = text[:ind] + " [value_choice] " + text[ind+len(db_str):]


    return text


def load_schema():

    processed_data_path = f"./data/pre-training_corpora/processed_data/CamRes676"
    
    data_path = f"{processed_data_path}/normalized_schema.yml"
    assert os.path.exists(data_path)
    with open(data_path, 'r') as yaml_file:
        schema = yaml.safe_load(yaml_file)

    return schema


def process_data(data, split, reader):

    # Start
    processed_data = {}

    for dial_idx, dial in enumerate(tqdm(data)):

        dial_id = f"{split}_{dial_idx}"
        turns = dial["dialogue_session"]
        processed_turns = []

        for turn_id, turn in enumerate(turns):
            processed_turn = {}
            for key in ['dial_id', 'turn_num', 'user', 'resp', 'nodelx_resp', 'dspn',
                        'bspn', 'bspn_dict', 'turn_bspn', 'turn_bspn_dict', 'bsdx', 'db', 
                        'db_ids', 'aspn', 'aspn_dict', 'turn_domain', 'all_domains']:

                if key in turn:
                    processed_turn[key] = turn[key]
                else:
                    processed_turn[key] = ""

            if not processed_turn["dial_id"]:
                processed_turn["dial_id"] = dial_id
                
            # parsing
            processed_turn["bspn_dict"] = paser_bs_to_dict(turn["bspn"])
            processed_turn["aspn_dict"] = paser_aspn_to_dict(turn["aspn"])

            # belief state update
            if turn_id == 0:
                turn_bspn = turn["bspn"]
                turn_bspn_dict = processed_turn["bspn_dict"]
            else:
                turn_bspn_dict = compare_dict(old_dict=paser_bs_to_dict(turns[turn_id-1]["bspn"]),
                                              new_dict=paser_bs_to_dict(turns[turn_id]["bspn"]))
                turn_bspn = paser_dict_to_bs(turn_bspn_dict)
            processed_turn["turn_bspn"] = turn_bspn
            processed_turn["turn_bspn_dict"] = turn_bspn_dict

            # db results
            bs_dict = paser_bs_to_dict(turn["bspn"])
            bs_dict = remove_bracket(bs_dict, level=1)
            dbs = reader.get_match_num(bs_dict)
            num_db = len(dbs)
            processed_turn["db"] = num_db
            processed_turn["db_ids"] = [db["id"] for db in dbs]

            # delexicalized response
            nodelx_resp = turn["resp"]
            delx_resp = get_delex_text(nodelx_resp, reader, num_db)
            processed_turn["nodelx_resp"] = nodelx_resp
            processed_turn["resp"] = delx_resp

            # domains, only restaurant
            processed_turn["dspn"] = "[restaurant]"
            processed_turn["turn_domain"] = ["[restaurant]"]
            processed_turn["all_domains"] = ["[restaurant]"]

            # save data
            processed_turns.append(processed_turn)
        
        processed_data[dial_id] = processed_turns

    return processed_data


def get_data_split(reader, n_train=-1, n_test=-1, return_list=False):

    preprocessed_data_path = f"./data/pre-training_corpora/separate_datasets/CamRes676"
    processed_data_path = f"./data/pre-training_corpora/processed_data/CamRes676"
    
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    # training data
    if not os.path.exists(f"{processed_data_path}/train_raw_dials.json"):
        with open(f"{preprocessed_data_path}/camres676_train.json", 'r') as file:
            train_data = json.load(file)
            train_data = process_data(train_data, split="train", reader=reader)
        with open(f"{processed_data_path}/train_raw_dials.json", 'w') as file:
            json.dump(train_data, file, indent=4)
    else:
        with open(f"{processed_data_path}/train_raw_dials.json", 'r') as file:
            train_data = json.load(file)

    # test data
    if not os.path.exists(f"{processed_data_path}/test_raw_dials.json"):
        with open(f"{preprocessed_data_path}/camres676_test.json", 'r') as file:
            test_data = json.load(file)
            test_data = process_data(test_data, split="test", reader=reader)
        with open(f"{processed_data_path}/test_raw_dials.json", 'w') as file:
            json.dump(test_data, file, indent=4)
    else:
        with open(f"{processed_data_path}/test_raw_dials.json", 'r') as file:
            test_data = json.load(file)

    # randomly sampled data
    if n_train != -1:
        if not os.path.exists(f"{processed_data_path}/train_{n_train}_ids.json"):
            train_data_ids = sample_data_ids(train_data, n_train)
            if n_train < len(train_data): # only record the sampled ids is not the full set
                with open(f"{processed_data_path}/train_{n_train}_ids.json", 'w') as file:
                    json.dump(train_data_ids, file)
        else:
            with open(f"{processed_data_path}/train_{n_train}_ids.json", 'r') as file:
                train_data_ids = json.load(file)
        
        sampled_train_data = {}
        for did in train_data_ids:
            sampled_train_data[did] = train_data[did]
        train_data = sampled_train_data

    if n_test != -1:
        if not os.path.exists(f"{processed_data_path}/test_{n_test}_ids.json"):
            test_data_ids = sample_data_ids(test_data, n_test)
            if n_test < len(test_data): # only record the sampled ids is not the full set
                with open(f"{processed_data_path}/test_{n_test}_ids.json", 'w') as file:
                    json.dump(test_data_ids, file)
        else:
            with open(f"{processed_data_path}/test_{n_test}_ids.json", 'r') as file:
                test_data_ids = json.load(file)

        sampled_test_data = {}
        for did in test_data_ids:
            sampled_test_data[did] = test_data[did]
        test_data = sampled_test_data
    
    if return_list:
        return unzip_session_data(train_data), unzip_session_data(test_data)
    else:
        return train_data, test_data


def retrieve_demo(dialogs, schema, domains, n=100, max_turns=2, bs_da_ratios=[1, 0]):
    
    """
    Demo selection
    """
    all_slots = []
    # select the required slots
    for domain in domains:
        for service in schema:
            if service["service_name"] == domain:
                break
        for slot in service["slots"]:
            if slot["is_informable"]:
                all_slots.append(slot)

    # select required dialog acts for the domains
    bs_ratio, da_ratio = bs_da_ratios
    covered_da_list, covered_bs_list = [], []

    filtered_ids = []
    for dial_id, turns in dialogs.items():
        # if len(turns) <= max_turns:
        filtered_ids.append(dial_id)

    # measure score
    demo_scores = []
    for dial_id in filtered_ids:
        
        covered_bs = []

        turns = dialogs[dial_id]
        # mentioned belief states, dialog acts
        turn = turns[-1]
   
        bs_dict = paser_aspn_to_dict(turn["bspn"])
        bs_list = paser_dict_to_list(bs_dict, level=2)
        for bs in bs_list:
            if bs not in covered_bs_list:
                covered_bs.append(bs)
    
        covered_bs = list(set(covered_bs))
            
        # diversity
        bs_score = len(covered_bs) / len(all_slots) if len(all_slots) > 0 else 0
        demo_score = bs_ratio*bs_score

        # length penality
        turn_num = len(turns)
        lp = (turn_num / max_turns) - 1
        lp = max(0, lp)
        lp = np.exp(-lp)
        demo_score *= lp

        demo_scores.append(demo_score)

    # rank the dialogues, select the top n
    sorted_pairs = sorted(zip(filtered_ids, demo_scores), key=lambda pair: pair[1], reverse=True)[:n]
    selected_demo_ids = [pair[0] for pair in sorted_pairs]

    return selected_demo_ids


def load_examples(data, ratio=0.1, max_turns=2, bs_da_ratios=[1.0, 0.]):
    
    schema = load_schema()

    all_examples = {}
    data_size = len(data) # number of dialogs
    example_size = min(int(ratio*data_size), 100) # take 10% for demonstration examples
    
    domain = "[restaurant]"
    example_ids = retrieve_demo(data, 
                                schema,
                                [domain], # retrieve single-domain examples
                                n=example_size, # size for each domain
                                max_turns=max_turns, # the max number of turns
                                bs_da_ratios=bs_da_ratios # the ratio between bs and da diversity
                                )
    all_examples[domain] = [data[dial_id] for dial_id in example_ids]

    # exclude the examples from the data
    new_data = {}
    for dial_id in data:
        if dial_id not in example_ids:
            new_data[dial_id] = data[dial_id]

    return all_examples, new_data

        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='CamRes676') #
    parser.add_argument('--split', type=str, default='test') #
 
    args, unknown = parser.parse_known_args()
    
    reader = Reader(f"./data/pre-training_corpora/raw_data/CamRest676")

    # component 1: process the normalized_schema.yml
    normalized_schema = load_schema(args.dataset)
    
    # component 2: post-process the dialogue data
    train_data, test_data = get_data_split(args.dataset, reader)

    # component 3: retrieve examples for the domain combinations              
    # examples = load_examples(args.dataset, train_data)
    
    





                        