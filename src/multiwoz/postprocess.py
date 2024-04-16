import os
import numpy as np
import pandas as pd
import json
import random
import copy
import yaml
from tqdm import trange
from tqdm import tqdm
import argparse

from src.multiwoz.utils import *
from src.multiwoz.utils.config import *
from src.multiwoz.utils.reader import *
from src.multiwoz.utils.utils import (paser_aspn_to_dict, 
                                              paser_dict_to_bs,
                                              paser_bs_to_dict,
                                              paser_dict_to_list,
                                              dialog_acts,
                                              requestable_slots,
                                              informable_slots,
                                              all_reqslot,
                                              all_slots,
                                              all_domain
                                              )
from src.utils import add_bracket, remove_bracket


# add actions
act_descriptions = {
    "inform": "provide information about an entity (if multiple matched results exist, choose one) in the form of [value_xxx] if requested by the user (required)",
    "request": "inform the number of available offers ([value_choice]) and ask the user for more preference on the requested entity to narrow down the search results (optional)",
    "recommend": "recommend an offer to the user and provide its information (optional)",
    "select": "ask the user to choose among available offers (optional)",
    "offerbook": "offer an entity to the user (if multiple matched results exist, choose one) and ask him if he would like to book it (required)",
    "offerbooked": "inform the user that the offer has been successfully booked and provide its booking information such as [value_name], [value_id], and [value_reference] (required)",
    "nooffer": "inform the user that no suitable offer could be found",
    "nobook": "inform the user that the offer can not be booked",
    "general": "greet and welcome the user, inquire if there is anything else they need help with after completing a requested service, and say goodbye to the user if they have everything they need"
}

def normalize_domain_slot(schema):
    normalized_schema = []
    for service in schema:
        if service['service_name'] == 'bus':
            # service['service_name'] = 'taxi'
            continue

        slots = service['slots']
        normalized_slots = []

        for slot in slots: # split domain-slots to domains and slots
            domain_slot = slot['name']
            domain, slot_name = domain_slot.split('-')
            if domain == 'bus':
                domain = 'taxi'
            if slot_name == 'bookstay':
                slot_name = 'stay'
            if slot_name == 'bookday':
                slot_name = 'day'
            if slot_name == 'bookpeople':
                slot_name = 'people'
            if slot_name == 'booktime':
                slot_name = 'time'
            if slot_name == 'arriveby':
                slot_name = 'arrive'
            if slot_name == 'leaveat':
                slot_name = 'leave'
            if slot_name == 'ref':
                slot_name = 'reference'
            domain_slot = "-".join([domain, slot_name])
            slot['name'] = domain_slot
            normalized_slots.append(slot)

        service['slots'] = normalized_slots
        normalized_schema.append(service)

    return normalized_schema


def load_schema(dataset_version="2.1"):

    if dataset_version == "2.0":
        processed_data_path = "multi-woz-2.0-final"
    elif dataset_version == "2.1":
        processed_data_path = "multi-woz-2.1-final"
    elif dataset_version == "2.2":
        processed_data_path = "multi-woz-2.2-final"
    elif dataset_version == "2.3":
        processed_data_path = "multi-woz-2.3-final"
    else:
        raise NotImplementedError

    processed_data_path = f"./data/multiwoz/data/{processed_data_path}"
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    data_path = f"{processed_data_path}/normalized_schema.yml"

    if not os.path.exists(data_path):
        with open(f"./data/multiwoz/schema.json", "r") as file:
            schema = json.load(file)

        normalized_schema = normalize_domain_slot(schema)

        my_schema = []
        for service in normalized_schema:
            service_name = service['service_name']
            service_desc = service["description"]
            slots = service['slots']
            for slot in service['slots']:
                domain, slot_name = slot['name'].split("-")

                if slot_name in informable_slots[domain]:
                    slot["is_informable"] = True
                else:
                    slot["is_informable"] = False

                if slot_name in requestable_slots[domain]:
                    slot["is_requestable"] = True
                else:
                    slot["is_requestable"] = False

            intents = service['intents']
            actions = []
            for act in dialog_acts[service_name]+['general']:
                actions.append(
                    {
                        "name": act,
                        "description": act_descriptions[act]
                    }
                )

            my_schema.append(
                {
                    "service_name": service_name,
                    "description": service_desc,
                    "slots": slots,
                    "intents": intents,
                    "actions": actions
                }
            )

        # add the general domain  
        general = {
            "service_name": "general",
            "slots": [],
            "intent": [],
            "actions": [
                {
                    "name": "greet",
                    "description": "greet the user"
                },
                {
                    "name": "welcome",
                    "description": "welcome the user"
                },
                {
                    "name": "reqmore",
                    "description": "inquire if there is anything else they need help with after completing a requested service"
                },
                {
                    "name": "bye",
                    "description": "say goodbye to the user if they have everything they need"
                }
            ]
        }
        my_schema.append(general)

        normalized_schema = my_schema
        with open(data_path, 'w') as file:
            yaml.dump(normalized_schema, file, sort_keys=False)

    else:
        # update the schema
        with open(data_path, 'r') as file:
            normalized_schema = yaml.safe_load(file)

        # # update the action descriptions
        # for service in normalized_schema:
        #     for action in service['actions']:
        #         action['description'] = act_descriptions[action['name']]
        # with open(data_path, 'w') as file:
        #     yaml.dump(normalized_schema, file, sort_keys=False)

    return normalized_schema


def compare_dict(old_dict, new_dict):
    differ = {}
    for domain, new_slots in new_dict.items():
        if domain not in old_dict:
            # differ.extend([f"{domain}-{slot}" for slot in slots.keys()])
            differ[domain] = new_dict[domain]
        else:
            old_slots = old_dict[domain]
            for slot in new_slots: 
                update = False
                if slot not in old_slots:
                    update = True
                elif new_slots[slot] != old_slots[slot]:
                    update = True

                if update:
                    if domain not in differ:
                        differ[domain] = {}
                        differ[domain][slot] = new_slots[slot]
                    else:
                        differ[domain][slot] = new_slots[slot]
    return differ


def process_data(data, reader):

    # Start
    processed_data = {}

    for dial_id in data:

        turns = data[dial_id]
        processed_turns = []
        for turn_id, turn in enumerate(turns):
            
            processed_turn = {}
            for key in ['dial_id', 'turn_num', 'user', 'resp', 'nodelx_resp',
                        'bspn', 'bspn_dict', 'turn_bspn', 'turn_bspn_dict', 'bsdx', 'db',
                        'dspn', 'aspn', 'aspn_dict', 'all_domains']:
                if key in turn:
                    processed_turn[key] = turn[key]
                else:
                    processed_turn[key] = ""

            # parsing
            processed_turn["bspn_dict"] = paser_bs_to_dict(turn["bspn"])
            processed_turn["aspn_dict"] = paser_aspn_to_dict(turn["aspn"])

            # belief state update in each turn
            if turn_id == 0:
                turn_bspn = processed_turn["bspn"]
                turn_bspn_dict = processed_turn["bspn_dict"]
            else:
                turn_bspn_dict = compare_dict(old_dict=paser_bs_to_dict(turns[turn_id-1]["bspn"]),
                                              new_dict=paser_bs_to_dict(turns[turn_id]["bspn"]))
                turn_bspn = paser_dict_to_bs(turn_bspn_dict)
            processed_turn["turn_bspn"] = turn_bspn
            processed_turn["turn_bspn_dict"] = turn_bspn_dict

            # db results
            bs_dict = paser_bs_to_dict(turn["bspn"])
            bs_dict = remove_bracket(bs_dict)
            db = len(reader.db.get_match_num(bs_dict, return_entry=True))
            processed_turn["db"] = db

            # all domains till this turn
            domains = set([t["dspn"] for t in turns[:turn_id]]+[turn['dspn']])
            processed_turn["all_domains"] = list(domains)

            # if this is the end of a dialog
            # processed_turn["end_of_dialog"] = True if turn_id+1 == len(turns) else False

            # save data
            processed_turns.append(processed_turn)
        
        processed_data[dial_id] = processed_turns

    return processed_data


def unzip_session_data(data):
    dialog_turns = []
    for dial_id, turns in data.items():
        dialog_turns.extend(turns)
    return dialog_turns


def zip_session_data(data):
    dial_data = {}
    for turn in data:
        dial_id = turn["dial_id"]
        if dial_id in dial_data:
            dial_data[dial_id].append(turn)
        else:
            dial_data[dial_id] = [turn]
    # sort the turns in each dialog
    for dial_id, dial_turns in dial_data.items():
        sorted_turns = sorted(dial_turns, key=lambda d: d["turn_num"])
        dial_data[dial_id] = sorted_turns
    return dial_data
    

def sample_data_ids(data, n):
   
    domain_freq = {
                    "[taxi]": 0,
                    "[hotel]": 0,
                    "[train]": 0,
                    "[restaurant]": 0,
                    "[attraction]": 0,
                    "[general]": 0
                    }

    all_ids = list(data.keys())
    n = min(len(all_ids), n)
    sampled_ids = random.sample(all_ids, n)
    for sampled_id in sampled_ids:
        turns = data[sampled_id]
        for turn in turns:
            if turn["dspn"] in domain_freq:
                domain_freq[turn["dspn"]] += 1
    print(domain_freq)

    return sampled_ids


def get_data_split(dataset_version, reader, n_train=-1, n_val=-1, n_test=-1, return_list=False):

    if dataset_version == "2.0":
        rawdata_path = "multi-woz-2.0-rawdata"
        processed_data_path = "multi-woz-2.0-final"
    elif dataset_version == "2.1":
        rawdata_path = "multi-woz-2.1-rawdata"
        processed_data_path = "multi-woz-2.1-final"
    elif dataset_version == "2.2":
        rawdata_path = "multi-woz-2.2-rawdata"
        processed_data_path = "multi-woz-2.2-final"
    elif dataset_version == "2.3":
        rawdata_path = "multi-woz-2.3-rawdata"
        processed_data_path = "multi-woz-2.3-final"
    else:
        raise NotImplementedError
    rawdata_path = f"./data/multiwoz/data/{rawdata_path}"
    processed_data_path = f"./data/multiwoz/data/{processed_data_path}"

    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    # training data
    if not os.path.exists(f"{processed_data_path}/train_raw_dials.json"):
        with open(f"{rawdata_path}/train_raw_dials.json", 'r') as file:
            train_data = json.load(file)
            train_data = process_data(train_data, reader)
        with open(f"{processed_data_path}/train_raw_dials.json", 'w') as file:
            json.dump(train_data, file, indent=4)
    else:
        with open(f"{processed_data_path}/train_raw_dials.json", 'r') as file:
            train_data = json.load(file)
    
    # val data
    if not os.path.exists(f"{processed_data_path}/dev_raw_dials.json"):
        with open(f"{rawdata_path}/dev_raw_dials.json", 'r') as file:
            val_data = json.load(file)
            val_data = process_data(val_data, reader)
        with open(f"{processed_data_path}/dev_raw_dials.json", 'w') as file:
            json.dump(val_data, file, indent=4)
    else:
        with open(f"{processed_data_path}/dev_raw_dials.json", 'r') as file:
            val_data = json.load(file)

    # test data
    if not os.path.exists(f"{processed_data_path}/test_raw_dials.json"):
        with open(f"{rawdata_path}/test_raw_dials.json", 'r') as file:
            test_data = json.load(file)
            test_data = process_data(test_data, reader)
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

    if n_val != -1:
        if not os.path.exists(f"{processed_data_path}/dev_{n_val}_ids.json"):
            val_data_ids = sample_data_ids(val_data, n_val)
            if n_val < len(val_data): # only record the sampled ids is not the full set
                with open(f"{processed_data_path}/dev_{n_val}_ids.json", 'w') as file:
                    json.dump(val_data_ids, file)
        else:
            with open(f"{processed_data_path}/dev_{n_val}_ids.json", 'r') as file:
                val_data_ids = json.load(file)

        sampled_val_data = {}
        for did in val_data_ids:
            sampled_val_data[did] = val_data[did]
        val_data = sampled_val_data

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
        return unzip_session_data(train_data), unzip_session_data(val_data), unzip_session_data(test_data)
    else:
        return train_data, val_data, test_data


def retrieve_demo(dialogs, domains, n=5, max_turns=6, bs_da_ratios=[0.6, 0.4]):
    
    # step 1: select the satisfied demos
    filtered_demos = []
    for dial_id, turns in dialogs.items():
        turn = turns[-1]
        if "[general]" not in domains:
            mentioned_domains = set(turn["all_domains"])
            mentioned_domains.discard("[general]")
            if set(domains) == mentioned_domains:
                filtered_demos.append(dial_id) 
        else: # xxx + [general]
            mentioned_domains = turn["all_domains"][-2:]
            if set(domains) == set(mentioned_domains):
                filtered_demos.append(dial_id) 

    if len(filtered_demos) == 0:
        print(f"No examples for {domains}")

    """
    Demo selection
    """
    bs_ratio, da_ratio = bs_da_ratios

    # select required dialog acts for the domains
    covered_da_list, covered_bs_list = [], []
    selected_demos, selected_demo_ids = [], []

    for i in range(n):
        
        if len(filtered_demos) == 0: break

        # measure score
        demo_scores, covered_bss, covered_das = [], [], []
        for dial_id in filtered_demos:
            
            covered_da, covered_bs = [], []
            if dial_id in selected_demo_ids:
                demo_score = -100
            else:
                turns = dialogs[dial_id]
                # mentioned belief states, dialog acts
                for turn in turns:
                    da_dict = paser_aspn_to_dict(turn["aspn"])
                    da_list = paser_dict_to_list(da_dict, level=2)
                    for da in da_list:
                        if da not in covered_da_list:
                            covered_da.append(da)
                        
                    bs_dict = paser_aspn_to_dict(turn["bspn"])
                    bs_list = paser_dict_to_list(bs_dict, level=2)
                    for bs in bs_list:
                        if bs not in covered_bs_list:
                            covered_bs.append(bs)
            
                covered_da = list(set(covered_da))
                covered_bs = list(set(covered_bs))
                
                # diversity
                demo_score = da_ratio*len(covered_da) + bs_ratio*len(covered_bs)

                # length penality
                turn_num = len(turns)
                lp = (turn_num / max_turns) - 1
                lp = max(0, lp)
                lp = np.exp(-lp)
                demo_score *= lp

            demo_scores.append(demo_score)
            covered_bss.append(covered_bs)
            covered_das.append(covered_da)
            
        # select
        selected_demo_id = filtered_demos[np.argmax(demo_scores)]
        selected_demo_ids.append(selected_demo_id)
        selected_demos.append(dialogs[selected_demo_id])
        filtered_demos.remove(selected_demo_id)

        # update 
        covered_bs = covered_bss[np.argmax(demo_scores)]
        covered_da = covered_das[np.argmax(demo_scores)]

        for bs in covered_bs:
            if bs not in covered_bs_list:
                covered_bs_list.append(bs)

        for da in covered_da:
            if da not in covered_da_list:
                covered_da_list.append(da)

    return selected_demos


def load_examples(dataset_version, data):

    if dataset_version == "2.0":
        processed_data_path = "multi-woz-2.0-final"
    elif dataset_version == "2.1":
        processed_data_path = "multi-woz-2.1-final"
    elif dataset_version == "2.2":
        processed_data_path = "multi-woz-2.2-final"
    elif dataset_version == "2.3":
        processed_data_path = "multi-woz-2.3-final"
    else:
        raise NotImplementedError

    example_data_path = f"./data/multiwoz/data/{processed_data_path}/examples.json"

    if not os.path.exists(example_data_path):
        
        combinations1 = all_domain
        combinations2 = []
        for i in range(len(all_domain)-1):
            for j in range(i+1, len(all_domain)):
                combinations2.append(all_domain[i]+"+"+all_domain[j])
        general_combinations = [f"{domain}+[general]" for domain in all_domain]
        combinations = combinations1+combinations2+general_combinations

        # print(combinations)

        all_examples = {}
        for comb in combinations:
            domains = comb.split("+")
            examples = retrieve_demo(data, domains, n=5)
            all_examples[comb] = examples

        with open(example_data_path, "w") as file:
            json.dump(all_examples, file, indent=4)
    
    else:
        with open(example_data_path, "r") as file:
            all_examples = json.load(file)
    
    return all_examples

        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='multiwoz') #
    parser.add_argument('--split', type=str, default='test') #
    parser.add_argument('--dataset_version', type=str, default='2.1', choices=['2.0', '2.1', '2.3']) #
    parser.add_argument('--task', type=str, default='dst', choices=['dst', 'nlg']) #

    args, unknown = parser.parse_known_args()

    # load configuration file and reader (for database query)
    data_prefix = "./data/multiwoz/data/"
    if args.dataset_version == "2.0":
        cfg = Config20(data_prefix)
    elif args.dataset_version == "2.1":
        cfg = Config21(data_prefix)
    elif args.dataset_version == "2.3":
        cfg = Config23(data_prefix)
    reader = MultiWozReader(tokenizer=None, cfg=cfg, data_mode=args.split)

    # component 1: process the normalized_schema.yml
    normalized_schema = load_schema(args.dataset_version)
    
    # component 2: post-process the dialogue data
    train_data, val_data, test_data = get_data_split(reader,
                                                    n_train=10000,
                                                    n_val=100,
                                                    n_test=100
                                                    )

    # component 3: retrieve examples for the domain combinations              
    examples = load_examples(train_data)
    
    domain_combinations = {}
    for dial_id in train_data:
        all_domains = train_data[dial_id][-1]["all_domains"]
        domain_comb = "+".join(all_domains)
        if domain_comb in domain_combinations:
            domain_combinations[domain_comb] += 1
        else:
            domain_combinations[domain_comb] = 1
    
    for key, value in domain_combinations.items():
        print(key, value)




                        