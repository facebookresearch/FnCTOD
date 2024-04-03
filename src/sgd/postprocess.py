import os
import numpy as np
import pandas as pd
import json
import copy
import yaml
import re
from tqdm import trange
from tqdm import tqdm
import argparse

from src.sgd.preprocess import (all_domain,
                                        all_acts)
from src.sgd.preprocess import extract_bracket_content
from src.multiwoz.postprocess import (compare_dict,
                                      unzip_session_data,
                                      zip_session_data,
                                      sample_data_ids,
                                      paser_dict_to_list)


action_description_dict = {
    'confirm': "Confirm the value of a slot before making a transactional service call",
    'goodbye': "End the dialogue.",
    'inform': "Inform the value for a slot to the user.",
    'inform_count': "Inform the number of items found that satisfy the user's request.",
    'notify_failure': "Inform the user that their request failed.",
    'notify_success': "Inform the user that their request was successful.",
    'offer': "Offer a certain value for a slot to the user. ",
    'offer_intent': "Offer a new intent to the user. Eg, \"Would you like to reserve a table?\". ",
    'request': "Request the value of a slot from the user.",
    'reqmore': "Asking the user if they need anything else."
}

slot_mapping_dict = {}
# slot_mapping_dict["Restaurants_1"] = {
#     "restaurant_name": "name",
#     "cuisine": "food"
# }
# slot_mapping_dict["Trains_1"] = {
#     "from": "departure",
#     "to": "destination",
#     "from_station": "departure_station",
#     "to_station": "destination_station",
#     "date_of_journey": "date",
#     "journey_start_time": "time",
# }
# slot_mapping_dict["Buses_1"] = {
#     "from_location": "departure",
#     "to_location": "destination",
#     "from_station": "departure_station",
#     "to_station": "destination_station",
#     "leaving_date": "day",
#     "leaving_time": "leave",
#     "travelers": "people",
# }
# slot_mapping_dict["Buses_2"] = {
#     "origin": "departure",
#     "origin_station_name": "departure_station",
#     "destination_station_name": "destination_station",    
#     "departure_date": "day",
#     "departure_time": "time",
#     "group_size": "people",
# }
# slot_mapping_dict["Buses_3"] = {
#     "departure_date": "day",
#     "departure_time": "time",
#     "num_passengers": "people"
# }
# slot_mapping_dict["Hotels_1"] = {
#     "number_of_rooms": "rooms",
#     "check_in_date": "day",
#     "has_laundry_service": "laundry",
#     "hotel_name": "name"
# }
# slot_mapping_dict["Hotels_2"] = {
#     "check_in_date": "check_in_day",
#     "check_out_date": "check_out_day",
#     "has_laundry_service": "laundry",
#     "hotel_name": "name"
# }
# slot_mapping_dict["Hotels_3"] = {
#     "number_of_rooms": "rooms",
#     "check_in_date": "check_in_day",
#     "check_out_date": "check_out_day",
#     "hotel_name": "name"
# }
# slot_mapping_dict["Hotels_4"] = {
#     "number_of_rooms": "rooms",
#     "check_in_date": "day",
#     "stay_length": "stay",
#     "place_name": "place"
# }
# slot_mapping_dict["Travel_1"] = {
#     "attraction_name": "name",
#     "category": "type",
# }
general_slot_mapping_dict = {
    "restaurant_name": "name",
    "hotel_name": "name",
    "phone_number": "phone",
    "street_address": "address",
    "price_range": "pricerange",
    "has_wifi": "internet",
    "star_rating": "stars",
    "passengers": "people",
    "number_of_adults": "people",
    "number_of_days": "stay",
    "category": "type",
    # "from": "departure",
    # "to": "destination",
    # "from_location": "departure_location",
    # "to_location": "destination_location",
    # "from_station": "departure_station",
    # "to_station": "destination_station",
    # "from_city": "departure_city",
    # "to_city": "destination_city",
    # "origin": "departure",
    # "origin_city": "departure_city",
}

domain_mapping_dict = {
    "Services_1": "salon",
    "Services_2": "dentist",
    "Services_3": "doctor",
    "Services_4": "therapist"
}

"""
Schema_Guided:
Domains: movie, restaurant, taxi

TODO
- reuse the schema of multiwoz
- add db exact information and delex response
- only examples for the restaurant domain
"""


def load_schema():

    raw_data_path = f"./data/pre-training_corpora/raw_data/dstc8-schema-guided-dialogue"
    processed_data_path = f"./data/pre-training_corpora/processed_data/Schema_Guided"
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path, exist_ok=True)

    schema_path = f"{processed_data_path}/normalized_schema.yml"

    if not os.path.exists(schema_path):
        
        schemas = []

        for split in ["train", "dev", "test"]:
            with open(f"{raw_data_path}/{split}/schema.json") as file:
                schema = json.load(file)
            for service in schema:
                if service not in schemas:
                    schemas.append(service)

        for service in schemas:

            # rename slots
            domain = service["service_name"]
            for slot in service["slots"]:
                slot_name = slot["name"]
                if domain in slot_mapping_dict:
                    if slot_name in slot_mapping_dict[domain]:
                        slot_name = slot_mapping_dict[domain][slot_name]
                if slot_name in general_slot_mapping_dict:
                    slot_name = general_slot_mapping_dict[slot_name]
                slot["name"] = slot_name

            # add actions
            service["actions"] = []
            for act, description in action_description_dict.items():
                action = {}
                action["name"] = act
                action["description"] = description
                service["actions"].append(action)
    
        # save data
        with open(schema_path, "w") as yaml_file:
            yaml.dump(schemas, yaml_file, sort_keys=False)
        
    else:
        with open(schema_path, 'r') as yaml_file:
            schemas = yaml.safe_load(yaml_file)

    return schemas


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

            for key in ['dial_id', 'turn_num', 'user', 'resp', 'nodelx_resp', 
                        'dspn', 'bspn', 'bspn_dict', 'turn_bspn_dict', 'bsdx', 'db',
                        'aspn', 'aspn_dict', 'turn_domain', 'all_domains']:

                if key in turn:
                    processed_turn[key] = turn[key]
                else:
                    processed_turn[key] = ""

            processed_turn["dial_id"] = dial_id
            
            # db
            processed_turn["db"] = len(turn["db"])
            processed_turn["db_results"] = turn["db"]
            
            # rename the slot in belief state
            cleaned_bspn_dict = {}
            for domain in turn["bspn_dict"]:
            
                cleaned_dict = {}
                for slot, value in turn["bspn_dict"][domain].items():
                    if domain[1:-1] in slot_mapping_dict:
                        if slot in slot_mapping_dict[domain[1:-1]]:
                            slot = slot_mapping_dict[domain[1:-1]][slot]
                    if slot in general_slot_mapping_dict:
                        slot = general_slot_mapping_dict[slot]
                    cleaned_dict[slot] = value

                # if domain[1:-1] in domain_mapping_dict:
                #     domain = "["+domain_mapping_dict[domain[1:-1]]+"]"
                cleaned_bspn_dict[domain] = cleaned_dict

            processed_turn["bspn_dict"] = cleaned_bspn_dict

            # rename the delex response based on slot name
            delx_resp = copy.deepcopy(turn["resp"])
            for domain in turn["aspn_dict"]:
                if domain[1:-1] in slot_mapping_dict:
                    for slot in slot_mapping_dict[domain[1:-1]]:
                        if f"[value_{slot}]" in delx_resp:
                            new_slot = slot_mapping_dict[domain[1:-1]][slot]
                            delx_resp = delx_resp.replace(f"[value_{slot}]", f"[value_{new_slot}]")
            for old_slot, new_slot in general_slot_mapping_dict.items():
                if f"[value_{old_slot}]" in delx_resp:
                    delx_resp = delx_resp.replace(f"[value_{old_slot}]", f"[value_{new_slot}]")
            processed_turn["resp"] = delx_resp

            # rename the slots in actions
            cleaned_aspn_dict = {}
            for domain in turn["aspn_dict"]:

                cleaned_dict = {}
                for act, slots in turn["aspn_dict"][domain].items():
                    cleaned_dict[act] = []
                    for slot in slots:
                        if domain[1:-1] in slot_mapping_dict:
                            if slot in slot_mapping_dict[domain[1:-1]]:
                                slot = slot_mapping_dict[domain[1:-1]][slot]
                        if slot in general_slot_mapping_dict:
                            slot = general_slot_mapping_dict[slot]
                        cleaned_dict[act].append(slot)
                    
                    # if domain[1:-1] in domain_mapping_dict:
                    #     domain = "["+domain_mapping_dict[domain[1:-1]]+"]"
                    cleaned_aspn_dict[domain] = cleaned_dict

            processed_turn["aspn_dict"] = cleaned_aspn_dict

            # belief state update
            if turn_id == 0:
                turn_dict = processed_turn["bspn_dict"]
            else:
                turn_dict = compare_dict(old_dict=processed_turns[-1]["bspn_dict"],
                                         new_dict=processed_turn["bspn_dict"])
            processed_turn["turn_bspn_dict"] = turn_dict
            
            if turn_dict:
                valid_turn += 1
            
            # all domains
            for domain in processed_turn["turn_domain"]:
                if domain not in dial_domains:
                    dial_domains.append(domain)
            processed_turn["all_domains"] = copy.deepcopy(dial_domains)
            # dspn, the real turn domain, extract from the aspn
            processed_turn["dspn"] = "["+extract_bracket_content(processed_turn["aspn"])[0]+"]"
            assert processed_turn["dspn"] in processed_turn["aspn_dict"]

            # save data
            processed_turns.append(processed_turn)
        
        # if valid_turn / (len(turns)) >= 0.2:
        processed_data[dial_id] = processed_turns

    return processed_data


def get_data_split(n_train=-1, n_val=-1, n_test=-1, return_list=False):

    preprocessed_data_path = f"./data/pre-training_corpora/separate_datasets/Schema_Guided"
    processed_data_path = f"./data/pre-training_corpora/processed_data/Schema_Guided"
    
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    # training data
    if not os.path.exists(f"{processed_data_path}/train_raw_dials.json"):
        with open(f"{preprocessed_data_path}/schema_guided_train.json", 'r') as file:
            train_data = json.load(file)
            train_data = process_data(train_data, split="train")
        with open(f"{processed_data_path}/train_raw_dials.json", 'w') as file:
            json.dump(train_data, file, indent=4)
    else:
        with open(f"{processed_data_path}/train_raw_dials.json", 'r') as file:
            train_data = json.load(file)

    # dev data
    if not os.path.exists(f"{processed_data_path}/dev_raw_dials.json"):
        with open(f"{preprocessed_data_path}/schema_guided_dev.json", 'r') as file:
            val_data = json.load(file)
            val_data = process_data(val_data, split="dev")
        with open(f"{processed_data_path}/dev_raw_dials.json", 'w') as file:
            json.dump(val_data, file, indent=4)
    else:
        with open(f"{processed_data_path}/dev_raw_dials.json", 'r') as file:
            val_data = json.load(file)

    # test data
    if not os.path.exists(f"{processed_data_path}/test_raw_dials.json"):
        with open(f"{preprocessed_data_path}/schema_guided_test.json", 'r') as file:
            test_data = json.load(file)
            test_data = process_data(test_data, split="test")
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


def retrieve_demo(dialogs, schema, domains, n=100, max_turns=3, bs_da_ratios=[0.8, 0.2]):
    
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
        demo_score = da_ratio*da_score + bs_ratio*bs_score

        # length penality
        turn_num = len(turns)
        lp = (turn_num / max_turns) - 1 # penalize if exceed 5 turns
        lp = max(0, lp)
        lp = np.exp(-lp)
        demo_score *= lp
        print(demo_score)

        demo_scores.append(demo_score)

    # rank the dialogues, select the top n
    sorted_pairs = sorted(zip(filtered_ids, demo_scores), key=lambda pair: pair[1], reverse=True)[:n]
    selected_demo_ids = [pair[0] for pair in sorted_pairs]

    return selected_demo_ids


def load_examples(data, ratio=0.1, max_turns=4, bs_da_ratios=[0.8, 0.2]):

    schema = load_schema()

    all_examples = {}
    all_example_ids = []
    data_size = len(data) # number of dialogs
    example_size = min(int(ratio*data_size), 1000) # take 10% for demonstration examples

    combinations1 = all_domain
    combinations2 = []
    for i in range(len(all_domain)-1):
        for j in range(i+1, len(all_domain)):
            combinations2.append(all_domain[i]+"+"+all_domain[j])
    combinations = combinations1+combinations2
    
    for combination in combinations:
        example_ids = retrieve_demo(data, 
                                    schema,
                                    combination.split("+"), # retrieve single-domain examples
                                    n=int(example_size//len(all_domain)), # size for each domain
                                    max_turns=max_turns, # the max number of turns
                                    bs_da_ratios=bs_da_ratios # the ratio between bs and da diversity
                                    )
        domain_examples = [data[dial_id] for dial_id in example_ids]
        all_examples[combination] = domain_examples
        all_example_ids.extend(example_ids)
        # print(f"{domain} example size: {len(example_ids)}")

    # exclude the examples from the data
    new_data = {}
    for dial_id in data:
        if dial_id not in all_example_ids:
            new_data[dial_id] = data[dial_id]

    return all_examples, new_data

        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='mse2e') #
    parser.add_argument('--split', type=str, default='test') #
 
    args, unknown = parser.parse_known_args()

    # component 1: process the normalized_schema.yml
    normalized_schema = load_schema()

    # component 2: post-process the dialogue data
    train_data, val_data, test_data = get_data_split()

    # component 3: retrieve examples for the domain combinations              
    examples = load_examples(train_data)
    
    





                        