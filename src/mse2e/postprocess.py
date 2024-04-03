import os
import numpy as np
import pandas as pd
import json
import copy
import yaml
from tqdm import trange
from tqdm import tqdm
import argparse

from src.multiwoz.postprocess import (compare_dict,
                                        unzip_session_data,
                                        sample_data_ids)
from src.multiwoz.utils.utils import paser_dict_to_list
from src.camres676.postprocess import find_substring

all_domain = ["[restaurant]", "[movie]",  "[taxi]"]

slot_schemas = {
    "[restaurant]": 
    {
        "alias": {
            "starttime": "time",
            "numberofpeople": "people",
            "restaurantname": "name",
            "restauranttype": "type",
            "personfullname": "guest",
            "distanceconstraints": "distance",
            "numberofkids": "kids",
            "cuisine": "food",
            "zip": "postcode",
            # "mealtype": "meal",
            # "pricing": "pricerange",
            # "date": "day"
            },
        "descriptions": {
            "numberofpeople": "number of guests",
            "numberofkids": "number of children",
            "cuisine": "cuisine preference",
            "city": "restaurant city",
            "state": "restaurant state",
            "zip": "restaurant zip",
            "starttime": "reservation time",
            "date": "reservation date",
            "restaurantname": "restaurant name",
            "restauranttype": "type of restaurant",
            "rating": "restaurant rating",
            "personfullname": "guest's full name",
            "food": "dish type",
            "mealtype": "meal type",
            "seating": "seating preference",
            "dress_code": "dress code",
            "occasion": "occasion type",
            "atmosphere": "atmosphere type",
            "distanceconstraints": "distance preference"
        },
        "included_slots": ["numberofpeople", "cuisine", "city", "starttime", "date", "state", 
                            "personfullname", "restaurantname", "restauranttype",
                            "food", "mealtype", "rating", "occasion", "address", "pricing", 
                            "atmosphere", "distanceconstraints", "zip", "seating",
                            "dress_code", "numberofkids"],
        "excluded_slots": ["greeting", "closing", "other", "choice", "result", "mc_list"]
    },
    "[movie]": 
    {
        "alias": {
            "numberofpeople": "people",
            "distanceconstraints": "distance",
            "numberofkids": "kids",
            "moviename": "name",
            "movie_series": "series",
            "zip": "postcode",
            },
        "descriptions": {
            "description": "movie description",
            "date": "preferred time for the movie",
            "starttime": "preferred time for the movie",
            "theater": "specific theater name",
            "city": "city",
            "state": "state",
            "zip": "zip",
            "moviename": "movie name",
            "numberofpeople": "number of tickets",
            "numberofkids": "number of tickets for kids",
            "mpaa_rating": "mpaa rating",
            "distanceconstraints": "distance preference",
            "critic_rating": "critic rating",
            "movie_series": "movie series",
            "theater_chain": "theater chain",
            "video_format": "video format",
            "actress": "lead actress",
            "actor": "lead actor",
            "seating": "seating preference",
            "price": "ticket price",
        },
        "included_slots": ["description", "date", "moviename", "numberofpeople", "theater", "city",
                           "state", "zip", "starttime", "actress", "genre", "video_format", "numberofkids",
                            "mpaa_rating", "distanceconstraints", "seating", "actor", "price", "critic_rating",
                            "movie_series", "theater_chain"],
        "excluded_slots": ["closing", "greeting", "other", "pickup_time", "pickup_location", "dropoff_location", 
                           "result", "food", "cuisine"]
    },       
    "[taxi]": 
    {
        "alias": {
            "numberofpeople": "people",
            "distanceconstraints": "distance",
            "pickup_location": "departure",
            "pickup_location_city": "departure_city",
            "dropoff_location": "destination",
            "dropoff_location_city": "destination_city",
            "pickup_time": "leave",
            "zip": "postcode",
            },
        "descriptions": {
            "numberofpeople": "number of passengers",
            "pickup_location": "pickup location",
            "pickup_location_city": "pickup city",
            "dropoff_location": "dropoff location",
            "dropoff_location_city": "dropoff city",
            "pickup_time": "pickup time",
            "car_type": "car type",
            "name": "passenger name",
            "city": "city",
            "state": "state",
            "zip": "zip",
            "date": "pickup date",
            "car_type": "cat type",
            "cost": "estimated fare range",
            "distanceconstraints": "distance preferences"
        },
        "included_slots": ["numberofpeople", "state", "pickup_location", "pickup_location_city", 
                            "dropoff_location", "dropoff_location_city", 
                            "date", "pickup_time", "car_type", "cost", "name", "city", "zip"],
        "excluded_slots": ["other", "greeting", "closing", "result"]
    }
}

act_schemas = {
    "[restaurant]": 
    {
        "alias": {},
        "descriptions": {
            "inform": "provide information",
            "request": "request information",
            "multiple_choice": "present options",
            "confirm_question": "verify the query",
            "confirm_answer": "confirm the response",
            "closing": "end the conversaiton",
            "not_sure": "admit uncertainty",
            },
        "included_actions": ["inform", "multiple_choice", "request", "greeting", "confirm_question", 
                            "thanks", "confirm_answer", "closing", "welcome", "deny", "not_sure"],
        "excluded_actions": [],
    },
    "[movie]": 
    {
        "alias": {},
        "descriptions": {
            "inform": "provide information",
            "request": "request information",
            "multiple_choice": "present options",
            "confirm_question": "verify the query",
            "confirm_answer": "confirm the response",
            "closing": "end the conversaiton",
            "not_sure": "admit uncertainty",
            },
        "included_actions": ["inform", "multiple_choice", "request", "greeting", "confirm_question", 
                            "thanks", "confirm_answer", "closing", "welcome", "deny", "not_sure"],
        "excluded_actions": [],
    },       
    "[taxi]": 
    {
        "alias": {},
        "descriptions": {
            "inform": "provide information",
            "request": "request information",
            "multiple_choice": "present options",
            "confirm_question": "verify the query",
            "confirm_answer": "confirm the response",
            "closing": "end the conversaiton",
            "not_sure": "admit uncertainty",
            },
        "included_actions": ['request', 'inform', 'confirm_answer', 'multiple_choice', 
                            'greeting', 'confirm_question', 'closing', 'thanks', 'welcome'],
        "excluded_actions": []
    }
}

class Reader(object):
    def __init__(self, data_prefix):
        # load db
        db_path = f"{data_prefix}/MS_E2E_DB.json"
        assert os.path.exists(db_path)

        with open(db_path, "r") as file:
            self.database = json.load(file)

    def get_match_num(self, constrains):

        if not constrains:
            return []

        matched_results = []

        for domain, slots in constrains.items():
            for d in self.database[domain]:
                satisfied = True
                for slot, slot_value in slots.items():
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


"""
MS_E2E:
Domains: movie, restaurant, taxi

TODO
- reuse the schema of multiwoz
- add db exact information and delex response
- only examples for the restaurant domain
"""

def get_delex_text(text, reader, ontology, domain):
    included_slots = {
        "[movie]": ["moviename", "genre", "theater", "zip"],
        "[restaurant]": ["restaurantname", "address"],
        "[taxi]": ["car_type", "cost"]
    }
    database = reader.database[domain]
    # search in the database
    for entity in database:
        for slot, value in entity.items():
            if slot in included_slots[domain] and len(value) > 2:
                substr = f" {value} "
                ind = find_substring(substr, text)
                if ind > 0:
                    if slot in slot_schemas[domain]["alias"]:
                        slot = slot_schemas[domain]["alias"][slot]
                    text = text[:ind] + f" [value_{slot}] " + text[ind+len(substr):]
    # search in the ontology
    for slot in included_slots[domain]:
        possible_values = ontology[domain]["slots"][slot]
        for value in possible_values:
            if len(value) > 2:
                substr = f" {value} "
                ind = find_substring(substr, text)
                if ind > 0:
                    if slot in slot_schemas[domain]["alias"]:
                        slot = slot_schemas[domain]["alias"][slot]
                    text = text[:ind] + f" [value_{slot}] " + text[ind+len(substr):]
    return text


def clean_belief_state(bs_dict):
    cleaned_bs_dict = {}
    for domain, slots in bs_dict.items():
        for slot, slot_value in slots.items():
            # make sure not in the exlcuded slots            
            if slot not in slot_schemas[domain]["excluded_slots"]:
                # rename
                if slot in slot_schemas[domain]["alias"]:
                    slot = slot_schemas[domain]["alias"][slot]

                if domain not in cleaned_bs_dict:
                    cleaned_bs_dict[domain] = {}
                cleaned_bs_dict[domain][slot] = slot_value

    return cleaned_bs_dict


def clean_dialog_act(da_dict):
    cleaned_da_dict = {}
    for domain, acts in da_dict.items():
        for act, slots in acts.items():
            act = act[1:-1]
            if act in act_schemas[domain]["alias"]:
                act = act_schemas[domain]["alias"][act]
            
            cleaned_slots = []
            for slot in slots:

                # make sure not in the exlcuded slots
                if slot not in slot_schemas[domain]["excluded_slots"]:
                    # rename
                    if slot in slot_schemas[domain]["alias"]:
                        slot = slot_schemas[domain]["alias"][slot]
                    cleaned_slots.append(slot)

            if domain not in cleaned_da_dict:
                cleaned_da_dict[domain] = {}
            cleaned_da_dict[domain][f"[{act}]"] = cleaned_slots
    return cleaned_da_dict


def load_schema():

    preprocessed_data_path = f"./data/pre-training_corpora/separate_datasets/MS_E2E"
    processed_data_path = f"./data/pre-training_corpora/processed_data/MS_E2E"
    schema_path = f"{processed_data_path}/normalized_schema.yml"
    ontology_path = f"{preprocessed_data_path}/ontology.json"

    if not os.path.exists(schema_path):

        with open(ontology_path, "r") as file:
            ontology = json.load(file)

        schema = []
        for domain in all_domain:
            service = {}
            service["service_name"] = domain[1:-1]
            
            # slots
            service["slots"] = []
            included_slots = slot_schemas[domain]["included_slots"]
            alias = slot_schemas[domain]["alias"]
            descriptions = slot_schemas[domain]["descriptions"]
            for slot in included_slots:
                # possible values
                possible_values = []
                for v in ontology[domain]["slots"][slot]:
                    # avoid values like {Burbank#Pasadena California}
                    if not (v.startswith("{") and v.endswith("}")) and v and \
                        v.lower() not in possible_values:
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
                # rename
                slot = alias[slot] if slot in alias else slot
                
                # append
                service["slots"].append(
                    {
                        "name": f"{domain[1:-1]}-{slot}",
                        "description": description if description else f"{domain[1:-1]} {slot}",
                        "possible_values": possible_values,
                        "is_categorical": is_categorical
                    }
                )
            
            # actions
            service["actions"] = []
            included_actions = act_schemas[domain]["included_actions"]
            alias = act_schemas[domain]["alias"]
            descriptions = act_schemas[domain]["descriptions"]
            for act in included_actions:
                # description
                description = descriptions[act] if act in descriptions else ""
                # rename
                act = alias[act] if act in alias else act
        
                # append
                service["actions"].append(
                    {
                        "name": act,
                        "description": description
                    }
                )
            
            schema.append(service)
        
        # save data
        with open(schema_path, "w") as yaml_file:
            yaml.dump(schema, yaml_file, sort_keys=False)
        
    else:
        with open(schema_path, 'r') as yaml_file:
            schema = yaml.safe_load(yaml_file)

    return schema


def process_data(data, split, reader, ontology):

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

            for key in ['dial_id', 'turn_num', 'user', 'resp', 'nodelx_resp', 'dspn',
                        'bspn', 'bspn_dict', 'turn_bspn_dict', 'bsdx', 'aspn', 'aspn_dict', 
                        'turn_domain', 'all_domains']:

                if key in turn:
                    processed_turn[key] = turn[key]
                else:
                    processed_turn[key] = ""
            
            processed_turn["dial_id"] = dial_id

            # domains
            if turn["turn_domain"][0] not in dial_domains:
                dial_domains.append(turn["turn_domain"][0])
            processed_turn["all_domains"] = copy.deepcopy(dial_domains)
            processed_turn["dspn"] = turn["turn_domain"][0]

            # delexicalized response
            nodelx_resp = turn["resp"]
            # TODO: delexicalized response
            # delex_resp = get_delex_text(nodelx_resp, reader, ontology, turn["turn_domain"][0])
            delex_resp = turn["resp"]
            processed_turn["nodelx_resp"] = nodelx_resp
            processed_turn["resp"] = delex_resp

            # clean belief state, dialog act
            bs_dict = turn["bspn_dict"]
            cleaned_bs_dict = clean_belief_state(bs_dict)
            processed_turn["bspn_dict"] = cleaned_bs_dict

            da_dict = turn["aspn_dict"]
            cleaned_da_dict = clean_dialog_act(da_dict)
            processed_turn["aspn_dict"] = cleaned_da_dict

            # belief state update
            if turn_id == 0:
                turn_dict = processed_turn["bspn_dict"]
            else:
                turn_dict = compare_dict(old_dict=processed_turns[-1]["bspn_dict"],
                                         new_dict=processed_turn["bspn_dict"])
            processed_turn["turn_bspn_dict"] = turn_dict
            
            if turn_dict:
                valid_turn += 1
            
            # save data
            processed_turns.append(processed_turn)
        
        if valid_turn / len(turns) >= 0.2:
            processed_data[dial_id] = processed_turns

    return processed_data


def get_data_split(reader, n_train=-1, n_test=-1, return_list=False):

    preprocessed_data_path = f"./data/pre-training_corpora/separate_datasets/MS_E2E"
    processed_data_path = f"./data/pre-training_corpora/processed_data/MS_E2E"
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    ontology_path = f"{preprocessed_data_path}/ontology.json"
    with open(ontology_path, "r") as file:
        ontology = json.load(file)

    # training data
    if not os.path.exists(f"{processed_data_path}/train_raw_dials.json"):
        with open(f"{preprocessed_data_path}/e2e_ms_train.json", 'r') as file:
            train_data = json.load(file)
            train_data = process_data(train_data, "train", reader, ontology)
        with open(f"{processed_data_path}/train_raw_dials.json", 'w') as file:
            json.dump(train_data, file, indent=4)
    else:
        with open(f"{processed_data_path}/train_raw_dials.json", 'r') as file:
            train_data = json.load(file)

    # test data
    if not os.path.exists(f"{processed_data_path}/test_raw_dials.json"):
        with open(f"{preprocessed_data_path}/e2e_ms_test.json", 'r') as file:
            test_data = json.load(file)
            test_data = process_data(test_data, "test", reader, ontology)
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
    
    print(f"Train dataset size: {len(train_data)}")
    print(f"Test dataset size: {len(test_data)}")

    if return_list:
        return unzip_session_data(train_data), unzip_session_data(test_data)
    else:
        return train_data, test_data


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

        # length penalty
        turn_num = len(turns)
        lp = (turn_num / max_turns) - 1
        lp = max(0, lp)
        lp = np.exp(-lp)
        demo_score *= lp
        print(demo_score)

        demo_scores.append(demo_score)

    # rank the dialogues, select the top n
    sorted_pairs = sorted(zip(filtered_ids, demo_scores), key=lambda pair: pair[1], reverse=True)[:n]
    selected_demo_ids = [pair[0] for pair in sorted_pairs]

    return selected_demo_ids


def load_examples(data, ratio=0.1, max_turns=3, bs_da_ratios=[0.8, 0.2]):

    schema = load_schema()
    all_examples = {}
    all_example_ids = []
    data_size = len(data) # number of dialogs
    example_size = min(int(ratio*data_size), 100) # take 10% for demonstration examples
    
    for domain in all_domain:
        example_ids = retrieve_demo(data, 
                                    schema,
                                    [domain], # retrive single-domain examples
                                    n=int(example_size//len(all_domain)), # size for each domain
                                    max_turns=max_turns, # the max number of turns
                                    bs_da_ratios=bs_da_ratios # the ratio between bs and da diversity
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
    parser.add_argument('--dataset', type=str, default='mse2e') #
    parser.add_argument('--split', type=str, default='test') #
 
    args, unknown = parser.parse_known_args()
    reader = Reader(f"./data/pre-training_corpora/separate_datasets/MS_E2E")

    # component 1: process the normalized_schema.yml
    normalized_schema = load_schema()

    # component 2: post-process the dialogue data
    train_data, test_data = get_data_split(reader)

    # component 3: retrieve examples for the domain combinations              
    examples = load_examples(train_data)
    
    





                        