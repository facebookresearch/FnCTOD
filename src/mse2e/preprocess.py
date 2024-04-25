#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Import from pptod
"""


def split_session_list(in_f):
    with open(in_f, "r", encoding="utf8") as i:
        lines = i.readlines()[1:]
    curr_track_id = 1
    all_session_list = []
    one_session_list = []
    for line in lines:
        item_list = line.strip("\n").split("\t")
        curr_sess_id = int(item_list[0])
        if curr_sess_id != curr_track_id:
            all_session_list.append(one_session_list)
            one_session_list = [line.strip("\n")]
            curr_track_id = curr_sess_id  # update track id
        else:
            one_session_list.append(line.strip("\n"))
    if len(one_session_list) > 0:
        all_session_list.append(one_session_list)
    return all_session_list


def build_session_list(sess_text_list):
    zip_turn_list = []
    one_turn_list = []
    target_speaker = "user"
    target_map = {"user": "agent", "agent": "user"}
    for sess_text in sess_text_list:
        if sess_text.strip("\n").split("\t")[3] == target_speaker:
            target_speaker = target_map[sess_text.strip("\n").split("\t")[3]]
            one_turn_list.append(sess_text)
            if len(one_turn_list) == 2:
                zip_turn_list.append(one_turn_list)
                one_turn_list = []
        else:
            continue
    return zip_turn_list


def parse_usr_goal(text):
    item_list = text.split("(")
    assert len(item_list) >= 2
    tuple_list = item_list[1].strip().strip(")").split(";")
    bs_list = []
    for one_tuple in tuple_list:
        one_tuple_split = one_tuple.split("=")
        if len(one_tuple_split) == 1:
            continue
        bs_list.append((one_tuple_split[0].strip(), one_tuple_split[1].strip()))
    return bs_list


def update_belief_state(prev_bs_list, text):
    res_list = prev_bs_list.copy()
    prev_slot_set = set()
    for item in res_list:
        prev_slot_set.add(item[0])

    res_slot_set = prev_slot_set.copy()
    curr_bs_list = parse_usr_goal(text)
    for item in curr_bs_list:
        if item[0] in prev_slot_set:
            continue
        res_list.append(item)
    return res_list


def update_belief_state(prev_bs_dict, prev_bs_name_list, text):
    res_bs_dict = prev_bs_dict.copy()
    res_bs_name_list = prev_bs_name_list.copy()
    try:
        curr_bs_list = parse_usr_goal(text)
    except AssertionError:
        # print (text)
        raise Exception()
    for item in curr_bs_list:
        slot, value = item
        if slot in res_bs_dict:
            res_bs_dict[slot] = value  # update value
        else:
            res_bs_name_list.append(slot)
            res_bs_dict[slot] = value  # add new value
    return res_bs_dict, res_bs_name_list


def parse_usr_belief_state(prev_bs_dict, prev_bs_name_list, text, domain):
    split_list = text.split("\t")[5:]
    # bs_dict, bs_name_list = {}, []
    bs_dict, bs_name_list = prev_bs_dict.copy(), prev_bs_name_list.copy()
    for text in split_list:
        # print (text)
        if len(text) == 0:
            break
        bs_dict, bs_name_list = update_belief_state(bs_dict, bs_name_list, text)
    bs_text = ""
    bsdx_text = ""
    for name in bs_name_list:
        # try:
        #     slot = token_map_dict[name]
        # except KeyError:
        #     slot = name
        slot = name
        bs_text += slot + " " + bs_dict[name] + " "
        bsdx_text += slot + " "
    bs_text = bs_text.strip().strip(",").strip()
    bsdx_text = bsdx_text.strip().strip(",").strip()
    if len(bs_text) == 0:
        bs_text = ""
    else:
        bs_text = domain + " " + bs_text
    if len(bsdx_text) == 0:
        bsdx_text = ""
    else:
        bsdx_text = domain + " " + bsdx_text.strip()
    return (
        " ".join(bs_text.split()).strip(),
        " ".join(bsdx_text.split()).strip(),
        bs_dict,
        bs_name_list,
    )


def parse_one_agent_action(text):
    item_list = text.split("(")
    assert len(item_list) >= 2
    action_type = "[" + item_list[0].strip() + "]"
    action_text = action_type + " "
    tuple_list = item_list[1].strip().strip(")").split(";")
    action_list = []
    for one_tuple in tuple_list:
        one_action = one_tuple.split("=")[0].strip()
        # try:
        #     one_action = token_map_dict[one_action]
        # except KeyError:
        #     one_action = one_action
        action_list.append(one_action)
    return action_type, action_list


def parse_agent_action(text, domain):
    split_list = text.split("\t")[5:]
    res_list = []
    action_dict, action_type_list = {}, []
    for text in split_list:
        if len(text) == 0:
            break
        else:
            one_action_type, one_action_list = parse_one_agent_action(text)
            try:
                for a in one_action_list:
                    if a in action_dict[one_action_type]:
                        pass
                    else:
                        action_dict[one_action_type].append(a)
            except KeyError:
                action_type_list.append(one_action_type)
                action_dict[one_action_type] = one_action_list
    res_text = domain + " "
    for key in action_type_list:
        res_text += key + " "
        one_list = action_dict[key]
        for item in one_list:
            res_text += item + " "
        res_text = res_text.strip().strip(",").strip() + " "
    return " ".join(res_text.split()).strip(), action_dict


def zip_turn(prev_bs_dict, prev_bs_name_list, turn_list, domain):
    usr_text, agent_text = turn_list
    try:
        assert usr_text.strip("\n").split("\t")[3] == "user"
        assert agent_text.strip("\n").split("\t")[3] == "agent"
    except:
        raise Exception()
    usr_uttr = usr_text.strip("\n").split("\t")[4].strip()
    usr_bs, usr_bsdx, bs_dict, bs_name_list = parse_usr_belief_state(
        prev_bs_dict, prev_bs_name_list, usr_text, domain
    )
    system_uttr = agent_text.strip("\n").split("\t")[4].strip()
    system_action_text, system_action_dict = parse_agent_action(agent_text, domain)
    return (
        usr_uttr,
        usr_bs,
        usr_bsdx,
        system_uttr,
        system_action_text,
        system_action_dict,
        bs_dict,
        bs_name_list,
    )


def process_session_list(session_list, domain):
    turn_num = len(session_list)
    if turn_num == 0:
        raise Exception()
    res_dict = {"dataset": "E2E_MS", "dialogue_session": []}
    for idx in range(turn_num):
        if idx == 0:
            bs_dict, bs_name_list = {}, []
        one_turn_list = session_list[idx]
        (
            one_usr_uttr,
            one_usr_bs,
            one_usr_bsdx,
            one_system_uttr,
            one_system_action,
            da_dict,
            bs_dict,
            bs_name_list,
        ) = zip_turn(bs_dict, bs_name_list, one_turn_list, domain)

        one_turn_dict = {"turn_num": idx}
        one_turn_dict["user"] = one_usr_uttr
        one_turn_dict["resp"] = one_system_uttr
        one_turn_dict["turn_domain"] = [domain]
        one_turn_dict["bspn"] = one_usr_bs
        one_turn_dict["bsdx"] = one_usr_bsdx
        one_turn_dict["aspn"] = one_system_action
        one_turn_dict["bspn_dict"] = {}
        one_turn_dict["bspn_dict"][domain] = bs_dict
        one_turn_dict["aspn_dict"] = {}
        one_turn_dict["aspn_dict"][domain] = da_dict

        res_dict["dialogue_session"].append(one_turn_dict)
    return res_dict


def process_file(in_f, domain):
    all_session_list = split_session_list(in_f)
    res_list = []
    for item in all_session_list:
        one_sess = build_session_list(item)
        if len(one_sess) == 0:
            continue
        one_res_dict = process_session_list(one_sess, domain)
        res_list.append(one_res_dict)
    print(len(res_list), len(all_session_list))
    return res_list


if __name__ == "__main__":
    print("Processing MSE2E Dataset...")
    in_f = (
        r"./data/pre-training_corpora/raw_data/e2e_dialog_challenge/data/taxi_all.tsv"
    )
    domain = "[taxi]"
    taxi_res_list = process_file(in_f, domain)

    in_f = (
        r"./data/pre-training_corpora/raw_data/e2e_dialog_challenge/data/movie_all.tsv"
    )
    domain = "[movie]"
    movie_res_list = process_file(in_f, domain)

    in_f = r"./data/pre-training_corpora/raw_data/e2e_dialog_challenge/data/restaurant_all.tsv"
    domain = "[restaurant]"
    restaurant_res_list = process_file(in_f, domain)

    all_data_list = taxi_res_list + movie_res_list + restaurant_res_list
    print(len(all_data_list))

    import random

    random.shuffle(all_data_list)
    test_data_list = all_data_list[:500]
    train_data_list = all_data_list[500:]

    import random
    import json
    import os

    save_path = r"./data/pre-training_corpora/separate_datasets/MS_E2E/"
    if os.path.exists(save_path):
        pass
    else:  # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    import json

    out_f = save_path + r"/e2e_ms_train.json"
    with open(out_f, "w") as outfile:
        json.dump(train_data_list, outfile, indent=4)

    out_f = save_path + r"/e2e_ms_test.json"
    with open(out_f, "w") as outfile:
        json.dump(test_data_list, outfile, indent=4)
    print("Processing MSE2E Dataset Finished!")

    # collect databse
    import pickle

    db_prefix = "./data/pre-training_corpora/raw_data/e2e_dialog_challenge/system/src/deep_dialog/"
    kb = {"[movie]": [], "[taxi]": [], "[restaurant]": []}

    # movie domain
    for kb_path in [
        f"{db_prefix}/data_movie/movie_kb.1k.p",
        f"{db_prefix}/data_movie/movie_kb.v2.p",
        f"{db_prefix}/data_movie/movie.kb.1k.v1.p",
    ]:
        with open(kb_path, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            p = u.load()
        kb["[movie]"].extend([entity for p, entity in p.items()])

    # restaurant domain
    for kb_path in [
        f"{db_prefix}/data_restaurant/restaurant.kb.1k.v1.p",
        f"{db_prefix}/data_restaurant/restaurant.kb.2k.v1.p",
        f"{db_prefix}/data_restaurant/restaurant.kb.nondup.v1.p",
    ]:
        with open(kb_path, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            p = u.load()
        kb["[restaurant]"].extend([entity for p, entity in p.items()])

    # taxi domain
    for kb_path in [
        f"{db_prefix}/data_taxi/taxi.kb.1k.v1.p",
        f"{db_prefix}/data_taxi/taxi.kb.2k.v1.p",
        f"{db_prefix}/data_taxi/taxi.kb.v2.nondup.p",
    ]:
        with open(kb_path, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            p = u.load()
        kb["[taxi]"].extend([entity for p, entity in p.items()])

    save_kb_path = (
        r"./data/pre-training_corpora/separate_datasets/MS_E2E/MS_E2E_DB.json"
    )
    with open(save_kb_path, "w") as file:
        json.dump(kb, file, indent=4)
    print("Processing MSE2E Dataset Database Finished!")

    # collect ontology
    all_domain = ["[restaurant]", "[movie]", "[taxi]"]
    ontology = {}
    for domain in all_domain:
        ontology[domain] = {}
        ontology[domain]["slots"] = {}
        ontology[domain]["intents"] = {}
        ontology[domain]["actions"] = {}

    for data in all_data_list:
        turns = data["dialogue_session"]
        for turn in turns:
            # belief state ontologies
            bs_dict = turn["bspn_dict"]
            if bs_dict:
                for domain in all_domain:
                    if domain in bs_dict:
                        slots = bs_dict[domain]
                        for slot, slot_value in slots.items():
                            if slot in ontology[domain]["slots"]:
                                if slot_value not in ontology[domain]["slots"][slot]:
                                    ontology[domain]["slots"][slot].append(slot_value)
                            else:
                                ontology[domain]["slots"][slot] = [slot_value]

            # dialog act ontologies
            da_dict = turn["aspn_dict"]
            if da_dict:
                for domain in all_domain:
                    if domain in da_dict:
                        acts = da_dict[domain]
                        for act, act_values in acts.items():
                            if act in ontology[domain]["actions"]:
                                for act_value in act_values:
                                    if (
                                        act_value
                                        not in ontology[domain]["actions"][act]
                                    ):
                                        ontology[domain]["actions"][act].append(
                                            act_value
                                        )
                            else:
                                ontology[domain]["actions"][act] = act_values

    # all slots
    print("All slots:")
    for domain in all_domain:
        print(domain)
        print("-" * 50)
        all_slots = list(ontology[domain]["slots"].keys())
        print(all_slots)
        print("-" * 50)

    # all actions
    print("All actions:")
    for domain in all_domain:
        print(domain)
        print("-" * 50)
        all_actions = list(ontology[domain]["actions"].keys())
        print(all_actions)
        print("-" * 50)

    # save ontology
    out_f = r"./data/pre-training_corpora/separate_datasets/MS_E2E/ontology.json"
    with open(out_f, "w") as outfile:
        json.dump(ontology, outfile, indent=4)
    print("Processing MSE2E Dataset Ontology Finished!")

    # # print slots
    # domain = "[restaurant]"
    # print(domain)
    # print("-"*50)
    # all_slots = list(ontology[domain]["slots"].keys())
    # for slot in all_slots:
    #     print(slot+":")
    #     print(ontology[domain]["slots"][slot])
    #     _ = input("*"*50)

    # print actions
    # domain = "[taxi]"
    # print(domain)
    # print("-"*50)
    # all_actions = list(ontology[domain]["actions"].keys())
    # for act in all_actions:
    #     print(act+":")
    #     print(ontology[domain]["actions"][act])
    #     _ = input("*"*50)
