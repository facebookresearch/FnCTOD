token_map_dict = {'arr_time_dst':'arrive time destination',
                 'arr_time_or':'arrive time origin',
                 'budget_ok':'is budget ok',
                 'count_amenities':'count amenities',
                 'count_category':'count category',
                 'count_dst_city':'count destination city',
                 'count_name':'count name',
                 'count_seat':'count seat',
                 'dep_time_dst':'departure time destination',
                 'dep_time_or':'departure time origin',
                 'dst_city':'destination city',
                 'dst_city_ok':'destination city ok',
                 'end_date':'end date',
                 'end_date_ok':'is end date ok',
                 'gst_rating':'guest rating',
                 'impl_anaphora':'impl anaphora',
                 'max_duration':'max duration',
                 'min_duration':'min duration',
                 'n_adults':'number of adults',
                 'n_adults_ok':'is number of adults ok',
                 'n_children':'number of children',
                 'or_city':'origin city',
                 'ref_anaphora':'ref anaphora',
                 'seat_ok':'is seat ok',
                 'str_date':'start date',
                 'str_date_ok':'is start date ok',
                 }

all_domain = ["[travel]"]
all_acts = ['[request]', '[no_result]', '[suggest]', '[hearmore]', '[sorry]', 
            '[confirm]', '[offer]', '[inform]', '[you_are_welcome]', '[affirm]', 
            '[greeting]', '[canthelp]', '[goodbye]', '[thankyou]', '[reject]', 
            '[negate]']

act_description_dict = {
    '[request]': 'ask for specific information',
    '[no_result]': 'indicate no results were found',
    '[suggest]': 'suggest offers or ideas',
    '[hearmore]': 'inquire about whether more details are needed',
    '[sorry]': 'express regret or apology',
    '[confirm]': 'verify information',
    '[offer]': 'provide options',
    '[inform]': 'provide information',
    '[you_are_welcome]': 'responding to gratitude or thanks',
    '[affirm]': 'confirm or agree with a statement',
    '[greeting]': 'say hello or welcome',
    '[canthelp]': 'indicate inability to help',
    '[goodbye]': 'end the conversation',
    '[thankyou]': 'express gratitude',
    '[reject]': 'reject something',
    '[negate]': 'deny a statement'
}


def extract_wizard_act(wizard_dict, domain):
    # e.g. wizard_dict = data[0]['turns'][0]
    assert wizard_dict['author'] == 'wizard'
    acts = wizard_dict['labels']['acts']
    action_list = []
    action_type_dict = {}
    action_type_list = []
    for a in acts:
        action_type = '[' + a['name'] + ']'
        if action_type not in action_type_dict:
            action_type_list.append(action_type)
            action_type_dict[action_type] = []
        else:
            pass
            
        action_value = a['args']
        if len(action_value) == 0:
            pass
        else:
            if action_value[0]['key'] == 'ref':
                pass
            else:
                for item in action_value:
                    try:
                        slot = "_".join(token_map_dict[item['key']].split())
                    except KeyError:
                        slot = item['key']
                    if slot in action_type_dict[action_type]:
                        pass
                    else:
                        action_type_dict[action_type].append(slot)
    #print (action_type_dict)
    action_text = domain + ' '
    for a_type in action_type_list:
        one_text = a_type + ' '
        for a in action_type_dict[a_type]:
            one_text += a + ' '
        one_text = one_text.strip().strip(',').strip()
        action_text += one_text + ' '
    action_text = action_text.strip()
    action_text = ' '.join(action_text.split()).strip()
    return action_text, action_type_dict

def update_user_belief_state(prev_bs_dict, prev_bs_name_list, usr_dict):
    # e.g. usr_dict = data[1]['turns'][0]
    res_bs_dict, res_bs_name_list = prev_bs_dict.copy(), prev_bs_name_list.copy()
    #print (res_bs_dict)
    assert usr_dict['author'] == 'user'
    for item in usr_dict['labels']['acts']:
        
        value_dict_list = item['args']
        #print (value_dict_list)
        if len(value_dict_list) == 0:
            continue
        else:
            #print (value_dict_list)
            for value_dict in value_dict_list:
                try:
                    key, value = value_dict['key'], value_dict['val']
                except KeyError:
                    continue
                if key == 'ref':
                    continue
                try:
                    assert type(key) == str
                    assert type(value) == str
                except:
                    continue
                try:
                    key = "_".join(token_map_dict[key].split())
                except KeyError:
                    pass
                if key in res_bs_name_list:
                    pass
                else:
                    res_bs_name_list.append(key)
                res_bs_dict[key] = value # update user belief state
    return res_bs_dict, res_bs_name_list

def zip_turn(usr_dict, system_dict, prev_bs_dict, prev_bs_name_list, domain):
    usr_uttr = usr_dict['text']
    system_uttr = system_dict['text']
    res_bs_dict, res_bs_name_list = update_user_belief_state(prev_bs_dict, prev_bs_name_list, usr_dict)
    #print (res_bs_name_list)
    if len(res_bs_name_list) == 0:
        bs_text = ''
        bsdx_text = ''
    else:
        bs_text = domain + ' '
        bsdx_text = domain + ' '
        for slot in res_bs_name_list:
            bs_text += slot + ' ' + res_bs_dict[slot] + ' '
            bsdx_text += slot + ' '
        bs_text = bs_text.strip().strip(',').strip()
        bsdx_text = bsdx_text.strip().strip(',').strip()
        bs_text = ' '.join(bs_text.split()).strip()
        bsdx_text = ' '.join(bsdx_text.split()).strip()
    action_text, action_dict = extract_wizard_act(system_dict, domain)
    return usr_uttr, bs_text, bsdx_text, res_bs_dict, res_bs_name_list, system_uttr, action_text, action_dict

def build_session_list(in_item):
    raw_session_list = in_item['turns']
    zip_turn_list = []
    one_turn_list = []
    target_speaker = 'user'
    target_map = {'user':'wizard',
                 'wizard':'user'}
    for sess in raw_session_list:
        if sess['author'] == target_speaker:
            target_speaker = target_map[sess['author']]
            one_turn_list.append(sess)
            if len(one_turn_list) == 2:
                zip_turn_list.append(one_turn_list)
                one_turn_list = []
        else:
            continue
    return zip_turn_list

def process_session(session_list, domain):
    turn_num = len(session_list)
    
    res_dict = {'dataset':'Frames',
               'dialogue_session':[]}
    for idx in range(turn_num):
        usr_dict, system_dict = session_list[idx]
        if idx == 0:
            bs_dict, bs_name_list = {}, []
        usr_uttr, bs_text, bsdx_text, bs_dict, bs_name_list, system_uttr, action_text, action_dict = \
        zip_turn(usr_dict, system_dict, bs_dict, bs_name_list, domain)
        
        one_turn_dict = {'turn_num':idx}
        one_turn_dict['user'] = usr_uttr
        one_turn_dict['resp'] = system_uttr
        one_turn_dict['turn_domain'] = [domain]
        one_turn_dict['dspn'] = domain
        one_turn_dict['bspn'] = bs_text
        one_turn_dict['bsdx'] = bsdx_text
        one_turn_dict['aspn'] = action_text
    
        # clean the bs dict
        clean_bs_dict = {}
        clean_bs_dict[domain] = {}
        for slot, value in bs_dict.items():
            if slot not in ["intent", "action", "impl anaphora", "ref anaphora"]: 
                # slot = "_".join(slot.split(" "))
                clean_bs_dict[domain][slot] = value
        one_turn_dict['bspn_dict'] = clean_bs_dict

        # clean the action dict
        clean_da_dict = {}
        clean_da_dict[domain] = {}
        for act, slots in action_dict.items():
            # slots = ["_".join(slot.split(" ")) for slot in slots]
            clean_da_dict[domain][act] = slots
        one_turn_dict['aspn_dict'] = clean_da_dict

        res_dict['dialogue_session'].append(one_turn_dict)
    return res_dict

import json
def process_file(in_f, domain):
    with open(in_f) as f:
        data = json.load(f)
    res_list = []
    for item in data:
        sess_list = build_session_list(item)
        if len(sess_list) == 0:
            continue
        res_dict = process_session(sess_list, domain)
        res_list.append(res_dict)
    print (len(res_list), len(data))
    return res_list

def collect_ontology(all_data_list):
    ontology = {}
    domain_list = ["[travel]"]
    for domain in domain_list:
        ontology[domain] = {}
        ontology[domain]["slots"] = {}
        ontology[domain]["intents"] = {}
        ontology[domain]["actions"] = {}

    for data in all_data_list:
        turns = data["dialogue_session"]
        for turn in turns:

            # belief state ontologies
            bs_dict = turn["bspn_dict"]
            domain = turn["turn_domain"][0]
            if bs_dict:
                for domain in domain_list:
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
            domain = turn["turn_domain"][0]
            if da_dict:
                for domain in domain_list:
                    if domain in da_dict:
                        acts = da_dict[domain]
                        for act, slots in acts.items():
                            if act in ontology[domain]["actions"]:
                                pass
                            else:
                                ontology[domain]["actions"][act] = []

    return ontology


if __name__ == '__main__':
    print ('Processing Frame Dataset...')
    import random
    import json
    import os
    save_path = r'./data/pre-training_corpora/separate_datasets/Frames/'
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    in_f = r'./data/pre-training_corpora/raw_data/frames.json'
    domain = '[travel]'
    res_list = process_file(in_f, domain)
    random.shuffle(res_list)
    test_list = res_list[:100]
    train_list = res_list[100:]

    out_f = save_path + r'/frames_test.json'
    with open(out_f, 'w') as outfile:
        json.dump(test_list, outfile, indent=4)

    out_f = save_path + r'/frames_train.json'
    with open(out_f, 'w') as outfile:
        json.dump(train_list, outfile, indent=4)
    print ('Processing Frame Dataset Finished!')

    ontology = collect_ontology(train_list+test_list)

    processed_data_path = r'./data/pre-training_corpora/processed_data/Frames'
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path, exist_ok=True)

    out_f = processed_data_path + r'/ontology.json'
    with open(out_f, 'w') as outfile:
        json.dump(ontology, outfile, indent=4)

    print ('Processing Frame Dataset Ontology Finished!')
