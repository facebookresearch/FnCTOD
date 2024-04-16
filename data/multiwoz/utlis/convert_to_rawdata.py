
import enum
import progressbar
import argparse
import logging
import time
import json
import random
import re
import copy
import os
import numpy as np

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration        
    parser.add_argument('--data_path_prefix', type=str, default='../data', help='The path where the data stores.')
    parser.add_argument('--data_version', type=str, default='2.1', help='The version of used multiwoz data, 2.0, 2.1, 2.3, 2.4')

    parser.add_argument('--model_name', type=str, default='t5-small', 
        help="the model type of t5, t5-small, t5-base, or t5-large.")

    parser.add_argument('--shuffle_mode', type=str, default='unshuffle', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")
    parser.add_argument('--use_db_as_input', type=str, default='True', 
        help="True or False, whether includes db result as part of the input when generating response.")
    parser.add_argument('--cascaded', type=str, default='False', 
        help="True or False, whether includes action when generating response.")
    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')
    
    parser.add_argument('--save_data_path_prefix', type=str, default='../data/multiwoz/data/multi-woz-2.1-rawdata', help='The path where the data stores.')

    return parser.parse_args()


import argparse
if __name__ == '__main__':
    args = parse_config()
    
    print ('Start loading data...')
    from dataclass import MultiWozData

    if args.data_version == "2.0":
        from config import Config

    elif args.data_version == "2.1":
        from config21 import Config

    else:
        raise Exception("Wrong MultiWOZ version!")

    cfg = Config(args.data_path_prefix)
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    if args.use_db_as_input == 'True':
        use_db_as_input = True
    elif args.use_db_as_input == 'False':
        use_db_as_input = False
    else:
        raise Exception('Wrong Use DB Mode!!!')

    if args.cascaded == 'True':
        cascaded = True
    elif args.cascaded == 'False':
        cascaded = False
    else:
        raise Exception('Wrong Use Cascaded Mode!!!')

    if args.add_prefix == 'True':
        add_prefix = True
    elif args.add_prefix == 'False':
        add_prefix = False
    else:
        raise Exception('Wrong Prefix Mode!!!')

    if args.add_special_decoder_token == 'True':
        add_special_decoder_token = True
    elif args.add_special_decoder_token == 'False':
        add_special_decoder_token = False
    else:
        raise Exception('Wrong Add Special Token Mode!!!')

    dialogs_analysis = {}

    # load dialogs
    data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
        data_mode='train', data_version=args.data_version, use_db_as_input=use_db_as_input, cascaded=cascaded, add_special_decoder_token=add_special_decoder_token)

    print("Start converting raw multiwoz data ......")
    train_dials, dev_dials, test_dials, train_dial_id_list, dev_dial_id_list, test_dial_id_list = data.convert_to_raw_data()

    print("Start saving raw multiwoz data......")
    # save the dialog turn info
    if not os.path.exists(args.save_data_path_prefix):
        os.makedirs(args.save_data_path_prefix)

    # save data
    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "train_raw_dials.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(train_dials, f)
    f.close()

    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "dev_raw_dials.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(dev_dials, f)
    f.close()

    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "test_raw_dials.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(test_dials, f)
    f.close()

    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "trainListFile.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(train_dial_id_list, f)
    f.close()

    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "valListFile.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(dev_dial_id_list, f)
    f.close()

    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "testListFile.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(test_dial_id_list, f)
    f.close()





