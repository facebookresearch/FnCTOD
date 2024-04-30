#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import random
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.utils import *
from src.woz.postprocess import (
    get_data_split,
    load_schema,
)
from src.utils import *
from chatbots.utils import *
from chatbots.llm import *
from src.multiwoz.schema2function import schema2function
from src.multiwoz.inference import domain2function_mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )  #
    parser.add_argument(
        "--template",
        type=str,
        default="llama2",
        help="the conversation template of data",
    )  #
    parser.add_argument(
        "--all_turn",
        type=str2bool,
        default=False,
        help="if add all turns of function calls",
    )  #

    args, unknown = parser.parse_known_args()
    print(args)

    # load schema, examples, data
    schema = load_schema()
    train_data, test_data = get_data_split(return_list=False)

    if args.split == "train":
        data = train_data
    elif args.split == "test":
        data = test_data

    # save data path
    data_path = f"./data/pre-training_corpora/prompting_data/woz/"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    save_path = f"{data_path}/{args.split}-{args.template}-allturn{args.all_turn}.json"

    # the conversation template
    conversation = Conversation(
        template_name=args.template,
        function_type="json",
        function_call_prefix=fc_prefix,
        function_call_suffix=fc_suffix,
    )

    # all the processed dials for each task
    processed_data = []

    # prepare input and output for each task
    for dial_id, turns in data.items():
        functions = []
        for service in schema:
            if service["service_name"] == "restaurant":
                function = schema2function(
                    service, rename_mapping=domain2function_mapping
                )
                functions.append(function)
                break

        messages = []
        # system instruction
        system_messages = [random.choice(tod_instructions)]
        system_messages.extend(tod_notes)
        system_message = "\n".join(system_messages)
        messages.append({"role": "system", "content": system_message})

        # add conversation messages
        for turn in turns:
            usr = turn["user"]
            resp = turn["nodelx_resp"]
            messages.append({"role": "user", "content": usr})

            turn_domain = turn["dspn"]
            turn_bs_dict = turn["turn_bspn_dict"]
            bs_dict = turn["bspn_dict"]

            function_call_dict = {}
            if args.all_turn:  # add function call at all the turns
                if turn_domain in bs_dict:
                    function_call_dict = {
                        "function": domain2function_mapping[turn_domain[1:-1]],
                        "arguments": bs_dict[turn_domain],
                    }
            else:  # only add function call when there are update
                if turn_domain in turn_bs_dict:
                    function_call_dict = {
                        "function": domain2function_mapping[turn_domain[1:-1]],
                        "arguments": bs_dict[turn_domain],
                    }

            if function_call_dict:
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp,
                        "function_call": function_call_dict,
                    }
                )
            else:
                messages.append({"role": "assistant", "content": resp})

        # construct the prompt, exclude the conversation part
        system_prompt = conversation.get_prompt(
            system_message=system_message,
            functions=functions,
            messages=[],
        )

        # construct each turn of the conversation with function calls
        conversation_prompt = []
        for message in messages[1:]:
            turn_prompt = conversation.get_conversation([message], predict=False)
            conversation_prompt.append(
                {"role": message["role"], "content": turn_prompt}
            )

        processed_data.append(
            {
                "system": system_prompt,
                "functions": functions,
                "conversation": conversation_prompt,
            }
        )

    # summarize
    print(f"Total dialogues: {len(data)}!")
    print(f"Total samples: {len(processed_data)}")

    # save data
    with open(save_path, "w") as file:
        json.dump(processed_data, file, indent=4)
