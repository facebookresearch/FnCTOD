#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

def schema2function(service, rename_mapping={}):
    # convert the schema to the function call format in GPT-3.5/4.
    # https://openai.com/blog/function-calling-and-other-api-updates
    """
    {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            }
        }
    """

    function = {
        "name": "",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    service_name = service["service_name"]
    if service_name in rename_mapping:
        function["name"] = rename_mapping[service_name]
    else:
        function["name"] = service_name

    contain_name_parameter = False
    for slot in service["slots"]:
        slot_name = slot["name"].split("-")[1].strip()
        parameter = {}
        parameter["description"] = slot["description"]
        parameter["type"] = "string"
        if all([v in ["yes", "no"] for v in slot["possible_values"]]):
            parameter["type"] = "boolean"
        elif all([v.isdigit() for v in slot["possible_values"]]):
            parameter["type"] = "integer"

        if "possible_values" in slot:
            if slot["is_categorical"]:
                parameter["enum"] = [str(v) for v in slot["possible_values"]]

        if slot_name == "name":
            contain_name_parameter = True

        function["parameters"]["properties"][slot_name] = parameter

    function["parameters"]["required"] = []

    description = service["description"] + ". "
    CAUTIONS = [
        "Set the value as : 'dontcare' ONLY when the user EXPLICITLY states they have no specific preference for a parameter.",
    ]
    if contain_name_parameter:
        CAUTIONS.append(
            "Always record the exact value of the 'name' parameter when mentioned. Avoid using pronouns or coreferences like 'the hotel' or 'the restaurant.'"
        )
    description += " ".join(CAUTIONS)
    function["description"] = description

    return function
