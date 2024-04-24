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
        slot_name = slot["name"]
        parameter = {}
        parameter["description"] = slot["description"]
        parameter["type"] = "string"
        if any([v in slot["possible_values"] for v in ["yes", "no"]]):
            parameter["type"] = "boolean"
        elif any([v in slot["possible_values"] for v in ["1", "2", "3", "4", "5"]]):
            parameter["type"] = "integer"
        elif slot_name in ["time", "arrive", "leave"]:
            parameter["type"] = "time (HH:MM)"

        if "possible_values" in slot:
            if slot["is_categorical"]:
                parameter["enum"] = [str(v) for v in slot["possible_values"]]
            else:
                if slot["possible_values"]:
                    examples = ", ".join(slot["possible_values"][:10])
                    parameter["description"] += f" such as {examples}, etc."

        function["parameters"]["properties"][slot_name] = parameter
        if slot_name == "name":
            contain_name_parameter = True

    function["parameters"]["required"] = service["intents"][0]["required_slots"]

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
