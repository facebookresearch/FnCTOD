import argparse

############## Utilities ##############
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [item.strip() for item in v.split("+")]
    else:
        raise argparse.ArgumentTypeError('List value expected.')
    
def word2num(word):
    word_to_num = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9
    }
    return word_to_num.get(word)

def string2int(s):
    # Check if it's an integer
    if s.isdigit():
        return s
    # Check if it's a spelled out number
    num = word2num(s.lower())
    if num is not None:
        return num
    return None

def add_bracket(api_dict, level=1):
    if level == 1:
        return {f"[{key}]": value for key, value in api_dict.items()}
    elif level == 2:
        api_dict = {f"[{key}]": value for key, value in api_dict.items()}
        return {key: {f"[{sub_key}]": sub_value for sub_key, sub_value in value.items()} for key, value in api_dict.items()}
    else:
        raise NotImplementedError

def remove_bracket(api_dict, level=1):
    if level == 1:
        return {key[1:-1]: value for key, value in api_dict.items()}
    elif level == 2:
        api_dict = {key[1:-1]: value for key, value in api_dict.items()}
        return {key: {sub_key[1:-1]: sub_value for sub_key, sub_value in value.items()} for key, value in api_dict.items()}
    else:
        raise NotImplementedError
############################################


############## Configurations ##############
domain_prefix = "<domain>"
domain_suffix = "</domain>"

fc_prefix = "<function_call> "
fc_suffix = "  </function_call> "

tod_instructions = [
    "You are a task-oriented assistant. You can use the given functions to fetch further data to help the users.",
    "Your role as an AI assistant is to assist the users with the given functions if necessary.",
    "You are a task-oriented assistant, concentrating on assisting users with the given functions if necessary.",
    "You are a task-oriented assistant to provide users with support using the given functions if necessary.",
    "You are a task-focused AI. Your primary function is to help the users to finish their tasks using the given function(s) to gather more information if necessary.",
    "You are a task-oriented assistant. Your primary objective is assisting users to finish their tasks, using the given function(s) if necessary.",
    "Your primary role is to assist users using the given function (s), as a specialized task-oriented assistant.",
    "As an AI with a task-focused approach, your primary focus is assisting users to finish their tasks using the given functions."
]

tod_notes = [
    "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    "Use only the argument values explicitly provided or confirmed by the user instead of the assistant. Don't add or guess argument values.",
    "Ensure the accuracy of arguments when calling functions to effectively obtain information of entities requested by the user."
]
