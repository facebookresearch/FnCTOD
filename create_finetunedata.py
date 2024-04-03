import argparse
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

import json
import random
import time
import yaml
from transformers import LlamaTokenizer
from src.utils import *

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", default='meta-llama/Llama-2-13b-chat-hf', type=str)
    parser.add_argument("--configfile", default='', type=str)
    parser.add_argument("--outputfile", default='', type=str)
    parser.add_argument("--seed", type=int, default=1799)
    
    parser.add_argument("--valid_ratio", type=float, default=0.02)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--domain_size", type=int, default=-1)

    parser.add_argument('--noshuffle', action='store_true')

    return parser.parse_args()


def sample_data_based_on_len(tokenizer, cutoff_len, data, size):

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    sampled_data = []
    if size > 0:
        for dp in data:
            conversation = dp["conversation"]
            system = dp["system"]
            prompt = system
            for message in conversation:
                ###### user turn ######
                if message["role"] == "user":
                    user_input = message["content"]
                    prompt += user_input
                elif message["role"] == "assistant":
                    assistant_output = message["content"]
                    prompt += assistant_output

            tokenized_prompt = tokenize(
                prompt, add_eos_token=False)
            prompt_len = len(tokenized_prompt["input_ids"])
            if prompt_len <= cutoff_len:
                sampled_data.append(dp)

            if len(sampled_data) >= size:
                return sampled_data
    
    return sampled_data


def encode_tasks(args):

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)

    all_data = []
    config = yaml.safe_load(open(args.configfile, 'r'))
    for c in config:
        split = c["split"]
        dataset = c["dataset"]
        dataset_size = c["size"]
        template = c["template"]
        all_turn = c["all_turn"]

        # evaluation dataset
        data_prefix = f'./data/pre-training_corpora/prompting_data/{dataset}'
        data_path = f'{data_prefix}/{split}-{template}-allturn{all_turn}.json'
        
        with open(data_path, "r") as file:
            data = json.load(file)

        # randomly sample conversations
        dataset_size = min(len(data), dataset_size)
        # sample the data
        sampled_data = random.sample(data, dataset_size)

        for dp in sampled_data:
            dp["dataset"] = dataset
            dp["template"] = template
            all_data.append(dp)

        print(f"Dataset: {dataset}, Sampled size: {len(sampled_data)}")

    domains = {}
    for data in all_data:
        for function in [function["name"] for function in data["functions"]]:
            domain = "{}-{}".format(data["dataset"], function).lower()
            if domain not in domains:
                domains[domain] = [data]
            else:
                domains[domain].append(data)
            
    # select data for each domain based on diversity gain
    if args.domain_size > 0:
        for domain, domain_data in domains.items():
            # sample size
            if "multiwoz" in domain:
                max_domain_size = 10000
            elif "taskmaster" in domain and "restaurant" in domain: # not good quality
                max_domain_size = 0
            elif "restaurant" in domain: # restaurant
                max_domain_size = args.domain_size 
            elif "hotel" in domain: # hotel
                max_domain_size = args.domain_size 
            elif "attraction" in domain: # hotel
                max_domain_size = args.domain_size 
            elif "taxi" in domain and "mse2e" in domain: # taxi
                max_domain_size = args.domain_size 
            elif "bus" in domain: # train
                max_domain_size = args.domain_size 
            else:
                max_domain_size = args.domain_size 

            # sample data
            # sampled_domain_data = random.sample(domain_data, k=max_domain_size)
            sampled_domain_data = sample_data_based_on_len(tokenizer, args.max_len, domain_data, max_domain_size)

            domains[domain] = sampled_domain_data
            print(f"{domain}", f"full size: {len(domain_data)}", 
                               f"sampled size: {len(sampled_domain_data)}")
            # _ = input("continue.....")

        all_domain_data = []
        for domain, domain_data in domains.items():
            all_domain_data.extend(domain_data)
        all_data = all_domain_data        
    
    print(f"Domain num size: {len(domains)}")
    print(f"Total data size: {len(all_data)}")

    if not args.noshuffle:
        random.shuffle(all_data)

    # save all the data
    dest_file = args.outputfile
    output_file = open(dest_file, 'w', encoding='utf-8')
    json.dump(all_data, output_file, indent=4) 


if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)
    startTime = time.time()

    encode_tasks(args)
    
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))