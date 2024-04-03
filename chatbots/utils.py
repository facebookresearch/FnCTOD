import os
import requests
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time
import json
import torch
from typing import Any, Dict, List, Union   
from transformers import (AutoModelForCausalLM, 
                          AutoModel, 
                          AutoTokenizer, 
                          AutoConfig)


import openai
import anthropic

from chatbots.configs import llm_configs
from chatbots.conversation import Conversation


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory
  

def load_hf_model(model_name_or_path, dtype=torch.float16, int8=False, reserve_memory=10):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # int8: whether to use int8 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Llama: set up the root dir
    # open_source_models = ["llama", "alpaca", "vicuna", "oasst"]
    # if any([m in model_name_or_path for m in open_source_models]):
    #     model_name_or_path = os.path.join(os.environ["LLAMA_ROOT"], model_name_or_path)

    # Load the FP16 model
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    if int8:
        logger.warn("Use LLM.int8")
        
    start_time = time.time()
    if "mpt" in model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
            )
        config.attn_config['attn_impl'] = 'triton'

        model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        # cache_dir=os.environ["TRANSFORMERS_CACHE"],
        trust_remote_code=True
        )
        model.cuda()
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    elif "llama" in model_name_or_path:
        from transformers import LlamaForCausalLM  
        model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=dtype,
                max_memory=get_max_memory(),
                load_in_8bit=int8,
                # cache_dir=os.environ["TRANSFORMERS_CACHE"],
                trust_remote_code=True
        )
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    elif "oasst" in model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
            )
        config.pad_token_id = config.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                # device_map="auto",
                torch_dtype=dtype,
                max_memory=get_max_memory(),
                load_in_8bit=int8,
                # cache_dir=os.environ["TRANSFORMERS_CACHE"],
                trust_remote_code=True
        )
        from transformers import GPTNeoXTokenizerFast
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name_or_path)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=dtype,
                max_memory=get_max_memory(),
                load_in_8bit=int8,
                # cache_dir=os.environ["TRANSFORMERS_CACHE"],
                trust_remote_code=True
            )
        except Exception as error:
            print(error)
            model = AutoModel.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=dtype,
                max_memory=get_max_memory(),
                load_in_8bit=int8,
                # cache_dir=os.environ["TRANSFORMERS_CACHE"],
                trust_remote_code=True
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Fix OPT bos token problem in HF
    if "opt" in model_name_or_path:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"

    return model, tokenizer


class OpenAI():
    def __init__(self, model="gpt-3.5-turbo", interval=0.5, timeout=10.0, exp=2, patience=10, max_interval=4, max_prompt_length=4096):
        self.model = model
        self.interval = interval
        self.timeout = timeout
        self.base_timeout = timeout
        self.patience = patience
        self.exp = exp
        self.max_prompt_length = max_prompt_length
        self.max_interval = max_interval
        from transformers import GPT2TokenizerFast
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def generate(
        self, messages, functions=[], function_call={}, temperature=1.0, top_p=1.0, max_tokens=64, n=1, 
        frequency_penalty=0, presence_penalty=0, stop=["Q:"], **kwargs) -> list[str]:

        openai.api_key = os.environ.get('OPENAI_API_KEY', None)
        
        # stop words
        if isinstance(stop, List):
            pass
        elif isinstance(stop, str):
            stop = [stop]

        retry_interval_exp = 1 
        t1 = time.time()

        while True and retry_interval_exp <= self.patience:
            try:
                params = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'n': n,
                    'top_p': top_p,
                    'frequency_penalty': frequency_penalty,
                    'presence_penalty': presence_penalty,
                    'stop': stop,
                    'request_timeout': self.timeout  # timeout!
                }
                if functions:
                    params['functions'] = functions
                if function_call:
                    params['function_call'] = function_call

                # call the function
                response = openai.ChatCompletion.create(**params)
                candidates = response["choices"]
                candidates = [candidate["message"] for candidate in candidates]
                return candidates

            # except openai.error.RateLimitError as e:
            except Exception as e:
                # logging.warning("OpenAI rate limit error. Retry")
                logging.warning(e)
                # Exponential backoff
                time.sleep(max(self.max_interval, self.interval * (self.exp ** retry_interval_exp)))
                retry_interval_exp += 1
        
        return [""] * n
   

class Claude():
    def __init__(self, model="claude-2.1", interval=1.0, timeout=10.0, exp=2, patience=10, max_interval=4, max_prompt_length=4096):
        self.model = model
        self.interval = interval
        self.timeout = timeout
        self.base_timeout = timeout
        self.patience = patience
        self.exp = exp
        self.max_prompt_length = max_prompt_length
        self.max_interval = max_interval

    def generate(
        self, prompt, temperature=1.0, top_p=1.0, max_tokens=64, n=1, 
        frequency_penalty=0, presence_penalty=0, stop=["Q:"], rstrip=False,
        **kwargs) -> list[str]:

        client = anthropic.Client(os.environ['ANTHROPIC_API_KEY'])

        if rstrip:
            prompt = prompt.rstrip()

        retry_interval_exp = 1 
        t1 = time.time()

        while True and retry_interval_exp <= self.patience:
            try:
                assert isinstance(prompt, str)
                # MAKE SURE THE PROMPT STARTS WITH HUMAN_PROMPT AND CONTAIN AI_PROMPT
                if not prompt.startswith(anthropic.HUMAN_PROMPT):
                    prompt = f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"
                # GENERATION
                response = client.completion(model=self.model,
                                             prompt=prompt,                       
                                             max_tokens_to_sample=max_tokens,
                                                    )  
                candidates = [response["completion"]]

                t2 = time.time()

                return candidates

            # except openai.error.RateLimitError as e:
            except Exception as e:
                # logging.warning("OpenAI rate limit error. Retry")
                logging.warning(e)
                # Exponential backoff
                time.sleep(max(self.max_interval, self.interval * (self.exp ** retry_interval_exp)))
                retry_interval_exp += 1
        
        return [""] * n
