import os
import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time
import json
import torch
from typing import Any, Dict, List, Union
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoConfig


import openai
import anthropic
import tiktoken

from chatbots.utils import *
from chatbots.configs import llm_configs
from chatbots.conversation import Conversation

"""
The main class for all LLMs: api-accessible gpt and claude, huggingface's llama-series and others
"""


class LLM:
    def __init__(
        self, model_name, interval=0.5, timeout=10.0, exp=2, patience=10, max_interval=4
    ):
        self.model_name = model_name
        self.openai_api = (
            True if any([x in self.model_name for x in ["gpt-3.5", "gpt-4"]]) else False
        )
        self.anthropic_api = True if "claude" in self.model_name else False

        # load model, either API-accessible or local models
        if self.openai_api:  # OPENAI API
            self.model = OpenAI(
                model=model_name,
                interval=interval,
                timeout=timeout,
                exp=exp,
                patience=patience,
                max_interval=max_interval,
            )
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif self.anthropic_api:  # CLAUDE API
            self.model = Claude(
                model=model_name,
                interval=interval,
                timeout=timeout,
                exp=exp,
                patience=patience,
                max_interval=max_interval,
            )
            self.tokenizer = None
        else:  # HUGGINGFACE MODELS
            print(model_name)
            self.model, self.tokenizer = load_hf_model(model_name)

    def generate(
        self,
        prompt,
        functions,  # only useful for openai or claude, otherwise have already included in prompt
        function_call,  # only useful for openai or claude, otherwise have already included in prompt
        temperature=0.5,
        top_p=0.5,
        max_tokens=128,
        n_seqs=1,
        stop=["\n\n", "User", "Example"],
    ):
        # api-accessible models (call api)
        if self.openai_api:  # the openai official api
            generations = self.model.generate(
                messages=prompt,
                functions=functions,
                function_call=function_call,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n_seqs,
                stop=stop,
            )

        elif self.anthropic_api:  # the openai official api (function calls in prompt)
            generations = self.model.generate(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n_seqs,
                stop=stop,
            )
        else:  # huggingface's models (# huggingface's models (local inference))
            inputs = self.tokenizer(
                [prompt],
                truncation=True,
                max_length=4096,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(self.model.device)
            stop = [] if stop is None else stop
            stop = list(
                set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])
            )  # In Llama \n is <0x0A>; In OPT \n is Ċ
            try:
                stop_token_ids = list(
                    set(
                        [
                            self.tokenizer._convert_token_to_id(stop_token)
                            for stop_token in stop
                        ]
                        + [self.model.config.eos_token_id]
                    )
                )
            except:  # some tokenizers don't have _convert_token_to_id function
                stop_token_ids = list(
                    set(
                        [
                            self.tokenizer.vocab.get(
                                stop_token, self.tokenizer.unk_token_id
                            )
                            for stop_token in stop
                        ]
                        + [self.model.config.eos_token_id]
                    )
                )

            if not self.tokenizer.unk_token_id:
                stop_token_ids.remove(self.tokenizer.unk_token_id)

            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                num_return_sequences=n_seqs,
                eos_token_id=stop_token_ids,
            )
            generations = [
                self.tokenizer.decode(
                    output[inputs["input_ids"].size(1) :], skip_special_tokens=True
                )
                for output in outputs
            ]

        return generations


"""
The wrapper for ChatCompletions
"""


class chat_completion(object):
    def __init__(
        self,
        model,
        api=False,
        system_message: str = "",
        system_template: str = "{system_message}",
        roles: List[str] = ["User", "Assistant"],
        offset: int = 20,
        colon: str = ": ",
        separators: List[str] = ["\n", "\n", "\n"],
        function_type: str = "json",
        function_call_prefix: str = "<function_call>",
        function_call_suffix: str = "</function_call>\n",
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.api = api
        model_name = llm_configs[model]["model_name"]
        self.model_name = model_name

        if self.api:  # docker api calling
            port = llm_configs[model]["port"]
            self.url = f"http://127.0.0.1:{port}/generate"
        else:  # local inference
            self.model = LLM(model_name=model_name)

        # default templates
        if "gpt-3.5" in model or "gpt-4" in model:
            template_name = "chatgpt"
        elif "claude" in model:
            template_name = "claude"
        elif "llama-2" in model and "-chat" in model:
            template_name = "llama2"
        elif "fnctod-llama2" in model:
            template_name = "llama2"
        elif "baichuan" in model and "-chat" in model:
            template_name = "baichuan2"
        elif "claude" in model:
            template_name = "claude"
        elif "vicuna" in model:
            template_name = "vicuna"
        elif "alpaca" in model:
            template_name = "alpaca"
        elif "baize" in model:
            template_name = "baize"
        elif "zephyr" in model:
            template_name = "zephyr"
        elif "openassistant" in model:
            template_name = "openassistant"
        self.template = template_name

        # the conversation template
        self.conversation = Conversation(
            template_name=template_name,
            system_template=system_template,
            system_message=system_message,
            roles=roles,
            offset=offset,
            colon=colon,
            function_type=function_type,
            function_call_prefix=function_call_prefix,
            function_call_suffix=function_call_suffix,
            separators=separators,
        )

    def complete(
        self,
        messages: List[Dict] = [],
        functions: List[Dict] = [],
        function_call: Dict = {},
        examples: List[Dict] = [],  # examples in the system instruction
        required: List[str] = ["function_call", "content"],
        temperature: float = 0.5,
        top_p: float = 0.5,
        max_tokens: int = 64,
        n_seqs: int = 1,
        stop: List[str] = ["\n\n"],  # ["\n\n", "###", "User", "Assistant", "Example"]
    ) -> List[str]:
        if self.template == "chatgpt":
            # system messages
            system_message = ""
            for message in messages:
                if message["role"] == "system":
                    system_message = message["content"]
                    break

            # construct system prompt with only examples
            system_prompt = self.conversation.get_prompt(
                system_message=system_message, examples=examples
            )

            # replace the original system message with the one with examples
            for message in messages:
                if message["role"] == "system":
                    message["content"] = system_prompt
                    break

            t1 = time.time()
            outputs = self.model.generate(
                prompt=messages,
                functions=functions,
                function_call=function_call,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n_seqs=n_seqs,
                stop=stop,
            )
            t2 = time.time()
            duration = t2 - t1

            # input tokens
            input_tokens = 0
            for message in messages:
                input_tokens += len(self.model.tokenizer.encode(message["content"]))

            # output tokens
            for idx, output in enumerate(outputs):
                output["duration"] = duration
                output["input_tokens"] = input_tokens
                outputs[idx] = output

        else:  # local inference
            # system messages
            system_message = ""
            for message in messages:
                if message["role"] == "system":
                    system_message = message["content"]
                    break

            # construct prompt from the user utterances
            prompt = self.conversation.get_prompt(
                system_message=system_message,
                messages=messages,
                functions=functions,
                function_call=function_call,
                examples=examples,
            )
            # input tokens
            input_tokens = len(self.model.tokenizer.encode(prompt))

            if self.verbose:
                print(prompt)

            t1 = time.time()
            if self.api:  # docker run
                data = {
                    "input": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "n_seqs": n_seqs,
                    "stop": stop,
                }
                response = requests.post(self.url, json=data)
                outputs = response.json()["generated_text"]
            else:
                outputs = self.model.generate(
                    prompt=prompt,
                    functions=functions,
                    function_call=function_call,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n_seqs=n_seqs,
                    stop=stop,
                )
            t2 = time.time()
            duration = t2 - t1

            # parse the output
            parsed_outputs = []
            for output in outputs:
                if self.verbose:
                    print("Before parsing:", output)
                parsed_output = self.conversation.get_response(
                    output, function_call, required=required
                )
                if self.verbose:
                    print("After parsing:", parsed_output)
                parsed_outputs.append(parsed_output)
            outputs = parsed_outputs

            # cost summary
            for idx, output in enumerate(outputs):
                output["duration"] = duration
                output["input_tokens"] = input_tokens
                outputs[idx] = output

        # chatml format: {"content": "xxx", "function": {}}
        return outputs


"""
The wrapper for TextCompletions
"""


class text_completion(object):
    def __init__(self, model, api=False, verbose=False):
        self.verbose = verbose
        self.api = api
        model_name = llm_configs[model]["model_name"]
        self.model_name = model_name

        if self.api:  # api calling
            port = llm_configs[model]["port"]
            self.url = f"http://127.0.0.1:{port}/generate"
        else:  # local inference
            self.model = LLM(model_name=model_name)

    def to_openai_chat_completion(self, input) -> list[dict[str, str]]:
        messages = [
            {
                "role": "user",
                "content": input,
            }
        ]
        return messages

    def to_claude_completion(self, input) -> list[dict[str, str]]:
        messages = [f"{anthropic.HUMAN_PROMPT} {input}", f"{anthropic.AI_PROMPT}"]
        return "\n\n".join(messages)

    def get_prompt(self, input):
        # construct prompt from the user utterances
        if any([x in self.model_name for x in ["gpt-3.5", "gpt-4"]]):  # ChatML
            prompt = self.to_openai_chat_completion(input=input)
        elif any([x in self.model_name for x in ["claude"]]):  # ChatML
            prompt = self.to_claude_completion(input=input)
        else:  # str
            prompt = input
        return prompt

    def complete(
        self,
        input: str,
        temperature: float = 0.5,
        top_p: float = 1.0,
        max_tokens: int = 64,
        n_seqs: int = 1,
        stop: List[str] = ["\n", "\n\n", "User", "Example"],
    ):
        prompt = self.get_prompt(input)

        if self.verbose:
            print(prompt)

        if self.api:  # docker api call
            data = {
                "input": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n_seqs": n_seqs,
                "stop": stop,
            }
            response = requests.post(self.url, json=data)
            outputs = response.json()["generated_text"]
        else:
            outputs = self.model.generate(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n_seqs=n_seqs,
                stop=stop,
            )

        if self.verbose:
            print(outputs)

        return outputs
