# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


export TRANSFORMERS_CACHE='HOME_PATH/.cache/huggingface/transformers'
export HF_HOME='HOME_PATH/.cache/huggingface'

devices=0

cd ..

# gpt-3.5 gpt-4 vicuna-7b-v1.5 vicuna-13b-v1.5 llama-2-7b-chat llama-2-13b-chat llama-2-70b-chat alpaca-7b
CUDA_VISIBLE_DEVICES=$devices python -m LLM.app --model vicuna-7b-v1.3