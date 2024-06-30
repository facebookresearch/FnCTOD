# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

export TRANSFORMERS_CACHE='HOME_PATH/.cache/huggingface/transformers/'

cd ..

devices=1,2

for model in llama-2-13b-chat-d5_allturnFalse-400-run9
do
    CUDA_VISIBLE_DEVICES=$devices python merge.py \
        --base_model_name_or_path meta-llama/Llama-2-13b-chat-hf \
        --peft_model_path ./ckpt/lora_ckpt/$model \
        --output_dir ./ckpt/hf_ckpt/$model
done