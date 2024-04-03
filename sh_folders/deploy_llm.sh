
export TRANSFORMERS_CACHE='XXX/.cache/huggingface/transformers'
export HF_HOME='XXX/.cache/huggingface'

devices=0

cd ..

# gpt-3.5 gpt-4 vicuna-7b-v1.5 vicuna-13b-v1.5 llama-2-7b-chat llama-2-13b-chat llama-2-70b-chat alpaca-7b
CUDA_VISIBLE_DEVICES=$devices python -m LLM.app --model vicuna-7b-v1.3