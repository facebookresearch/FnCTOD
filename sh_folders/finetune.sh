export TRANSFORMERS_CACHE='HOME_PATH/.cache/huggingface/transformers'
export HF_HOME='HOME_PATH/.cache/huggingface'

devices=1

cd ..

for data in sft-llama2-200
do
    for train_on_response in False
    do
        for lr in 0.0003
        do
            for ep in 1
            do
                CUDA_VISIBLE_DEVICES=$devices python finetune.py \
                    --base_model meta-llama/Llama-2-13b-chat-hf \
                    --data-path ./data/finetunedata/$data.json \
                    --output_dir ./ckpt/lora_ckpt/llama-2-13b-chat-$data \
                    --wandb_project llama-2-13b-chat-$data \
                    --wandb_watch all \
                    --batch_size 2 \
                    --micro_batch_size 1 \
                    --num_epochs $ep \
                    --learning_rate $lr \
                    --cutoff_len 4096 \
                    --val_set_size 0 \
                    --lora_r 16 \
                    --lora_alpha 16 \
                    --lora_dropout 0.05 \
                    --lora_target_modules '[q_proj, v_proj]' \
                    --train_on_response $train_on_response \
                    --add_eos_token False \
                    --group_by_length False \
                    --lr_scheduler 'cosine' \
                    --warmup_steps 100
            done
        done
    done
done