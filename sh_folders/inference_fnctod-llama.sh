export TRANSFORMERS_CACHE='HOME_PATH/.cache/huggingface/transformers'
export HF_HOME='HOME_PATH/.cache/huggingface'

devices=3

cd ..

# main DST results on multiwoz 2.1
for dataset_version in 2.1
do
    for split in test
    do
        for n_eval in 1000
        do
            for multi_domain in False
            do
                for ref_domain in False
                do
                    for ref_bs in False
                    do
                        for add_prev in True
                        do
                            for task in dst
                            do
                                for dst_nshot in 0
                                do
                                    for nlg_nshot in 0
                                    do
                                        for function_type in json # text
                                        do
                                            for model in fnctod-llama2-13b-200
                                            do
                                                CUDA_VISIBLE_DEVICES=$devices python -m src.multiwoz.inference \
                                                                                        --dataset_version $dataset_version \
                                                                                        --target_domains $target_domains \
                                                                                        --split $split \
                                                                                        --n_eval $n_eval \
                                                                                        --model $model \
                                                                                        --task $task \
                                                                                        --dst_nshot $dst_nshot \
                                                                                        --nlg_nshot $nlg_nshot \
                                                                                        --add_prev $add_prev \
                                                                                        --ref_domain $ref_domain \
                                                                                        --ref_bs $ref_bs \
                                                                                        --multi_domain $multi_domain \
                                                                                        --function_type $function_type \
                                                                                        --generate \
                                                                                        --verbose \
                                                                                        # --debug
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# e2e results with generated bs on multiwoz 2.2
for dataset_version in 2.2
do
    for split in test
    do
        for n_eval in 1000
        do
            for multi_domain in False
            do
                for ref_domain in False
                do
                    for ref_bs in False
                    do
                        for add_prev in True
                        do
                            for task in e2e # dst nlg
                            do
                                for dst_nshot in 0
                                do
                                    for nlg_nshot in 5
                                    do
                                        for function_type in json
                                        do
                                            for model in fnctod-llama2-13b-200
                                            do
                                                CUDA_VISIBLE_DEVICES=$devices python -m src.multiwoz.inference \
                                                                                        --dataset_version $dataset_version \
                                                                                        --target_domains $target_domains \
                                                                                        --split $split \
                                                                                        --n_eval $n_eval \
                                                                                        --model $model \
                                                                                        --task $task \
                                                                                        --dst_nshot $dst_nshot \
                                                                                        --nlg_nshot $nlg_nshot \
                                                                                        --add_prev $add_prev \
                                                                                        --ref_domain $ref_domain \
                                                                                        --ref_bs $ref_bs \
                                                                                        --multi_domain $multi_domain \
                                                                                        --function_type $function_type \
                                                                                        --generate \
                                                                                        --verbose \
                                                                                        # --debug
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done