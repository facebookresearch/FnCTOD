# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

export OPENAI_API_KEY='XXXX'

devices=0

cd ..

# main DST results on multiwoz2.1
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
                        for add_prev in False
                        do
                            for task in dst
                            do
                                for dst_nshot in 0
                                do
                                    for nlg_nshot in 0
                                    do
                                        for function_type in json
                                        do
                                            for model in gpt-3.5-0125
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
