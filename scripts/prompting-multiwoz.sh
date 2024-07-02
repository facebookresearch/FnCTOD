# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# multiwoz
cd ..

for dataset_version in 2.1
do
    for split in train
    do
        for n_sample in 80 400 800
        do
            for template in vicuna # llama2
            do  
                for all_turn in False True
                do 
                    CUDA_VISIBLE_DEVICES=$devices python -m src.multiwoz.prompting \
                                                            --dataset_version $dataset_version \
                                                            --split $split \
                                                            --n_sample $n_sample \
                                                            --template $template \
                                                            --all_turn $all_turn
                done
            done
        done
    done
done
