# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

cd ..

for pd_size in 100 200 300 400
do
    python create_finetunedata.py --configfile ./data/finetunedata/sft-llama2.yml \
                        --outputfile ./data/finetunedata/sft-llama2-pd$pd_size.json \
                        --domain_size $pd_size \
                        --max_len 4096
done