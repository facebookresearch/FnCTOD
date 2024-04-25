# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

cd ..

python -m src.mse2e.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir MS_E2E
cd ../../..
mv ./src/mse2e/normalized_schema.yml ./data/pre-training_corpora/processed_data/MS_E2E/
python -m src.mse2e.postprocess