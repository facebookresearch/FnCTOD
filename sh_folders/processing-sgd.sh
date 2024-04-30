# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

cd ..

python -m src.sgd.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir Schema_Guided
cd ../../..
mv ./src/sgd/normalized_schema.yml ./data/pre-training_corpora/processed_data/Schema_Guided/
python -m src.sgd.postprocess