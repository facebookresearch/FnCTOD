# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

cd ..

python -m src.taskmaster.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir TaskMaster
cd ../../..
mv ./src/taskmaster/normalized_schema.yml ./data/pre-training_corpora/processed_data/TaskMaster/
python -m src.taskmaster.postprocess