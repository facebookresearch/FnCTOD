# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

cd ..

python -m src.frames.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir Frames
cd ../../..
mv ./src/frames/normalized_schema.yml ./data/pre-training_corpora/processed_data/Frames/
python -m src.frames.postprocess