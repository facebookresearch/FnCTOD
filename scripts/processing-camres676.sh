# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

cd ..

# remember to manually clean the 'CamRestDB.json' data (remove the comments at the beginning)
python -m src.camres676.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir CamRes676
cd ../../..
mv ./src/camres676/normalized_schema.yml ./data/pre-training_corpora/processed_data/CamRes676/
python -m src.camres676.postprocess