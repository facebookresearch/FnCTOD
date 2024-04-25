# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

python -m spacy download en_core_web_sm
cd ./ubar-preprocessing/data
wget https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip?raw=true -O MultiWOZ_2.1.zip
unzip MultiWOZ_2.1.zip
cd ..
python data_analysis21.py
python preprocess21.py
cd ..
rm -rf ./ubar-preprocessing/data/MultiWOZ_2.1
rm -rf ./ubar-preprocessing/data/MultiWOZ_2.1.zip
rm -rf ./ubar-preprocessing/data/__MACOSX
cp -r ./ubar-preprocessing/data ./
cp -r ./ubar-preprocessing/db ./data/
cd ./utlis
python postprocessing_dataset21.py
python convert_to_rawdata.py --data_version '2.1' --save_data_path_prefix '../data/multi-woz-2.1-rawdata'
cd ..
cp special_token_list.txt ./data/multi-woz-2.1-fine-processed/special_token_list.txt
cp schema.json ./data/multi-woz-2.1-fine-processed/schema.json
cp possible_slot_values.json ./data/multi-woz-2.1-fine-processed/possible_slot_values.json