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
git clone https://github.com/budzianowski/multiwoz.git
cd multiwoz/data/MultiWOZ_2.2
python convert_to_multiwoz_format.py --multiwoz21_data_dir="../../../MultiWOZ_2.1" --output_file="moz22_data.json"
mv moz22_data.json ../../../MultiWOZ_2.1/
cd ../../../MultiWOZ_2.1/
rm data.json
mv moz22_data.json data.json
cd ..
mv MultiWOZ_2.1 MultiWOZ_2.2
rm -rf multiwoz
cd ..
python data_analysis22.py
python preprocess22.py
cd ..
rm -rf ./ubar-preprocessing/data/MultiWOZ_2.2
rm -rf ./ubar-preprocessing/data/MultiWOZ_2.1.zip
rm -rf ./ubar-preprocessing/data/__MACOSX
cp -r ./ubar-preprocessing/data ./
cp -r ./ubar-preprocessing/db ./data/
cd ./utlis
python postprocessing_dataset22.py
python convert_to_rawdata.py --data_version '2.2' --save_data_path_prefix '../data/multi-woz-2.2-rawdata'
cd ..
cp special_token_list.txt ./data/multi-woz-2.2-fine-processed/special_token_list.txt
cp schema.json ./data/multi-woz-2.2-fine-processed/schema.json
cp possible_slot_values.json ./data/multi-woz-2.2-fine-processed/possible_slot_values.json