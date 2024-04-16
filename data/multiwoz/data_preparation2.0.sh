python -m spacy download en_core_web_sm
cd ./ubar-preprocessing/data
cd multi-woz
unzip data.json.zip
cd ..
cd ..
python data_analysis.py
python preprocess.py 
cd ..
cp -r ./ubar-preprocessing/data ./
cd ./utlis
python postprocessing_dataset.py
python convert_to_rawdata.py --data_version '2.0' --save_data_path_prefix '../data/multi-woz-2.0-rawdata'
cd ..
cp special_token_list.txt ./data/multi-woz-fine-processed/special_token_list.txt
cp schema.json ./data/multi-woz-fine-processed/schema.json
cp possible_slot_values.json ./data/multi-woz-fine-processed/possible_slot_values.json
