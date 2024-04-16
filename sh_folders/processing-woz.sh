cd ..

python -m src.woz.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir WOZ
cd ../../..
mv ./src/woz/normalized_schema.yml ./data/pre-training_corpora/processed_data/WOZ/
python -m src.woz.postprocess