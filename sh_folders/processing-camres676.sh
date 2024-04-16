cd ..

# remember to manually clean the 'CamRestDB.json' data (remove the comments at the beginning)
python -m src.camres676.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir CamRes676
cd ../../..
mv ./src/camres676/normalized_schema.yml ./data/pre-training_corpora/processed_data/CamRes676/
python -m src.camres676.postprocess