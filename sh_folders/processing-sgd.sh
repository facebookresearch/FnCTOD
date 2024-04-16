cd ..

python -m src.sgd.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir Schema_Guided
cd ../../..
mv ./src/sgd/normalized_schema.yml ./data/pre-training_corpora/processed_data/Schema_Guided/
python -m src.sgd.postprocess