cd ..

python -m src.kvret.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir KVRET
cd ../../..
mv ./src/kvret/normalized_schema.yml ./data/pre-training_corpora/processed_data/KVRET/
python -m src.kvret.postprocess