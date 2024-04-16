cd ..

python -m src.mse2e.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir MS_E2E
cd ../../..
mv ./src/mse2e/normalized_schema.yml ./data/pre-training_corpora/processed_data/MS_E2E/
python -m src.mse2e.postprocess