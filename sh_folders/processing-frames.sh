cd ..

python -m src.frames.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir Frames
cd ../../..
mv ./src/frames/normalized_schema.yml ./data/pre-training_corpora/processed_data/Frames/
python -m src.frames.postprocess