cd ..

python -m src.taskmaster.preprocess
cd ./data/pre-training_corpora/processed_data
mkdir TaskMaster
cd ../../..
mv ./src/taskmaster/normalized_schema.yml ./data/pre-training_corpora/processed_data/TaskMaster/
python -m src.taskmaster.postprocess