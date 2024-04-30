We thank the authors of [PPTOD](https://github.com/awslabs/pptod) for gathering up the publicly available human-written multi-turn dialogue corpora.

# Preparation of Benchmark TOD Task Datasets:
## MultiWOZ Data:
The MultiWOZ dataset is used for both end-to-end task-oriented dialogue modelling and dialogue state tracking tasks.
### (1) Preparation:
To acquire the processed dataset, you can run the following commands. 
```yaml
cd ./multiwoz/
chmod +x ./download_data.sh 
chmod +x ./data_preparation2.1.sh 
chmod +x ./data_preparation2.2.sh 

sh ./download_data.sh # download the necessary data
sh ./data_preparation2.1.sh # prepare the data for multiwoz 2.1
sh ./data_preparation2.2.sh # prepare the data for multiwoz 2.2
```
Take a coffee, this process will take around 60 minutes.

### (2) Data Format:
```json
[
    {
        "dial_id": "PMUL1170",
        "user": "i need to take a train out of cambridge , i will be leaving town on wednesday .",
        "resp": "there are [value_choice] trains out of [value_departure] on [value_day] . do you have a departure time in mind ?",
        "bspn": "[train] day wednesday departure cambridge",
        "aspn": "[train] [inform] choice departure day [request] leave",
        "turn_num": 0,
        "db": "[db_3]",
    },
    {
        "dial_id": "PMUL1170",
        "user": " i would like to go to peterborough and leave after 12:45 , i have to attend a meeting beforehand .",
        "resp": "[value_id] leaves at [value_leave] on [value_day] . will that work for you ?",
        "bspn": "[train] day wednesday departure cambridge leave 12:45 destination peterborough",
        "aspn": "[train] [inform] day leave id",
        "turn_num": 1,
        "db": "[db_3]",
    },
    ...
]
```
We use json to store the data. Each dialogue session is represented as a list of turns. Each turn is represented as a dictionary that contains the following fields:

* **dial_id** - The unique ID for the dialogue session instance. 
* **user** - The user's utterance.
* **resp** - The delexicalized reference system response.
* **bspn** - The belief state.
* **aspn** - The system action.
* **turn_num** - This argument indicates the turn position in the dialogue session, e.g., if turn_num = 0 means this is the very first turn in the whole dialogue session.
* **db** - The database query result.


# Pre-training Corpora Preparation:
We download the raw data of several publicly available human-written multi-turn dialogue corpora collected by [ToD-BERT](https://github.com/jasonwu0731/ToD-BERT). To run the pre-training data preparation scripts, please first install gdown library as:
```yaml
pip3 install gdown
```

Then, run the following commands to download the raw data.
```yaml
cd pre-training_corpora
sh download_raw_data.sh
```

For the following processing of the raw data, please refer to [Prompt-based Fine-tuning](https://github.com/facebookresearch/FnCTOD/tree/main?tab=readme-ov-file#prompt-based-fine-tuning) seciton in the main README.


