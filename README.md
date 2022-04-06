# Are Shortest Rationales the Best Explanations for Human Understanding?


This repository aims to investigate <em><strong>if the shortest rationales are best understandable for humans</em></strong>, which includes the codes of:

1. <strong>LimitedInk Model</strong>: A self-explaining model that can control rationale length.
2. <strong>Human Study</strong>: the user interfaces to implement MTurk experiments.


The work is to support the paper 
["Are Shortest Rationales the Best Explanations For Human Understanding?"](https://hua-shen.org/assets/files/ACL2022_LimitedInk.pdf)
 by Hua Shen, Tongshuang Wu, Wenbo Guo, Ting-Hao 'Kenneth' Huang. The paper is accepted by the [ACL 2022 main conference](https://www.2022.aclweb.org/).


## 1. LimitedInk: A Self-Explaining Model with Control on Rationale Length  

This part contains the detailed implementation of the LimitedInk model, which allows users to extract rationales at any target length.

## Install Dependencies

You need to install the packages in `requirements.txt` before running the code:
```
pip install -r requirements.txt
```


## Download Data
We cover five datasets from the ERASER benchmark, including *Movies*, *BoolQ*, *Evidence Inference*, *MultiRC*, *FEVER*.

Datasets can be downloaded and automatically unpacked to the `/data` directory by:
```
bash download_data.sh
```


## Run LimitedInk 

To train the LimitedInk, find the running bash files in `/limitedink/bash` by: 
```
$ cd limitedink/bash
```


Then run model by the corresponding `.sh` file. For example, to train LimitedInk (with token-level rationales) on movies dataset, run:
```
$ bash run_movies_token.sh
```


## LimitedInk Model Details


### A. Clarification of running model bash script 

In the running bash script, please set `REPO_PATH` as the absolute path to this repo (e.g., `REPO_PATH="/workspace/projects/LimitedInk`").
The important arguments in bash script contains:

- `--length`: set the rationale length as the percentage of overall input length (e.g., k=0.5).
- `--seed`: set the random seed for model running (e.g., rand="1234").
- `--data_dir`: set the directory of downloaded dataset. (e.g., "$REPO_PATH/data/movies")
- `--save_dir`: set the directory for saving trained model checkpoints.
- `--configs`: define the hyperparameters in the LimitedInk model.
- `LOG_DIR`: set the directory to save the training script log.
- `CUDA_VISIBLE_DEVICES=0`: set the index of your GPU.


Below shows `run_movies_main_token.sh` -- an example of running token-level LimitedInk on *Movies* dataset:
```
REPO_PATH="your_absolute_path_to_this_repo"
export PYTHONPATH=$REPO_PATH

k=0.5;
rand="1234";
SAVE_DIR="$REPO_PATH/checkpoints/movies/distilbert/token_rationale/length_level_$k/seed_$rand";
mkdir -p $SAVE_DIR
LOG_DIR="$SAVE_DIR/train.log";

CUDA_VISIBLE_DEVICES=0 python ../main.py --data_dir "$REPO_PATH/data/movies" --save_dir $SAVE_DIR --configs "$REPO_PATH/limitedink/params/movies_config_token.json" --length $k --seed $rand > $LOG_DIR


```


### B. Clarification of model hyperparameter config

You can set the detailed model hyperparameters in the config `.json` files at `/limitedink/params` by:
```
$ cd limitedink/params
```

Some important model configurations include:
- `batch_size`: batch size for training on one GPU.
- `tau`: set the temperature of the gumbel softmax.
- `epochs`: set the training epoch number.
- `lr`: set the learning rate of model training.
- `continuity_lambda`: set the continuity regularization term in the loss function. 
- `sparsity_lambda`: set the coefficiency of length control regularization term in the loss function. 
- `comprehensive_lambda` (optional): set the coefficiency of comprehensive prediction loss. 


Below shows `movies_config_token.json` -- an example of model config on token-level LimitedInk model:
```
{
    "data_params": {
        "task": "movies",
        "model": "distilbert",
        "batch_size": 3,
        "max_seq_length": 512,
        "max_query_length": 9,
        "max_num_sentences": 36,
        "classes": [ "NEG", "POS"],
        "labels_ids": {"NEG": 0, "POS": 1},
        "truncate": false,
        "partial_train": 1.0,
        "rationale_level": "token",
        "overwrite_cache": false,
        "cached_features_file": "token_cached_features_file.pt",
        "remove_query_input": false
    },
    "model_params": {
        "tau": 0.1,
        "num_labels": 2,
        "model_type": "distilbert-base-uncased",
        "dropout": 0.5,
        "loss_function": "limitedink"
    },
    "train_params": {
        "epochs": 6,
        "lr": 2e-5
    },
    "model_kwargs": {
        "continuity_lambda": 0.5, 
        "sparsity_lambda": 0.3,
        "comprehensive_lambda": 0.0001
    }
}
```



## 2. Human Evaluation

The human study contains two stages to strictly control worker participation.
We provide the user interfaces of both stages:

> **Stage1: Participants recruiting.**

The *qualification* HIT (Human Intelligence Task) interface only contains one review, see:
`human_evaluation/user_interface/recruit_participant/human_eval_recruiting.html`

> **Stage2: Human Evaluation.** 

There are 20 batches of *task* HITs in this human study. We provide all 200 interfaces, for example, see:
`human_evaluation/user_interface/human_study/human_eval_batch_0_workergroup0.html`



## Citation
If you find this repo helpful to your research, please cite the paper:
```
@article{shen2022shortest,
  title={Are Shortest Rationales the Best Explanations for Human Understanding?},
  author={Shen, Hua and Wu, Tongshuang and Guo, Wenbo and Huang, Ting-Hao'Kenneth'},
  journal={arXiv preprint arXiv:2203.08788},
  year={2022}
}
```