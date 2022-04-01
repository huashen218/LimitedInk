
# LimitedInk

This project is about investigating <em>if the shortest rationales are best understandable for humans</em>. This repository includes the codes of:

- <strong>LimitedInk</strong>: A self-explaining model that can control rationale length.
- <strong>Human Study</strong>: the user interfaces to implement MTurk experiments.


The work is to support the paper 
["Are Shortest Rationales the Best Explanations For Human Understanding?"](https://hua-shen.org/assets/files/ACL2022_LimitedInk.pdf)
, which is accepted by the [ACL 2022 main conference](https://www.2022.aclweb.org/).



## Installation

You need to install the packages in `requirements.txt` before running the code:
```
pip install -r requirements.txt
```


## Download Data

Datasets can be downloaded and unpacked to the `data` directory by:
```
bash download_data.sh
```


## LimitedInk Model

To train the LimitedInk, find the running bash files in `limitedink/bash`. Then run model by the corresponding `.sh` file. For example, to train LimitedInk (with token-level rationales) on movies dataset, run:
```
$ cd limitedink/bash
$ bash run_movies_token.sh
```


## Human Evaluation

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