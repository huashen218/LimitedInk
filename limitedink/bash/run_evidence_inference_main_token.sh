#!/usr/bin/env bash
set -x;
set -e;
REPO_PATH="/home/hqs5468/hua/workspace/projects/LimitedInk"
export PYTHONPATH=$REPO_PATH

k=0.5;
rand="1234";
SAVE_DIR="$REPO_PATH/checkpoints/evidence_inference/distilbert/token_rationale/length_level_$k/seed_$rand";
mkdir -p $SAVE_DIR
LOG_DIR="$SAVE_DIR/train.log";

CUDA_VISIBLE_DEVICES=0 python ../main.py --data_dir "$REPO_PATH/data/evidence_inference" --save_dir $SAVE_DIR --configs "$REPO_PATH/limitedink/params/evidence_inference_config_token.json" --length $k --seed $rand > $LOG_DIR

