#!/bin/bash
CACHE=/home/wth/My_codes/doremi/cache
DOREMI_DIR=/home/wth/My_codes/doremi
SLIM_DIR=/home/wth/My_codes/doremi/data/slimpajama/split
PREPROCESSED_SLIM_DIR=/home/wth/My_codes/doremi/data/slimpajama/bucket  # will be created by scripts/run_preprocess_pile.sh
MODEL_OUTPUT_DIR=/home/wth/My_codes/doremi/output
WANDB_API_KEY=7c5c21e2ade70f6c6f192a30c14a696804a1c9f7  # Weights and Biases key for logging
PARTITION=partition # for slurm
mkdir -p ${CACHE}
mkdir -p ${MODEL_OUTPUT_DIR}
# source ${DOREMI_DIR}/venv/bin/activate  # if you installed doremi in venv