#!/bin/bash
CACHE=/home/wth/My_codes/doremi/cache
DOREMI_DIR=/home/wth/My_codes/doremi
MD_DIR=/home/wth/My_codes/doremi/data/multi_domain
PREPROCESSED_MD_DIR=/home/wth/My_codes/doremi/data/md
MODEL_OUTPUT_DIR=/home/wth/My_codes/doremi/output/md
WANDB_API_KEY=7c5c21e2ade70f6c6f192a30c14a696804a1c9f7  # Weights and Biases key for logging
PARTITION=partition # for slurm
mkdir -p ${CACHE}
mkdir -p ${MODEL_OUTPUT_DIR}
# source ${DOREMI_DIR}/venv/bin/activate  # if you installed doremi in venv