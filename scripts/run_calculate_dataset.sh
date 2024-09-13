#!/bin/bash

source constants.sh

python scripts/calculate_dataset.py \
    --preprocessed_dir ${PREPROCESSED_PILE_DIR} \
    --nopack True \
    --tokenizer /home/wth/My_codes/doremi/tokenizer

