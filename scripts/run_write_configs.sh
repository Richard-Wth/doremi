#!/bin/bash

source /home/wth/My_codes/doremi/scripts/slimpajama_run/runs/slim_constants.sh

python scripts/write_config.py \
    --config_name slim_baseline_50kvocab_nopack_bucket \
    --preprocessed_dir ${PREPROCESSED_RP_DIR} \
    --cache_dir ${CACHE} \
    --nopack True \
    --tokenizer /home/wth/My_codes/doremi/tokenizer
# python scripts/write_config.py --config_name pile_uniform
# python scripts/write_config.py --config_name doremi_280M_256kvocab

