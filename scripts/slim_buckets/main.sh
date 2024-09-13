#!/bin/bash
#
# This sample script runs 1 round of DoReMi using a 120M proxy model on the Pile, mostly following the DoReMi paper. Run using `bash scripts/run_pile.sh`.
# the reference weights in the DoReMi paper were computed by counting the number of chunks in each domain, without packing and with padding, which slightly changes the weights. We can also compare to pile_baseline_50kvocab which includes packing and no padding, with similar results. All the trained models here use packing during training, regardless of how the reference weights were computed.
REFERENCE_WEIGHTS_NAME=slim_baseline_50kvocab_nopack_bucket  
ROUND=1
REFERENCE_MODEL_NAME=${REFERENCE_WEIGHTS_NAME}_120M
bash scripts/slim_buckets/runs/run_main120M.sh ${ROUND} ${REFERENCE_WEIGHTS_NAME}