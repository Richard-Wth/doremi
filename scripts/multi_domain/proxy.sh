# the reference weights in the DoReMi paper were computed by counting the number of chunks in each domain, without packing and with padding, which slightly changes the weights. We can also compare to pile_baseline_50kvocab which includes packing and no padding, with similar results. All the trained models here use packing during training, regardless of how the reference weights were computed.
REFERENCE_WEIGHTS_NAME=md_baseline_50kvocab_nopack 

ROUND=1
REFERENCE_MODEL_NAME=${REFERENCE_WEIGHTS_NAME}_120M
bash scripts/multi_domain/runs/run_doremi120M.sh ${ROUND} ${REFERENCE_WEIGHTS_NAME} ${REFERENCE_MODEL_NAME}