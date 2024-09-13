REFERENCE_WEIGHTS_NAME=pile_baseline_50kvocab_nopack 

# the reference weights in the DoReMi paper were computed by counting the number of chunks in each domain, 
# without packing and with padding, 
# which slightly changes the weights. 
# We can also compare to pile_baseline_50kvocab which includes packing and no padding, with similar results. 
# All the trained models here use packing during training, regardless of how the reference weights were computed.
bash scripts/runs/run_pile_baseline120M.sh ${REFERENCE_WEIGHTS_NAME}
