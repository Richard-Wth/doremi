# Sample commands to run Pile data preprocessing

# load global parameters
source /home/wth/My_codes/doremi/Constants/random_constants.sh

INTERMEDIATE_SCRATCH_PATH=${CACHE}/pile_preprocessed_random_tmp
TOKENIZER=/home/wth/My_codes/doremi/tokenizer

SPLIT=train
for PILE_DOMAIN in "BookCorpus2" "Books3" "Gutenberg_(PG-19)" "Ubuntu_IRC" ; do
for SUBSET in 00 01 02; do
LOGDIR=logs/preprocess_pile_random/${SPLIT}
mkdir -p ${LOGDIR}

mkdir -p $CACHE 
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE

echo "Processing ${PILE_DOMAIN}_${SUBSET}"
python /home/wth/My_codes/doremi/scripts/pile_process/preprocess_pile_rdm.py \
        --pile_path_dir ${PILE_DIR} \
        --output_dir ${PREPROCESSED_PILE_DIR} \
        --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} \
        --cache_dir ${CACHE} \
        --split ${SPLIT} \
        --domain "${PILE_DOMAIN}" \
        --tokenizer ${TOKENIZER} \
        --seed 111 \
        --nproc 2 \
        --subset ${SUBSET} > ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} 2>&1
done
done