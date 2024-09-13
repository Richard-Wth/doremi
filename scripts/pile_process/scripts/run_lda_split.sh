# load global parameters
source lda_constants.sh

INTERMEDIATE_SCRATCH_PATH=${CACHE}/lda_split_tmp

SPLIT=train
for PILE_DOMAIN in "ArXiv" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Wikipedia_(en)" "YoutubeSubtitles"; do
for SUBSET in 00 01 02; do
LOGDIR=logs/lda_split/${SPLIT}
mkdir -p ${LOGDIR}

source constants.sh
mkdir -p $CACHE 
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE

echo "Processing ${PILE_DOMAIN}_${SUBSET}"
python /home/wth/My_codes/doremi/scripts/lda_split.py \
        --pile_path_dir ${PILE_DIR} \
        --output_dir ${PREPROCESSED_PILE_DIR} \
        --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} \
        --cache_dir ${CACHE} \
        --split ${SPLIT} \
        --domain "${PILE_DOMAIN}" \
        --seed 111 \
        --subset ${SUBSET} > ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} 2>&1
done
done