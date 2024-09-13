# Sample commands to run Pile data preprocessing

# load global parameters
source constants.sh

INTERMEDIATE_SCRATCH_PATH=${CACHE}/pile_preprocessed_tmp
TOKENIZER=/home/wth/My_codes/doremi/Redpajama

SPLIT=validation
for PILE_DOMAIN in "ArXiv" "BookCorpus2" "Books3" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "Gutenberg_(PG-19)" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Ubuntu_IRC" "Wikipedia_(en)" "YoutubeSubtitles"; do
LOGDIR=logs/preprocess_pile/${SPLIT}
mkdir -p ${LOGDIR}

source constants.sh
mkdir -p $CACHE 
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE

python /home/wth/My_codes/doremi/scripts/preprocess_pile.py \
        --pile_path_dir ${PILE_DIR} \
        --output_dir ${PREPROCESSED_PILE_DIR} \
        --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} \
        --cache_dir ${CACHE} \
        --split ${SPLIT} \
        --domain "${PILE_DOMAIN}" \
        --tokenizer ${TOKENIZER} \
        --seed 111 \
        --nproc 1 > ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} 2>&1
echo "Processing ${PILE_DOMAIN}"
done
done

# SPLIT=validation
# for PILE_DOMAIN in "ArXiv" "BookCorpus2" "Books3" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "Gutenberg_(PG-19)" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Ubuntu_IRC" "Wikipedia_(en)" "YoutubeSubtitles"; do
# LOGDIR=logs/preprocess_pile/${SPLIT}
# mkdir -p ${LOGDIR}
# jid=$(sbatch \
#         --parsable \
#         --partition ${PARTITION} \
#         --mem 64G \
#         -c 1 \
#         --output ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} \
#         /home/wth/My_codes/doremi/scripts/run.sh "python /home/wth/My_codes/doremi/scripts/preprocess_pile.py --pile_path_dir ${PILE_DIR} --output_dir ${PREPROCESSED_PILE_DIR} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain \"${PILE_DOMAIN}\" --tokenizer ${TOKENIZER} --seed 111 --nproc 1 --subset ${SUBSET}")
# echo -n "${jid} "
# done