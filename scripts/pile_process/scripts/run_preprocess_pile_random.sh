# Sample commands to run Pile data preprocessing with random choice

# load global parameters
source random_constants.sh

INTERMEDIATE_SCRATCH_PATH=${CACHE}/pile_preprocessed_random_tmp
TOKENIZER=/home/wth/My_codes/doremi/Redpajama

SPLIT=train
for PILE_DOMAIN in "ArXiv" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Wikipedia_(en)" "YoutubeSubtitles"; do
for SUBDOMAIN in "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43" "44" "45" "46" "47" "48" "49" "50" "51" "52" "53" "54" "55" "56" "57" "58" "59" "60" "61" "62" "63" "64" "65" "66" "67" "68" "69" "70" "71" "72" "73" "74" "75" "76" "77" "78" "79" "80" "81" "82" "83" "84" "85" "86" "87" "88" "89" "90" "91" "92" "93" "94" "95" "96" "97" "98" "99" "100"; do
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

echo "Processing ${PILE_DOMAIN}_${SUBDOMAIN}_${SUBSET}"
python /home/wth/My_codes/doremi/scripts/preprocess_pile_random.py \
        --pile_path_dir ${PILE_DIR} \
        --output_dir ${PREPROCESSED_PILE_DIR} \
        --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} \
        --cache_dir ${CACHE} \
        --split ${SPLIT} \
        --domain "${PILE_DOMAIN}" \
        --subdomain "${SUBDOMAIN}" \
        --tokenizer ${TOKENIZER} \
        --seed 111 \
        --nproc 1 \
        --subset ${SUBSET} > ${LOGDIR}/${PILE_DOMAIN}_${SUBDOMAIN}_${SUBSET} 2>&1
done
done
done

# SPLIT=train
# for PILE_DOMAIN in "ArXiv" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Wikipedia_(en)" "YoutubeSubtitles"; do
# for SUBSET in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
# LOGDIR=logs/preprocess_pile/${SPLIT}
# mkdir -p ${LOGDIR}
# jid=$(sbatch \
#         --parsable \
#         --partition ${PARTITION} \
#         --mem 8G \
#         -c 1 \
#         --output ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} \
#         /home/wth/My_codes/doremi/scripts/run.sh "python /home/wth/My_codes/doremi/scripts/preprocess_pile.py --pile_path_dir ${PILE_DIR} --output_dir ${PREPROCESSED_PILE_DIR} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain \"${PILE_DOMAIN}\" --tokenizer ${TOKENIZER} --seed 111 --nproc 1 --subset ${SUBSET}")
# echo -n "${jid} "
# done
# done


# SPLIT=train
# for PILE_DOMAIN in "BookCorpus2" "Books3" "Gutenberg_(PG-19)" "Ubuntu_IRC" ; do
# for SUBSET in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
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
# done


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
