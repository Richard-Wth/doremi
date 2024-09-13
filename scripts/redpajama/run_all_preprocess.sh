#!/bin/bash

# load global parameters
source constants.sh

set -x

# this can be parallelized for faster execution
NUM_SUBSETS=1
for DOMAIN in 'RedPajamaStackExchange_0' 'RedPajamaStackExchange_1' 'RedPajamaStackExchange_2' 'RedPajamaStackExchange_3' 'RedPajamaStackExchange_4' 'RedPajamaStackExchange_5' 'RedPajamaStackExchange_6' 'RedPajamaStackExchange_7' 'RedPajamaStackExchange_8' 'RedPajamaStackExchange_9' 'RedPajamaArXiv_0' 'RedPajamaArXiv_1' 'RedPajamaArXiv_2' 'RedPajamaArXiv_3' 'RedPajamaArXiv_4' 'RedPajamaArXiv_5' 'RedPajamaArXiv_6' 'RedPajamaArXiv_7' 'RedPajamaArXiv_8' 'RedPajamaArXiv_9' 'RedPajamaCommonCrawl_0' 'RedPajamaCommonCrawl_1' 'RedPajamaCommonCrawl_2' 'RedPajamaCommonCrawl_3' 'RedPajamaCommonCrawl_4' 'RedPajamaCommonCrawl_5' 'RedPajamaCommonCrawl_6' 'RedPajamaCommonCrawl_7' 'RedPajamaCommonCrawl_8' 'RedPajamaCommonCrawl_9' 'RedPajamaC4_0' 'RedPajamaC4_1' 'RedPajamaC4_2' 'RedPajamaC4_3' 'RedPajamaC4_4' 'RedPajamaC4_5' 'RedPajamaC4_6' 'RedPajamaC4_7' 'RedPajamaC4_8' 'RedPajamaC4_9' 'RedPajamaBook_0' 'RedPajamaBook_1' 'RedPajamaBook_2' 'RedPajamaBook_3' 'RedPajamaBook_4' 'RedPajamaBook_5' 'RedPajamaBook_6' 'RedPajamaBook_7' 'RedPajamaBook_8' 'RedPajamaBook_9' 'RedPajamaGithub_0' 'RedPajamaGithub_1' 'RedPajamaGithub_2' 'RedPajamaGithub_3' 'RedPajamaGithub_4' 'RedPajamaGithub_5' 'RedPajamaGithub_6' 'RedPajamaGithub_7' 'RedPajamaGithub_8' 'RedPajamaGithub_9' 'RedPajamaWikipedia_0' 'RedPajamaWikipedia_1' 'RedPajamaWikipedia_2' 'RedPajamaWikipedia_3' 'RedPajamaWikipedia_4' 'RedPajamaWikipedia_5' 'RedPajamaWikipedia_6' 'RedPajamaWikipedia_7' 'RedPajamaWikipedia_8' 'RedPajamaWikipedia_9';
do
for ((SUBSET=0; SUBSET<${NUM_SUBSETS}; SUBSET++));
do
bash /home/wth/My_codes/doremi/scripts/redpajama/run_preprocess.sh ${DOMAIN} ${SUBSET} ${NUM_SUBSETS}
done
done
