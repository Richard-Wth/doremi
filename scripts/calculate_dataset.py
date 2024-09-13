from pathlib import Path
import argparse
from datasets import load_from_disk
from collections import defaultdict
import shutil
import math
from tqdm import tqdm
from joblib import Parallel, delayed
from transformers import AutoTokenizer

PILE_DOMAINS = ['ArXiv', 'BookCorpus2', 'Books3', 'DM Mathematics', 'Enron Emails', 'EuroParl', 'FreeLaw', 'Github', 'Gutenberg (PG-19)', 'HackerNews', 'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers', 'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange', 'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles']
SLIM_DOMAINS = ['RedPajamaCommonCrawl', 'RedPajamaC4', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaBook', 'RedPajamaArXiv', 'RedPajamaStackExchange']

def compute_data(shard_dir, nopack=True, tokenizer=None):
    curr_count = 0
    ds = load_from_disk(dataset_path=str(shard_dir))
    if nopack:
        # in the DoReMi paper, we first padded to the context length then counted
        # the number of chunks, and dynamically packed the examples
        # together (possibly even from different domains)
        num_tokens_in_curr_doc = 0
        chunk_size = 1024
        for ex in tqdm(ds):
            toks = ex['input_ids']
            sep_idxs = [i for i in range(len(toks)) if toks[i] == tokenizer.eos_token_id]
            if len(sep_idxs) > 0:
                prev_sep_idx = -1
                for sep_idx in sep_idxs:
                    num_tokens_in_curr_doc += sep_idx - prev_sep_idx - 1
                    prev_sep_idx = sep_idx
                    curr_count += math.ceil(num_tokens_in_curr_doc / chunk_size)
                    num_tokens_in_curr_doc = 0
                if prev_sep_idx != len(toks) - 1:
                    num_tokens_in_curr_doc += len(toks) - prev_sep_idx - 1
            else:
                num_tokens_in_curr_doc += len(toks)
        if num_tokens_in_curr_doc > 0:
            curr_count += math.ceil(num_tokens_in_curr_doc / chunk_size)
    else:
        curr_count = len(ds)

    return curr_count

def count_all_domains(domain_lens):
    nums = 0
    for domain in domain_lens.keys():
        nums += domain_lens[domain]
    return nums

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dir", type=str, default="/path/to/data")
    parser.add_argument("--nopack", type=bool, default=True)
    parser.add_argument("--tokenizer", type=str, default='togethercomputer/RedPajama-INCITE-Base-7B-v0.1')
    args = parser.parse_args()

    domain_lens = defaultdict(int)
    preprocessed_dir = Path(args.preprocessed_dir) / 'train'
    for domain_dir in preprocessed_dir.iterdir():
        print("Counting domain", domain_dir.name)
        counts = Parallel(n_jobs=30)(delayed(compute_data)(shard_dir) for shard_dir in domain_dir.iterdir())
        domain_lens[domain_dir.name] = sum(counts)

    nums = count_all_domains(domain_lens)
    print(nums)

if __name__ == "__main__":
    main()

    