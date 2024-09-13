"""
Filter the Pile data into subdomains with LDA and tokenize the data.
"""

from gensim import corpora
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import numpy as np

from datasets import load_dataset, Dataset, IterableDataset, Features, Sequence, Value
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
from pathlib import Path
from itertools import cycle, chain
import torch
import os
import shutil
import random
from collections import defaultdict
from tokenizers.processors import TemplateProcessing

PILE_DOMAINS = ['ArXiv', 'BookCorpus2', 'Books3', 'DM Mathematics', 'Enron Emails', 'EuroParl', 'FreeLaw', 'Github', 'Gutenberg (PG-19)', 'HackerNews', 'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers', 'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange', 'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles']
PILE_SUBDOMAINS = ["sub1", "sub2", "sub3", "sub4", "sub5"]

DOMAIN_TO_IDX = {
    name: idx for idx, name in enumerate(PILE_DOMAINS)
}

SUBDOMAIN_TO_IDX = {
    name: idx for idx, name in enumerate(PILE_SUBDOMAINS)
}

PILE_SUBSETS = [f'0{i}' if i < 10 else str(i) for i in range(0, 30)]

def pile_transform(tokenizer, max_length, seed=None):
    def transform(batch):
        examples = tokenizer(batch['text'])

        # Concatenate all texts. attention mask is all 1
        examples = {k: list(chain(*examples[k])) for k in examples.keys() if k!= 'attention_mask'}
        total_length = len(examples[list(examples.keys())[0]])
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length

        # Split by chunks of max_len
        result = {
            k: [t[i: i+max_length] for i in range(0, total_length, max_length)]
            for k, t in examples.items()
        }
        return result
    return transform

def preprocess_text(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def lda_subdomain_division(ds, num_subdomains=5, num_passes=1):
    # Prepare corpus
    texts = [preprocess_text(example['text']) for example in ds]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train LDA model
    lda_model = LdaMulticore(corpus=corpus, 
                             id2word=dictionary, 
                             num_topics=num_subdomains, 
                             passes=num_passes,
                             workers=4)  # Adjust number of workers as needed

    def assign_subdomain(example):
        text = preprocess_text(example['text'])
        bow = dictionary.doc2bow(text)
        topic_distribution = lda_model.get_document_topics(bow)
        subdomain = max(topic_distribution, key=lambda x: x[1])[0]
        return subdomain

    return assign_subdomain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pile_path_dir", type=str, default="/path/to/pile")
    parser.add_argument("--output_dir", type=str, default="/path/to/output")
    parser.add_argument("--intermediate_dir", type=str, default="/path/to/intermediate")
    parser.add_argument("--domain", type=str, default="Book3")
    parser.add_argument("--subdomain", type=str, default="sub1")
    parser.add_argument("--subset", type=str, default="01")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default="gpt-2")
    parser.add_argument("--cache_dir", type=str, default="/path/to/cache")
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    args.domain = args.domain.replace("_", " ")

    # move from intermediate dir to output dir
    if args.split == "train":
        output_dir = Path(args.output_dir) / args.split / args.domain / args.subdomain / args.subset
    else:
        output_dir = Path(args.output_dir) / args.split / args.domain / args.subdomain
    if output_dir.exists():
        print("Already done, skipping")
        return
    
    pile_dir = Path(args.pile_path_dir)
    
    if args.split == 'train':
        train_pile_dir = Path(os.path.join(pile_dir, args.split))
        data_files = [str(train_pile_dir / f"{args.subset}.jsonl.zst")]
    elif args.split == 'validation':
        data_files = [str(pile_dir / "val.jsonl.zst")]
    else:
        data_files = [str(pile_dir / f"test.jsonl.zst")]

    
    # Load dataset
    ds = load_dataset('json',
                      data_files=data_files,
                      cache_dir=args.cache_dir,
                      streaming=True)['train']

    def filter_fn(ex, idx):
        return ex['meta']['pile_set_name'] == args.domain

    ds = ds.filter(filter_fn, with_indices=True)

    # Train LDA model and get subdomain assigner
    assign_subdomain = lda_subdomain_division(ds, num_subdomains=5)

    # Tokenize and assign subdomains
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="$A "+tokenizer.eos_token,
        special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)])
    transform = pile_transform(tokenizer, args.max_length, seed=args.seed)

    def process_and_assign(example):
        tokenized = transform({'text': example['text']})
        subdomain = assign_subdomain(example)
        tokenized['domain_id'] = DOMAIN_TO_IDX[args.domain]
        tokenized['subdomain_id'] = subdomain
        return tokenized

    processed_ds = ds.map(process_and_assign, remove_columns=['text', 'meta'])

    features = Features({
        "input_ids": Sequence(Value("int32")),
        "domain_id": Value("int32"),
        "subdomain_id": Value("int32"),
    })

    # Save dataset
    if args.split == "train":
        intermediate_dir = Path(args.intermediate_dir) / args.split / args.domain / args.subset
    else:
        intermediate_dir = Path(args.intermediate_dir) / args.split / args.domain

    intermediate_dir.parent.mkdir(parents=True, exist_ok=True)
    processed_ds.save_to_disk(intermediate_dir, max_shard_size='1GB', num_proc=args.nproc)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(intermediate_dir), str(output_dir))

if __name__ == '__main__':
    main()