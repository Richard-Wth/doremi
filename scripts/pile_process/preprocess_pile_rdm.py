import random
from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
from pathlib import Path
from itertools import cycle
import torch
import os
import numpy as np
from datasets import Features, Sequence, Value
import shutil
from itertools import chain
from tokenizers.processors import TemplateProcessing

PILE_DOMAINS = ['BookCorpus2', 'Books3', 'Gutenberg (PG-19)', 'HackerNews', 'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers', 'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange', 'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles']

DOMAIN_TO_IDX = {name: idx for idx, name in enumerate(PILE_DOMAINS)}

NUM_SUBDOMAINS = 10

PILE_SUBSETS = [f'0{i}' if i < 10 else str(i) for i in range(0, 30)]

def pile_transform(tokenizer, max_length, seed=None):
    def transform(batch):
        examples = tokenizer(batch['text'])
        examples = {k: list(chain(*examples[k])) for k in examples.keys() if k != 'attention_mask'}
        total_length = len(examples[list(examples.keys())[0]])
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in examples.items()
        }
        return result
    return transform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pile_path_dir', type=str, default='/path/to/pile')
    parser.add_argument('--output_dir', type=str, default='/path/to/output')
    parser.add_argument('--intermediate_dir', type=str, default='/path/to/intermediate')
    parser.add_argument('--domain', type=str, default='Books3')
    parser.add_argument('--subset', type=str, default='01')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--cache_dir', type=str, default='/path/to/cache')
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    args.domain = args.domain.replace('_', ' ')
    random.seed(args.seed)

    pile_dir = Path(args.pile_path_dir)
    if args.split == 'train':
        train_pile_dir = Path(os.path.join(pile_dir, args.split))
        data_files = [str(train_pile_dir / f"{args.subset}.jsonl.zst")]
    elif args.split == 'validation':
        data_files = [str(pile_dir / "val.jsonl.zst")]
    else:
        data_files = [str(pile_dir / f"test.jsonl.zst")]

    ds = load_dataset('json',
                      data_files=data_files,
                      cache_dir=args.cache_dir,
                      streaming=True)['train']

    def filter_fn(ex, idx):
        return ex['meta']['pile_set_name'] == args.domain

    def domain_id_fn(ex):
        return DOMAIN_TO_IDX[args.domain]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="$A "+tokenizer.eos_token,
            special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)])
    transform = pile_transform(tokenizer, args.max_length, seed=args.seed)

    ds = ds.filter(filter_fn, with_indices=True)
    ds = ds.map(transform, batched=True, remove_columns=['text', 'meta'])

    def data_generator():
        for i, ex in enumerate(ds):
            ex['domain_id'] = domain_id_fn(ex)
            ex['subdomain_id'] = random.randint(0, NUM_SUBDOMAINS - 1)
            yield ex

    features = Features({
            "input_ids": Sequence(Value("int32")),
            "domain_id": Value("int32"),
            "subdomain_id": Value("int32"),
        })
    processed_ds = Dataset.from_generator(data_generator, features=features)

    # Save dataset for each subdomain
    for subdomain_id in range(NUM_SUBDOMAINS):
        if args.split == 'train':
            subdomain_dir = Path(args.intermediate_dir) / args.split / args.domain / f"subdomain_{subdomain_id}" / args.subset
        else:
            subdomain_dir = Path(args.intermediate_dir) / args.split / args.domain / f"subdomain_{subdomain_id}"

        subdomain_dir.parent.mkdir(parents=True, exist_ok=True)
        
        subdomain_ds = processed_ds.filter(lambda example: example['subdomain_id'] == subdomain_id)
        subdomain_ds = subdomain_ds.remove_columns(['subdomain_id'])
        
        subdomain_ds.save_to_disk(subdomain_dir, max_shard_size='1GB', num_proc=args.nproc)

        # Move from intermediate dir to output dir
        output_subdomain_dir = Path(args.output_dir) / subdomain_dir.relative_to(args.intermediate_dir)
        output_subdomain_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(subdomain_dir), str(output_subdomain_dir))

if __name__ == '__main__':
    main()