import math
import json
import random
from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
from pathlib import Path
from itertools import cycle
import torch
import numpy as np
from datasets import Features, Sequence, Value
import shutil
from itertools import chain
from tokenizers.processors import TemplateProcessing

NUM_SUBDOMAINS = 10

def get_transform(tokenizer, max_length, domain_id, seed=None):
    def transform(batch):
        # tokenize
        examples = tokenizer(batch["text"])

        # Concatenate all texts.
        examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(examples[list(examples.keys())[0]])
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in examples.items()
        }
        result['domain_id'] = [domain_id for _ in range(0, total_length, max_length)]
        return result
    return transform

def assign_to_subdomain(dataset, num_subdomains):
    dataset_size = len(dataset)
    subdomain_size = dataset_size // num_subdomains
    remainder = dataset_size % num_subdomains

    subdomain_datasets = [[] for _ in range(num_subdomains)]
    indices = list(range(dataset_size))
    random.shuffle(indices)

    start = 0
    for i in range(num_subdomains):
        end = start + subdomain_size + (1 if i < remainder else 0)
        subdomain_indices = indices[start: end]
        subdomain_datasets[i] = dataset.select(subdomain_indices)
        start = end
    
    return subdomain_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/path/to/redpajama')
    parser.add_argument('--output_dir', type=str, default='/path/to/output_dir')
    parser.add_argument('--domain', type=str, default='common_crawl')
    parser.add_argument('--subset', type=int, default=0)
    parser.add_argument('--num_subsets', type=int, default=95)
    parser.add_argument('--num_validation_examples', type=int, default=1000000)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--tokenizer', type=str, default='/home/wth/My_codes/doremi/tokenizer')
    parser.add_argument('--cache_dir', type=str, default='/path/to/cache')
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    output_dir_train = Path(args.output_dir) / 'train' / args.domain / str(args.subset)
    output_dir_val = Path(args.output_dir) / 'validation' / args.domain / str(args.subset)
    if output_dir_train.exists() and output_dir_val.exists():
        print("Already done, skipping")
        return
    
    dataset_dir = Path(args.dataset_dir)
    DOMAINS = list(sorted([str(domain_dir.name) for domain_dir in dataset_dir.iterdir() if not str(domain_dir.name).endswith('txt')]))
    DOMAIN_TO_IDX = {name: idx for idx, name in enumerate(DOMAINS)}

    # figure out data files
    assert(args.domain in DOMAINS)

    domain_dir = dataset_dir /args.domain

    all_files = [str(path) for path in domain_dir.rglob("*") if not path.is_dir()]
    random.Random(42).shuffle(all_files)

    # split into chunks
    num_subsets = min(len(all_files), args.num_subsets)
    if args.subset >= num_subsets:
        return
    
    subset_size = len(all_files) // num_subsets
    if args.subset < num_subsets - 1:
        data_files = all_files[args.subset*subset_size:(args.subset + 1)*subset_size]
    else:
        data_files = all_files[args.subset*subset_size:]

    # load dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # add s separator token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="$A "+tokenizer.eos_token,
            special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)])
    transform = get_transform(tokenizer, args.max_length, DOMAIN_TO_IDX[args.domain], seed=args.seed)
    
    if args.domain not in {'RedPajamaBook', 'RedPajamaGithub'}:
        ds = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=args.cache_dir,
            num_proc=args.nproc
        )['train']
        # split into train and validation
        if args.num_validation_examples > len(ds):
            num_validation_examples = len(ds) // 20
        else:
            num_validation_examples = args.num_validation_examples
        ds_dict = ds.train_test_split(test_size=num_validation_examples, shuffle=True, seed=args.seed)
        ds_train = ds_dict['train'].map(
            transform,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=args.nproc
        )
        ds_val = ds_dict['test'].map(
            transform,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=args.nproc
        )
    else:
        def data_gen(data_files):
            for data_file in data_files:
                with open(data_file, 'r') as f:
                    for line in f:
                        try:
                            ex = json.loads(line)
                            ex.pop('meta')
                            yield ex
                        except Exception:
                            # sometimes there is a json read error
                            pass
        ds = Dataset.from_generator(data_gen, gen_kwargs={'data_files': data_files}, num_proc=args.nproc)
        # split into train and validation
        if args.num_validation_examples > len(ds):
            num_validation_examples = len(ds) // 20
        else:
            num_validation_examples = args.num_validation_examples
        ds = ds.train_test_split(test_size=num_validation_examples, shuffle=True, seed=args.seed)
        if args.domain == "RedPajamaBook":
            batch_size = 10
        else:
            batch_size = 1000
        
        ds_train = ds["train"].map(
            transform,
            batched=True,
            batch_size=batch_size,
            remove_columns=['text'],
            num_proc=args.nproc
        )
        print(ds)
        ds_val = ds["test"].map(
            transform,
            batched=True,
            batch_size=batch_size,
            remove_columns=['text'],
            num_proc=args.nproc
        )
    
    # Assign to subdomains
    train_subdomains = assign_to_subdomain(ds_train, NUM_SUBDOMAINS)

    # Saving training datasets
    for subdomain_id, subdomain_ds in enumerate(train_subdomains):
        subdomain_name = str(args.domain) + "_" + str(subdomain_id)
        output_dir = Path(args.output_dir) / 'train' / args.domain / subdomain_name / str(args.subset)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        subdomain_ds.save_to_disk(output_dir, max_shard_size='1GB', num_proc=args.nproc)
    
    # Save validation dataset (not split into subdomains)
    val_output_dir = Path(args.output_dir) / 'validation' / args.domain / str(args.subset)
    val_output_dir.parent.mkdir(parents=True, exist_ok=True)
    ds_val.save_to_disk(val_output_dir, max_shard_size='1GB', num_proc=args.nproc)

    # shutil.rmtree(str(Path(args.cache_dir) / 'downloads'))
    # shutil.rmtree(str(Path(args.cache_dir) / 'downloads'))

if __name__ == "__main__":
    main()
