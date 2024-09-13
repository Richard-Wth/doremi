import os
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
from itertools import chain
from tokenizers.processors import TemplateProcessing
from cusim import CuLDA
import time
import h5py
from tqdm import tqdm

try:
    import zstandard
except ImportError:
    raise ImportError("Please install zstandard: pip install zstandard")

PILE_DOMAINS = ['ArXiv', 'BookCorpus2', 'Books3', 'DM Mathematics', 'Enron Emails', 'EuroParl', 'FreeLaw', 'Github', 'Gutenberg (PG-19)', 'HackerNews', 'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers', 'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange', 'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles']

DOMAIN_TO_IDX = {name: idx for idx, name in enumerate(PILE_DOMAINS)}

NUM_SUBDOMAINS = 10

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_and_filter_data(args):
    if args.split == 'train':
        data_files = [str(Path(args.pile_path_dir) / args.split / f"{args.subset}.jsonl.zst")]
    elif args.split == 'validation':
        data_files = [str(Path(args.pile_path_dir) / "val.jsonl.zst")]
    else:
        data_files = [str(Path(args.pile_path_dir) / "test.jsonl.zst")]

    # 使用 'text' 加载方式，然后手动解析 JSON
    ds = load_dataset('text', data_files=data_files, cache_dir=args.cache_dir, streaming=True)['train']
    
    def parse_and_filter(ex):
        import json
        try:
            data = json.loads(ex['text'])
            if data['meta']['pile_set_name'] == args.domain:
                return data
        except json.JSONDecodeError:
            pass
        return None

    return ds.map(parse_and_filter, remove_columns=['text']).filter(lambda x: x is not None)

def prepare_culda_data(texts, args):
    # Create vocab
    vocab = set()
    word_counts = []
    for text in texts:
        words = text.split()
        vocab.update(words)
        word_counts.append(dict((word, words.count(word)) for word in set(words)))

    # Prepare data for CuLDA
    data_path = os.path.join(args.intermediate_dir, f"docword.{args.domain}.txt")
    keys_path = os.path.join(args.intermediate_dir, f"vocab.{args.domain}.txt")
    processed_data_path = os.path.join(args.intermediate_dir, f"docword.{args.domain}.h5")

    # Create vocab file
    with open(keys_path, 'w') as f:
        for word in vocab:
            f.write(f"{word}\n")

    # Create document-word file
    with open(data_path, 'w') as f:
        f.write(f"{len(word_counts)} {len(vocab)} {sum(sum(doc.values()) for doc in word_counts)}\n")
        for doc_id, doc in enumerate(word_counts):
            for word, count in doc.items():
                f.write(f"{doc_id + 1} {word} {count}\n")

    return data_path, keys_path, processed_data_path, word_counts

def run_lda(rank, world_size, args, texts):
    setup(rank, world_size)
    
    data_path, keys_path, processed_data_path, _ = prepare_culda_data(texts, args)
    
    opt = {
        "data_path": data_path,
        "processed_data_path": processed_data_path,
        "keys_path": keys_path,
        "num_topics": NUM_SUBDOMAINS,
        "num_iters_in_e_step": 10,
        "reuse_gamma": True,
    }
    
    start = time.time()
    lda = CuLDA(opt)
    lda.train_model()
    print(f"LDA training time: {time.time() - start} seconds")

    # Get document-topic distributions
    with h5py.File(processed_data_path, 'r') as f:
        doc_topic_dist = f['theta'][:]

    # Assign topics to documents
    doc_topics = np.argmax(doc_topic_dist, axis=1)
    
    cleanup()
    
    return doc_topics

def tokenize_and_process(example, tokenizer, max_length):
    tokens = tokenizer(example['text'], truncation=True, max_length=max_length)
    return {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask'],
        'domain_id': DOMAIN_TO_IDX[example['meta']['pile_set_name']]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pile_path_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--intermediate_dir', type=str, required=True)
    parser.add_argument('--domain', type=str, required=True)
    parser.add_argument('--subset', type=str, default='01')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--cache_dir', type=str, default='/tmp/huggingface_cache')
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    args.domain = args.domain.replace('_', ' ')

    # Step 1: Load and filter data
    ds = load_and_filter_data(args)

    # Step 2: Extract texts for LDA
    texts = [example['text'] for example in ds]

    # Step 3: Run LDA
    world_size = torch.cuda.device_count()
    doc_topics = mp.spawn(run_lda, args=(world_size, args, texts), nprocs=world_size, join=True)

    # Step 4: Tokenize and process data
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="$A " + tokenizer.eos_token,
        special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)]
    )

    # Step 5: Process and save data by subdomain
    for subdomain_id in range(NUM_SUBDOMAINS):
        if args.split == 'train':
            subdomain_dir = Path(args.output_dir) / args.split / args.domain / f"subdomain_{subdomain_id}" / args.subset
        else:
            subdomain_dir = Path(args.output_dir) / args.split / args.domain / f"subdomain_{subdomain_id}"

        subdomain_dir.parent.mkdir(parents=True, exist_ok=True)

        def data_generator():
            for example, topic in zip(ds, doc_topics):
                if topic == subdomain_id:
                    yield tokenize_and_process(example, tokenizer, args.max_length)

        features = Features({
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int32")),
            "domain_id": Value("int32"),
        })

        subdomain_ds = Dataset.from_generator(data_generator, features=features)
        subdomain_ds.save_to_disk(subdomain_dir, max_shard_size='1GB', num_proc=args.nproc)

        print(f"Saved {len(subdomain_ds)} examples to {subdomain_dir}")

if __name__ == "__main__":
    main()