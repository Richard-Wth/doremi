import os
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from datasets import load_dataset, Dataset
from cusim import CuLDA
import json
import zstandard
from tqdm import tqdm
import random

PILE_DOMAINS = ['ArXiv', 'BookCorpus2', 'Books3', 'DM Mathematics', 'Enron Emails', 'EuroParl', 'FreeLaw', 'Github', 'Gutenberg (PG-19)', 'HackerNews', 'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers', 'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange', 'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles']

NUM_SUBDOMAINS = 10
BATCH_SIZE = 10000  # Adjust this based on your available memory

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_and_filter_data(args):
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
    
    return ds.filter(lambda x: x['meta']['pile_set_name'] == args.domain)

def prepare_culda_data(texts, output_dir, rank, world_size):
    local_vocab = set()
    local_word_counts = []
    for text in texts:
        words = text.split()
        local_vocab.update(words)
        local_word_counts.append(dict((word, words.count(word)) for word in set(words)))

    all_vocabs = [None for _ in range(world_size)]
    all_word_counts = [None for _ in range(world_size)]
    dist.all_gather_object(all_vocabs, local_vocab)
    dist.all_gather_object(all_word_counts, local_word_counts)

    if rank == 0:
        vocab = set().union(*all_vocabs)
        word_counts = [item for sublist in all_word_counts for item in sublist]

        data_path = os.path.join(output_dir, "docword.txt")
        keys_path = os.path.join(output_dir, "vocab.txt")
        processed_data_path = os.path.join(output_dir, "docword.h5")

        with open(keys_path, 'w') as f:
            for word in vocab:
                f.write(f"{word}\n")

        with open(data_path, 'w') as f:
            f.write(f"{len(word_counts)} {len(vocab)} {sum(sum(doc.values()) for doc in word_counts)}\n")
            for doc_id, doc in enumerate(word_counts):
                for word, count in doc.items():
                    f.write(f"{doc_id + 1} {word} {count}\n")

        return data_path, keys_path, processed_data_path
    else:
        return None, None, None

class CuLDAWrapper(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.lda = CuLDA(opt)

    def forward(self):
        self.lda.train_model()
        return self.lda.get_doc_topic_dist()

def run_lda(rank, world_size, args, texts):
    setup(rank, world_size)
    
    data_path, keys_path, processed_data_path = prepare_culda_data(texts, args.intermediate_dir, rank, world_size)
    
    if rank == 0:
        opt = {
            "data_path": data_path,
            "processed_data_path": processed_data_path,
            "keys_path": keys_path,
            "num_topics": NUM_SUBDOMAINS,
            "num_iters_in_e_step": 10,
            "reuse_gamma": True,
        }
        
        lda_model = CuLDAWrapper(opt)
        lda_model = DDP(lda_model, device_ids=[rank])
        
        doc_topic_dist = lda_model()
        doc_topics = np.argmax(doc_topic_dist, axis=1)
    else:
        doc_topics = None

    doc_topics = [doc_topics]
    dist.broadcast_object_list(doc_topics, src=0)
    doc_topics = doc_topics[0]
    
    cleanup()
    
    return doc_topics

def save_to_subdomain_batch(data_batch, doc_topics_batch, output_dir, domain, subset, rank, batch_id):
    cctx = zstandard.ZstdCompressor()
    for subdomain in range(NUM_SUBDOMAINS):
        if subset:
            output_path = os.path.join(output_dir, domain, f"subdomain_{subdomain}", f"{subset}_part{rank}_batch{batch_id}.jsonl.zst")
        else:
            output_path = os.path.join(output_dir, domain, f"subdomain_{subdomain}_part{rank}_batch{batch_id}.jsonl.zst")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            with cctx.stream_writer(f) as compressor:
                for item, topic in zip(data_batch, doc_topics_batch):
                    if topic == subdomain:
                        compressor.write(json.dumps(item).encode('utf-8') + b'\n')

def process_batch(rank, world_size, args, batch_data, batch_id):
    texts = [item['text'] for item in batch_data]
    doc_topics = run_lda(rank, world_size, args, texts)
    save_to_subdomain_batch(batch_data, doc_topics, args.output_dir, args.domain, args.subset, rank, batch_id)

def main(rank, world_size, args):
    setup(rank, world_size)

    ds = load_and_filter_data(args)

    batch_data = []
    batch_id = 0
    for item in tqdm(ds, desc=f"Processing data on rank {rank}"):
        batch_data.append(item)
        if len(batch_data) == BATCH_SIZE:
            process_batch(rank, world_size, args, batch_data, batch_id)
            batch_data = []
            batch_id += 1

    if batch_data:
        process_batch(rank, world_size, args, batch_data, batch_id)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pile_path_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--intermediate_dir', type=str, required=True)
    parser.add_argument('--domain', type=str, required=True)
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--cache_dir', type=str, default='/tmp/huggingface_cache')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)