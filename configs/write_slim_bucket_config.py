from pathlib import Path
import json
import os

preprocessed_dir = "/home/wth/My_codes/doremi/data/slim_preprocessed/random_preprocessed"
json_path = "/home/wth/My_codes/doremi/configs/slim_uniform_bucket.json"

def gen_weights(preprocessed_dir, split):
    preprocessed_dir = Path(preprocessed_dir) / split
    domain_names = []
    for domain in preprocessed_dir.iterdir():
        domain_names.append(domain.name)
    domains_num = len(domain_names)
    uniform_domain_weight = float(1 / domains_num)
    domain_weights = {}
    domain_names = sorted(domain_names)
    for domain in domain_names:
        domain_weights[domain] = uniform_domain_weight
    return domain_weights

train_domain_weights = gen_weights(preprocessed_dir, split='train')
eval_domain_weights = gen_weights(preprocessed_dir, 'validation')
uniform_bucket_weight = {}
uniform_bucket_weight["train_domain_weights"] = train_domain_weights
uniform_bucket_weight["eval_domain_weights"] = eval_domain_weights

with open(json_path, 'w', encoding="utf-8") as fw:
    json.dump(uniform_bucket_weight, fw)
    fw.close()
