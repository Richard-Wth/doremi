# We transfer SlimPajama Dataset into the form of :
"""
```
top_level/
    domain_name_1/
        files...
    domain_name_2/
        files...
    ...
```
"""

import jsonlines
import json
import os
from collections import defaultdict

data_path = "/home/wth/My_codes/doremi/data/slimpajama/subdataset/train"
output_path = "/home/wth/My_codes/doremi/scripts/redpajama/train"

def write_to_jsonl(domain, data, max_size_mb=500):
    # 创建以domain命名的文件夹
    os.makedirs(domain, exist_ok=True)
    file_index = 1
    current_size = 0
    current_file = None

    for item in data:
        item_size = len(json.dumps(item).encode('utf-8'))
        
        if current_file is None or current_size + item_size > max_size_mb * 1024 * 1024:
            if current_file:
                current_file.close()
            
            # 在domain文件夹下创建新的jsonl文件
            filename = os.path.join(output_path, f"{domain}_{file_index}.jsonl")
            current_file = open(filename, 'w', encoding='utf-8')
            file_index += 1
            current_size = 0

        json.dump(item, current_file)
        current_file.write('\n')
        current_size += item_size

    if current_file:
        current_file.close()


# Define domains in the SlimPajama
domains = [
    "RedPajamaStackExchange", "RedPajamaArXiv", "RedPajamaCommonCrawl",
    "RedPajamaC4", "RedPajamaBook", "RedPajamaGithub", "RedPajamaWikipedia"
]

domain_texts = defaultdict(list)

for file in os.listdir(data_path):
    data_file = os.path.join(data_path, file)

    with open(data_file, "r", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            domain_name = item["meta"]["redpajama_set_name"]
            if domain_name in domains:
                domain_texts[domain_name].append(item)

for domain in domains:
    print(f"Processing {domain}...")
    write_to_jsonl(domain, domain_texts[domain])
    print(f"Finished processing {domain}. Number of items: {len(domain_texts[domain])}")

print("All domains processed.")