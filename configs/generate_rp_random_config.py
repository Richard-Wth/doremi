import os
import json
import random
import argparse
from decimal import Decimal, getcontext
from pathlib import Path

def gen_uniform_config(domains, subdomain_num):
    getcontext().prec = 30
    total = Decimal("1")

    domain_num = len(domains)
    all_subdomain_num = int(domain_num * subdomain_num)
    parts = all_subdomain_num

    each_part = total / Decimal(parts)

    all_parts = []

    for _ in range(parts - 1):
        rounded_part = each_part.quantize(Decimal('1.0000000000000000000000000000'))
        all_parts.append(rounded_part)
    last_part = total - sum(all_parts)
    all_parts.append(last_part)

    subdomain_names = []
    for domain in domains:
        for i in range(subdomain_num):
            subdomain_name = domain + "_" + str(i)
            subdomain_names.append(subdomain_name)
    subdomain_weights = {}
    for j in range(all_subdomain_num):
        subdomain_weights[subdomain_names[j]] = float(all_parts[j])
    return subdomain_weights

def gen_random_config(domains, subdomain_num):
    def _generate_random_parts(num_parts):
        # generate random number
        random_numbers = [random.random() for _ in range(num_parts)]
        total = sum(random_numbers)
        normalized_numbers = [num / total for num in random_numbers]
        return normalized_numbers
    
    domain_num = len(domains)
    all_subdomain_num = int(domain_num * subdomain_num)
    parts = _generate_random_parts(all_subdomain_num)

    random_subdomain_weights = {}
    subdomain_names = []
    for domain in domains:
        for i in range(subdomain_num):
            subdomain_name = domain + "_" + str(i)
            subdomain_names.append(subdomain_name)
    
    for j in range(all_subdomain_num):
        random_subdomain_weights[subdomain_names[j]] = float(parts[j])
    return random_subdomain_weights

def main():
    '''
    Generate a config.json of the subdomain dataset
    '''
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,default='/path/to/config.json')
    parser.add_argument('--subdomain_num', type=int, default=10)
    parser.add_argument('--output_file_name', type=str, default='output_file_name')
    args = parser.parse_args()

    domains = ['RedPajamaStackExchange', 'RedPajamaArXiv', 'RedPajamaCommonCrawl', 'RedPajamaC4', 'RedPajamaBook', 'RedPajamaGithub', 'RedPajamaWikipedia']

    output_dir_path = os.path.join(args.output_dir, args.output_file_name)
    rp_original_path = "/home/wth/My_codes/doremi/configs/rp_uniform.json"
    fr = open(rp_original_path, "r", encoding="utf-8")
    rp_uniform = json.load(fr)
    eval_rp_weights = rp_uniform["eval_domain_weights"]

    # modify this setting when you want to generate the random weights
    # # generate uniform weights
    # rp_uniform_subdomain = {}
    # uniform_weights = gen_uniform_config(domains, args.subdomain_num)
    # rp_uniform_subdomain["train_domain_weights"] = uniform_weights
    # rp_uniform_subdomain["eval_domain_weights"] = eval_rp_weights

    # generate random weights
    rp_random_subdomain = {}
    random_weights = gen_random_config(domains, args.subdomain_num)
    rp_random_subdomain["train_domain_weights"] = random_weights
    rp_random_subdomain["eval_domain_weights"] = eval_rp_weights

    # write in json file 
    subdomain_fw = open(output_dir_path, "w", encoding="utf-8")
    # json.dump(rp_uniform_subdomain, subdomain_fw)
    json.dump(rp_random_subdomain, subdomain_fw)
    print("=== finish writing ===")

if __name__ == "__main__":
    main()


