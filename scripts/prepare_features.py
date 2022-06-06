import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from import_config import import_configs_objs
from datasets import load_dataset, DatasetDict


def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--config",help="Json file with the tokenizer loading configuration")
    parser.add_argument("--tokenizer_dir",help="Directory that holds the tokenizer files")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Process the arguments
    config_objs = import_configs_objs(args["config"])["config"]
    tokenizer_dir = args["tokenizer_dir"]
    output_dir = args["out"]

    return config_objs, tokenizer_dir, output_dir


def load_tokenizer(tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer


def init_features_extractor(tokenizer,**config):
    extractor_class = config.pop("features_extractor_class")
    extractor = extractor_class(tokenizer,**config)
    return extractor



def main():
    # config, tokenizer_dir, output_dir = parse_args()
    # tokenizer = load_tokenizer(tokenizer_dir)
    # extractor = init_features_extractor(tokenizer,**config)
    # torch.save(extractor,os.path.join(output_dir,"extractor.pkl"))
    pass
    

    
if __name__ == "__main__":
    main()