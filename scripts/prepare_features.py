import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from import_config import import_configs_objs
from datasets import load_dataset, DatasetDict

from nlp.features import init_features_extractor, save_features_extractor


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


def main():
    config, tokenizer_dir, output_dir = parse_args()
    tokenizer = load_tokenizer(tokenizer_dir)
    extractor_model = config.pop("type")
    config["tokenizer"] = tokenizer
    extractor = init_features_extractor(extractor_model,**config)
    save_features_extractor(extractor_model,extractor,output_dir)
    pass
    

    
if __name__ == "__main__":
    main()