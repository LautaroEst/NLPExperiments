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



def prepare_data_for_training(
        tokenizer: PreTrainedTokenizerFast,
        output_dir,
        **config
    ):

    preprocessing_function = config.pop("preprocessing_function")

    data_dict = {}
    for split in ["train", "validation"]:
        dataset = config[f"{split}_data"]
        dataset = dataset.map(
            lambda sample: preprocessing_function(sample,tokenizer),
            **config["mapping_args"]
        )
        columns_to_remove = list(set(dataset.features.keys()) - config["columns"])
        dataset = dataset.remove_columns(columns_to_remove)
        data_dict[split] = dataset

    DatasetDict(data_dict).save_to_disk(output_dir)



def main():
    config, tokenizer_dir, output_dir = parse_args()
    tokenizer = load_tokenizer(tokenizer_dir)

    # Train and Validation Data:
    prepare_data_for_training(tokenizer,output_dir,**config)

    
if __name__ == "__main__":
    main()