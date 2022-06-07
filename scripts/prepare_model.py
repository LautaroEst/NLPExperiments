import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from import_config import import_configs_objs
from datasets import load_dataset, DatasetDict

from nlp.models import init_model, save_model


def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--config",help="Json file with the tokenizer loading configuration")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Process the arguments
    config_objs = import_configs_objs(args["config"])["config"]
    output_dir = args["out"]

    return config_objs, output_dir


def main():
    config, output_dir = parse_args()
    model_name = config.pop("type")
    torch_model = init_model(model_name,**config)
    save_model(model_name,torch_model,output_dir)
    print(torch_model)
    

    
if __name__ == "__main__":
    main()