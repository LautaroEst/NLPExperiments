

import argparse
import json
import os

from transformers import AutoTokenizer
from import_config import import_configs_objs
from nlp.features import load_features_extractor


def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--tokenizer_dir",help="Directory that holds the tokenizer files")
    parser.add_argument("--features_dir",help="File with the feature extractor configuration.")
    parser.add_argument("--data_dir",help="Directory that holds the preprocessed data")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Process the arguments
    directories = {
        "tokenizer": args["tokenizer_dir"],
        "features": args["features_dir"],
        "data": args["data_dir"],
    }
    config = None #import_configs_objs(args["training_config"])["config"]
    output_dir = args["out"]

    return config, directories, output_dir



def load_tokenizer(tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer


def load_extractor(features_dir):
    with open(os.path.join(features_dir,"params.json"),"r") as f:
        extractor_type = json.load(f)["type"]
    extractor = load_features_extractor(extractor_type,features_dir)
    return extractor


def load_data(data_dir):
    dataset = None
    return dataset



def inspect_features(tokenizer,features_extractor,dataset,output_dir,**config):
    pass


def main():
    config, directories, output_dir = parse_args()
    tokenizer = load_tokenizer(directories["tokenizer"])
    features_extractor = load_extractor(directories["features"])
    dataset = load_data(directories["data"])

    inspect_features(tokenizer,features_extractor,dataset,output_dir,**config)



if __name__ == "__main__":
    main()