import argparse
from importlib.machinery import SourceFileLoader
from types import ModuleType

from transformers import AutoTokenizer

def import_configs_objs(config_file):
    """Dynamicaly loads the configuration file"""
    
    loader = SourceFileLoader('config', config_file)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)
    config_objs = vars(mod)
    return config_objs


def load_tokenizer(tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer

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