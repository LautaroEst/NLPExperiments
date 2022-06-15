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


def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--config",help="Python file with the tokenizer loading configuration")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Extract config and output directory
    output_dir = args["out"]
    config = import_configs_objs(args["config"])["config"]

    return config, output_dir


def load_pretrained_tokenizer(**config):
    tokenizer = AutoTokenizer.from_pretrained(**config)
    return tokenizer