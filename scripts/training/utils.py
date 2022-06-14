import argparse
from importlib.machinery import SourceFileLoader
import json
import os
from types import ModuleType
from datasets import DatasetDict

from transformers import AutoTokenizer
from nlp.features import load_features_extractor
from nlp.models import load_model, SKLEARN_MODELS
from torch.utils.data import DataLoader


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
    parser.add_argument("--tokenizer_dir",help="Directory that holds the tokenizer files")
    parser.add_argument("--features_dir",help="File with the feature extractor configuration.")
    parser.add_argument("--model_dir",help="File with the model configuration.")
    parser.add_argument("--data_dir",help="Directory that holds the preprocessed data")
    parser.add_argument("--training_config",help="File with the training configuration.")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Process the arguments
    directories = {
        "tokenizer": args["tokenizer_dir"],
        "features": args["features_dir"],
        "data": args["data_dir"],
        "model": args["model_dir"],
    }
    config = import_configs_objs(args["training_config"])["config"]
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


def load_main_model(model_dir):
    with open(os.path.join(model_dir,"params.json"),"r") as f:
        params = json.load(f)
    model_type = params["type"]
    task = params["task"]
    model = load_model(model_type,model_dir)
    is_sklearn_model = model_type in SKLEARN_MODELS
    return model, is_sklearn_model, task


def load_data_in_batches(
        data_dir,
        tokenizer,
        train_batch_size,
        validation_batch_size,
        padding_strategy,
        num_workers
    ):

    def data_collate(batch,tokenizer,padding,max_legth):
        return tokenizer.pad(
            batch,
            return_tensors="pt",
            padding=padding,
            max_length=max_legth
        )

    padding = padding_strategy
    max_length = tokenizer.model_max_length if padding == "max_length" else None
    batch_sizes = {"train": train_batch_size, "validation": validation_batch_size}

    dataset = DatasetDict.load_from_disk(data_dir)
    dataloaders = {}
    
    for split in ["train","validation"]:
        dataset[split].set_format(type='torch', columns=list(dataset[split].features.keys()))
        dataloaders[split] = DataLoader(
            dataset[split],
            batch_size=batch_sizes[split],
            shuffle= split == "train",
            collate_fn=lambda batch: data_collate(batch,tokenizer,padding,max_length),
            num_workers=num_workers
        )
    return dataloaders