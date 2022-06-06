import argparse
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from import_config import import_configs_objs
from datasets import DatasetDict



def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--tokenizer_dir",help="Directory that holds the tokenizer files")
    parser.add_argument("--features_config",help="File with the feature extractor configuration.")
    parser.add_argument("--model_config",help="File with the model configuration.")
    parser.add_argument("--data_dir",help="Directory that holds the preprocessed data")
    parser.add_argument("--training_config",help="File with the training configuration.")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Process the arguments
    config = {
        "tokenizer_dir": args["tokenizer_dir"],
        "features": import_configs_objs(args["features_config"])["config"],
        "data_dir": args["data_dir"],
        "model": import_configs_objs(args["model_config"])["config"],
        "training": import_configs_objs(args["training_config"])["config"]
    }
    output_dir = args["out"]

    return config, output_dir



def load_tokenizer(tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer


def load_features_extractor(tokenizer,**config):
    extractor_class = config.pop("features_extractor_class")
    extractor = extractor_class(tokenizer,**config)
    return extractor

def load_data(data_dir,tokenizer,**dataloader_args):

    data_collate = dataloader_args["data_collator"]

    dataset = DatasetDict.load_from_disk(data_dir)
    dataloaders = {}
    for split in ["train","validation"]:
        dataset[split].set_format(type='torch', columns=list(dataset[split].features.keys()))
        dataloaders[split] = DataLoader(
            dataset[split],
            batch_size=dataloader_args[f"{split}_batch_size"],
            shuffle= split == "train",
            collate_fn=lambda batch: data_collate(batch,tokenizer)
        )
    return dataloaders


def load_model(**config):
    model_class = config.pop("model_class")
    model = model_class(**config)
    return model



        

def train_sequence_classifier(
        features_extractor,main_model,train_dataloader,val_dataloader,**training_config
    ):

    full_model_class = training_config.pop("full_model_class")
    model = full_model_class(features_extractor,main_model)
    trainer = pl.Trainer(**training_config["trainer_args"])
    trainer.fit(model,train_dataloader,val_dataloader)

    results_history = None
    return results_history


def main():
    config, output_dir = parse_args()
    tokenizer = load_tokenizer(config["tokenizer_dir"])
    features_extractor = load_features_extractor(tokenizer,**config["features"])
    dataloaders = load_data(config["data_dir"],tokenizer,**config["training"]["dataloader_args"])
    main_model = load_model(**config["model"])

    result_history = train_sequence_classifier(
        features_extractor,
        main_model,
        dataloaders["train"],
        dataloaders["validation"],
        **config["training"]
    )



if __name__ == "__main__":
    main()