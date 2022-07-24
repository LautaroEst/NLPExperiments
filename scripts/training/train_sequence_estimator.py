import json
import os
from datasets import DatasetDict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from utils import (
    parse_args,
    load_extractor, load_main_model, load_tokenizer,
    load_data_in_batches
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from nlp.trainers import SupervisedNeuralModelTrainer, SupervisedGenericMLModelTrainer
from torch import nn


def prepare_data_for_training(tokenizer,data_dir,is_torch_model,**config):
    if is_torch_model:
        dataset = load_data_in_batches(
            data_dir,
            tokenizer,
            config["train_batch_size"],
            config["validation_batch_size"],
            config["padding_strategy"],
            config["num_workers"]
        )

    else:
        dataset = DatasetDict.load_from_disk(data_dir)
            
    return dataset


def train_model(features_extractor,main_model,dataset,output_dir,**config):
    is_torch_model = isinstance(main_model,nn.Module)
    config["output_dir"] = output_dir
    if is_torch_model:
        trainer = SupervisedNeuralModelTrainer(features_extractor,main_model,**config)
    else:
        trainer = SupervisedGenericMLModelTrainer(features_extractor,main_model,**config)
        
    trainer.fit(dataset)

    return trainer


def validate_model_and_save_results(trainer,dataset,output_dir,**config):

    results = trainer.validate(dataset)

    is_neural_model = isinstance(trainer,SupervisedNeuralModelTrainer)
    if is_neural_model:
        trainer.logger.experiment.add_hparams(
            dict(
                train_batch_size=config["train_batch_size"],
                validation_batch_size=config["validation_batch_size"],
                min_epochs=config["min_epochs"],
                max_epochs=config["max_epochs"],
                val_check_interval=config["val_check_interval"]
            ),
            results
        )

    with open(os.path.join(output_dir,"results.json"),"w") as f:
        json.dump(results,f,indent=4,separators=", ")
    
    return results


def main():
    # Argument parser:
    config, directories, output_dir = parse_args()

    # Load the pretrained tokenizer from local directory
    tokenizer = load_tokenizer(directories["tokenizer"])

    # Load the initialized extractor from local_directory
    features_extractor = load_extractor(tokenizer,directories["features"])

    # Load the initialized model from local_directory
    main_model = load_main_model(directories["model"])

    # Prepare data for training
    is_torch_model = isinstance(main_model,nn.Module)
    dataset = prepare_data_for_training(tokenizer,directories["data"],is_torch_model,**config)

    # Train model
    trainer = train_model(features_extractor,main_model,dataset,output_dir,**config)

    # Show results
    results = validate_model_and_save_results(trainer,dataset,output_dir,**config)
    print(results)




if __name__ == "__main__":
    main()