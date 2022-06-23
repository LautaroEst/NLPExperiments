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
from lightning_models import ClassificationModel, RegressionModel, TBLogger
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
    is_torch_model = main_model.framework == "torch"

    if is_torch_model:
        configure_optimizers=config.pop("configure_optimizers")
        optimizer_step=config.pop("optimizer_step")

        if main_model.task == "classification":
            model = ClassificationModel(features_extractor,main_model,configure_optimizers,optimizer_step)
        elif main_model.task == "regression":
            model = RegressionModel(features_extractor,main_model,configure_optimizers,optimizer_step)

        logger = TBLogger(save_dir=output_dir,name="",default_hp_metric=False)
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_dir,"version_0/checkpoints"),
            filename="step={step}-epoch={epoch}-val_loss={loss/validation:.2f}",
            monitor="loss/validation",
            save_top_k=1,
            save_weights_only=False,
            auto_insert_metric_name=False
        )
        trainer = pl.Trainer(
            default_root_dir=output_dir,
            logger=logger,
            callbacks=[checkpoint_callback],
            deterministic=True,
            # fast_dev_run=True, # Uncomment to simulate training
            enable_checkpointing=config["enable_checkpointing"],
            min_epochs=config["min_epochs"],
            max_epochs=config["max_epochs"],
            accelerator=config["accelerator"],
            devices=config["devices"],
            val_check_interval=config["val_check_interval"],
            log_every_n_steps=config["log_every_n_steps"]
        )
        trainer.fit(model,dataset["train"],dataset["validation"])


        best_model_name = os.path.join(output_dir,"version_0/checkpoints/",sorted(
            os.listdir(os.path.join(output_dir,"version_0/checkpoints/")),
            key=lambda ckp: float(ckp.split("val_loss=")[1].split(".ckpt")[0]),
            reverse=True
        )[0])

        results = trainer.validate(model,dataset["validation"],best_model_name,verbose=True)

    else:
        
        vectorized_datasets = {}
        for split in ["train", "validation"]:
            X = features_extractor.extractor.fit_transform(dataset[split]) if split == "train" else \
                features_extractor.extractor.transform(dataset[split])
            vectorized_datasets[split] = {
                "X": X,
                "y": np.array(dataset[split]["label"])
            }
        
        main_model.model.fit(
            vectorized_datasets["train"]["X"],
            vectorized_datasets["train"]["y"]
        )

        y_preds = {
            split: main_model.model.predict(vectorized_datasets[split]["X"]) \
            for split in ["train", "validation"]
        }

        results = {}
        for split in ["train", "validation"]:
            results[f"f1-score/{split}"] = f1_score(vectorized_datasets[split]["y"],y_preds[split],average="macro")
            results[f"accuracy/{split}"] = accuracy_score(vectorized_datasets[split]["y"],y_preds[split])
        logger = None
        
    return results, logger


def show_results(results,logger,is_torch_model,output_dir,**config):

    if is_torch_model:
        results = {f"{key.split('/')[1]}/{key.split('/')[0]}": val for key, val in results[0].items()}
        logger.experiment.add_hparams(
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
    print(results)


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
    results, logger = train_model(features_extractor,main_model,dataset,output_dir,**config)

    # Show results
    show_results(results,logger,is_torch_model,output_dir,**config)





if __name__ == "__main__":
    main()