import json
import os
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import DatasetDict

from utils import load_data_in_batches, load_extractor, load_main_model, load_tokenizer, parse_args
from lightning_models import ClassificationModel, RegressionModel, TBLogger
from sklearn.metrics import f1_score, accuracy_score


def train_neural_sequence_classifier(
        features_extractor,main_model,data_dir,tokenizer,output_dir,**config
    ):

    train_batch_size=config.pop("train_batch_size")
    validation_batch_size=config.pop("validation_batch_size")
    padding_strategy=config.pop("padding_strategy")
    num_workers=config.pop("num_workers")
    dataloaders = load_data_in_batches(
        data_dir,
        tokenizer,
        train_batch_size,
        validation_batch_size,
        padding_strategy,
        num_workers
    )

    configure_optimizers=config.pop("configure_optimizers")
    optimizer_step=config.pop("optimizer_step")
    model = ClassificationModel(features_extractor,main_model,configure_optimizers,optimizer_step)

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
        **config
    )
    trainer.fit(model,dataloaders["train"],dataloaders["validation"])


    best_model_name = os.path.join(output_dir,"version_0/checkpoints/",sorted(
        os.listdir(os.path.join(output_dir,"version_0/checkpoints/")),
        key=lambda ckp: float(ckp.split("val_loss=")[1].split(".ckpt")[0]),
        reverse=True
    )[0])

    results = trainer.validate(model,dataloaders["validation"],best_model_name,verbose=True)

    return results, logger


def train_sklearn_sequence_classifier(
    features_extractor,main_model,data_dir,output_dir
):
    
    dataset = DatasetDict.load_from_disk(data_dir)

    vectorized_datasets = {}
    for split in ["train", "validation"]:
        X = features_extractor.fit_transform(dataset[split]) if split == "train" else \
            features_extractor.transform(dataset[split])
        vectorized_datasets[split] = {
            "X": X,
            "y": np.array(dataset[split]["label"])
        }
    
    main_model.fit(
        vectorized_datasets["train"]["X"],
        vectorized_datasets["train"]["y"]
    )

    y_preds = {
        split: main_model.predict(vectorized_datasets[split]["X"]) \
        for split in ["train", "validation"]
    }

    results = {}
    for split in ["train", "validation"]:
        results[f"f1-score/{split}"] = f1_score(vectorized_datasets[split]["y"],y_preds[split],average="macro")
        results[f"accuracy/{split}"] = accuracy_score(vectorized_datasets[split]["y"],y_preds[split])
    
    with open(os.path.join(output_dir,"results.json"),"w") as f:
        json.dump(results,f,indent=4,separators=", ")
        
    return results


def show_results(results,is_sklearn_model=False,logger=None,**config):

    if is_sklearn_model:
        print(results)
    
    else:
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

    
def main():

    # Argument parser:
    config, directories, output_dir = parse_args()

    # Load the pretrained tokenizer from local directory
    tokenizer = load_tokenizer(directories["tokenizer"])

    # Load the initialized extractor from local_directory
    features_extractor = load_extractor(directories["features"])

    # Load the initialized model from local_directory
    main_model, is_sklearn_model, task = load_main_model(directories["model"])

    # If model is not a NN...
    if is_sklearn_model:

        # ...train it with the sklearn function...
        results = train_sklearn_sequence_classifier(
            features_extractor,
            main_model,
            directories["data"],
            output_dir
        )
        logger = None
        
    else:

        # ...in any other case, train it with the pl trainer
        results, logger = train_neural_sequence_classifier(
            features_extractor,
            main_model,
            directories["data"],
            tokenizer,
            output_dir,
            **config
        )

    # Show training results
    show_results(results,is_sklearn_model,logger,**config)


if __name__ == "__main__":
    main()