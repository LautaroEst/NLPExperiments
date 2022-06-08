import argparse
import json
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from import_config import import_configs_objs
from datasets import DatasetDict

from nlp.features import load_features_extractor
from nlp.models import load_model
from sklearn.metrics import f1_score, accuracy_score



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
        model_type = json.load(f)["type"]
    model = load_model(model_type,model_dir)
    return model


def load_data(
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




class FullModel(pl.LightningModule):

    def __init__(
        self,
        features_extractor,
        main_model,
        configure_optimizers,
        optimizer_step
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.main_model = main_model
        self.__configure_optimizers = configure_optimizers
        self.__optimizer_step = optimizer_step

    def forward(self,batch):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        y_hat = torch.argmax(F.log_softmax(scores,dim=-1),dim=-1)
        return y_hat

    def training_step(self,batch,batch_idx):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        y_hat = torch.argmax(F.log_softmax(scores,dim=-1),dim=-1).cpu().detach().numpy()
        y_true = batch["label"].cpu().detach().numpy()
        loss = F.cross_entropy(scores,batch["label"])
        self.log_dict({
            "loss/train": loss.item(),
            "accuracy/train": accuracy_score(y_true,y_hat),
            "f1-score/train": f1_score(y_true,y_hat,average="macro")
        },logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        y_hat = torch.argmax(F.log_softmax(scores,dim=-1),dim=-1)
        y_true = batch["label"]
        loss = F.cross_entropy(scores,y_true,reduction="sum")
        return {
            "num_samples": len(y_true),
            "loss": loss.item(),
            "y_hat": y_hat,
            "y_true": y_true
        }

    def validation_epoch_end(self, outputs):
        num_samples = sum([output["num_samples"] for output in outputs])
        avg_loss = sum([output["loss"] for output in outputs]) / num_samples
        y_hat = torch.hstack([output["y_hat"] for output in outputs]).cpu().detach().numpy()
        y_true = torch.hstack([output["y_true"] for output in outputs]).cpu().detach().numpy()
        self.log_dict({
            "loss/validation": avg_loss,
            "accuracy/validation": accuracy_score(y_true,y_hat),
            "f1-score/validation": f1_score(y_true,y_hat,average="macro")
        },logger=True)

    def backward(self,loss,optimizer,optimizer_idx):
        loss.backward()

    def configure_optimizers(self):
        return self.__configure_optimizers(self)

    def optimizer_step(
        self,
        epoch=None,
        batch_idx=None,
        optimizer=None,
        optimizer_idx=None,
        optimizer_closure=None,
        on_tpu=None,
        using_native_amp=None,
        using_lbfgs=None
    ):
        return self.__optimizer_step(epoch,batch_idx,optimizer,optimizer_idx,optimizer_closure,on_tpu,using_native_amp,using_lbfgs)


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)


def train_sequence_classifier(
        features_extractor,main_model,train_dataloader,val_dataloader,output_dir,**config
    ):

    configure_optimizers=config.pop("configure_optimizers")
    optimizer_step=config.pop("optimizer_step")
    model = FullModel(features_extractor,main_model,configure_optimizers,optimizer_step)

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
    trainer.fit(model,train_dataloader,val_dataloader)


    best_model_name = os.path.join(output_dir,"version_0/checkpoints/",sorted(
        os.listdir(os.path.join(output_dir,"version_0/checkpoints/")),
        key=lambda ckp: float(ckp.split("val_loss=")[1].split(".ckpt")[0]),
        reverse=True
    )[0])
    trainer.validate(model,val_dataloader,best_model_name,verbose=True)

    return model, trainer, logger

# def eval_model(output_dir,features_extractor,main_model,val_dataloader,**config):
#     best_model_name = sorted(
#         os.listdir(os.path.join(output_dir,"version_0/checkpoints/")),
#         key=lambda ckp: float(ckp.split("val_loss=")[1].split(".ckpt")[0]),
#         reverse=True
#     )[0]

#     configure_optimizers=config.pop("configure_optimizers")
#     optimizer_step=config.pop("optimizer_step")
#     best_model = FullModel.load_from_checkpoint(
#         os.path.join(output_dir,"version_0/checkpoints/",best_model_name),
#         features_extractor=features_extractor,
#         main_model=main_model,
#         configure_optimizers=configure_optimizers,
#         optimizer_step=optimizer_step
#     )
#     best_model.eval()
    





def main():
    config, directories, output_dir = parse_args()
    tokenizer = load_tokenizer(directories["tokenizer"])
    features_extractor = load_extractor(directories["features"])
    main_model = load_main_model(directories["model"])

    train_batch_size=config.pop("train_batch_size")
    validation_batch_size=config.pop("validation_batch_size")
    padding_strategy=config.pop("padding_strategy")
    num_workers=config.pop("num_workers")
    dataloaders = load_data(
        directories["data"],
        tokenizer,
        train_batch_size,
        validation_batch_size,
        padding_strategy,
        num_workers
    )

    model, trainer, logger = train_sequence_classifier(
        features_extractor,
        main_model,
        dataloaders["train"],
        dataloaders["validation"],
        output_dir,
        **config
    )

    # eval_model(output_dir,features_extractor,main_model,dataloaders["validation"],**config)

    # logger.experiment.add_hparams(
    #     dict(
    #         train_batch_size=train_batch_size,
    #         validation_batch_size=validation_batch_size,
    #         min_epochs=config["min_epochs"],
    #         max_epochs=config["max_epochs"],
    #         val_check_interval=config["val_check_interval"]
    #     )
    # )


if __name__ == "__main__":
    main()