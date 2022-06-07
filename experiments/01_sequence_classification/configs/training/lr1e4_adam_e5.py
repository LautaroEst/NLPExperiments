import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score


def data_collate(batch,tokenizer):
    return tokenizer.pad(
        batch,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length
    )



class FullModel(pl.LightningModule):

    def __init__(
        self,
        features_extractor,
        main_model,
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.main_model = main_model

    def forward(self,batch):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        y_hat = torch.argmax(F.log_softmax(scores,dim=-1),dim=-1)
        return y_hat

    def training_step(self,batch,batch_idx):
        # print(batch_idx,len(batch["input_ids"]))
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        y_hat = torch.argmax(F.log_softmax(scores,dim=-1),dim=-1).cpu().detach().numpy()
        y_true = batch["label"].cpu().detach().numpy()
        loss = F.cross_entropy(scores,batch["label"])
        self.log("train_loss",loss)
        self.log("train_performance",{
            "accuracy": accuracy_score(y_true,y_hat),
            "f1": f1_score(y_true,y_hat,average="macro")
        })
        return loss

    def validation_step(self,batch,batch_idx):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        y_hat = torch.argmax(F.log_softmax(scores,dim=-1),dim=-1).cpu().detach().numpy()
        y_true = batch["label"].cpu().detach().numpy()
        loss = F.cross_entropy(scores,batch["label"])
        self.log("val_loss",loss)
        self.log("val_performance",{
            "accuracy": accuracy_score(y_true,y_hat),
            "f1": f1_score(y_true,y_hat,average="macro")
        })
        return loss

    def backward(self,loss,optimizer,optimizer_idx):
        loss.backward()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

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
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


config = {


    "dataloader_args": {
        "train_batch_size": 32,
        "validation_batch_size": 64,
        "data_collator": data_collate,
        "num_workers": 8
    },

    "full_model_class": FullModel,
    "trainer_args": dict(
        deterministic=True,
        min_epochs=1,
        max_epochs=3,
        val_check_interval=200,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=True
    )
}