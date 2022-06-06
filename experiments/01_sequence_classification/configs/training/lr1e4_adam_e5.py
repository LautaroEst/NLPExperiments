import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl

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
        y_hat = torch.argmax(F.log_softmax(scores),dim=-1)
        return y_hat

    def training_step(self,batch,batch_idx):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        loss = F.cross_entropy(scores,batch["label"])
        self.log("train_loss",loss)
        return loss

    def validation_step(self,batch,batch_idx):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        loss = F.cross_entropy(scores,batch["label"])
        self.log("val_loss",loss)
        return loss

    def backward(self,loss,optimizer,optimizer_idx):
        loss.backward()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def optimizer_step(self,epoch,batch_idx,optimizer,optimizer_idx):
        optimizer.step()


config = {

    "dataloader_args": {
        "train_batch_size": 32,
        "validation_batch_size": 64,
        "data_collator": data_collate
    },
    "full_model_class": FullModel,
    "trainer_args": dict(
        max_epochs=3
    )
}