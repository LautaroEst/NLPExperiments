
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)


class SequenceEstimatorModel(pl.LightningModule):

    def __init__(   
        self,
        features_extractor,
        main_model,
        configure_optimizers,
        optimizer_step
    ):
        super().__init__()
        self.features_extractor = features_extractor.extractor
        self.main_model = main_model.model
        self.__configure_optimizers = configure_optimizers
        self.__optimizer_step = optimizer_step

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



class ClassificationModel(SequenceEstimatorModel):

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


class RegressionModel(SequenceEstimatorModel):

    def forward(self,batch):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        y_hat = torch.argmax(F.log_softmax(scores,dim=-1),dim=-1)
        return y_hat

    def training_step(self,batch,batch_idx):
        features = self.features_extractor(batch)
        score = self.main_model(features).squeeze(dim=-1)
        # y_hat = torch.argmax(F.log_softmax(scores,dim=-1),dim=-1).cpu().detach().numpy()
        # y_true = batch["label"].cpu().detach().numpy()
        # loss = F.cross_entropy(scores,batch["label"])
        loss = F.l1_loss(score,batch["label"])
        self.log_dict({
            "loss/train": loss.item(),
            # "accuracy/train": accuracy_score(y_true,y_hat),
            # "f1-score/train": f1_score(y_true,y_hat,average="macro")
        },logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        features = self.features_extractor(batch)
        score = self.main_model(features).squeeze(dim=-1)
        # y_hat = torch.argmax(F.log_softmax(scores,dim=-1),dim=-1)
        y_true = batch["label"]
        # loss = F.cross_entropy(scores,y_true,reduction="sum")
        loss = F.l1_loss(score,y_true,reduction="sum")
        return {
            "num_samples": len(y_true),
            "loss": loss.item(),
            # "y_hat": y_hat,
            # "y_true": y_true
        }

    def validation_epoch_end(self, outputs):
        num_samples = sum([output["num_samples"] for output in outputs])
        avg_loss = sum([output["loss"] for output in outputs]) / num_samples
        # y_hat = torch.hstack([output["y_hat"] for output in outputs]).cpu().detach().numpy()
        # y_true = torch.hstack([output["y_true"] for output in outputs]).cpu().detach().numpy()
        self.log_dict({
            "loss/validation": avg_loss,
            # "accuracy/validation": accuracy_score(y_true,y_hat),
            # "f1-score/validation": f1_score(y_true,y_hat,average="macro")
        },logger=True)