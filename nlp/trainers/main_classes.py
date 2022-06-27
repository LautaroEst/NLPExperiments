import os
import numpy as np

import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score

from .lightning_models import ClassificationModel, RegressionModel, TBLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class SupervisedModelTrainer:

    def init_extractor_main_model_config_params(self,extractor,main_model,**config_params):
        self.config_params = config_params
        self._extractor = extractor
        self._main_model = main_model



class SupervisedNeuralModelTrainer(pl.Trainer, SupervisedModelTrainer):

    def __init__(self,extractor,main_model,**config_params):
        super().init_extractor_main_model_config_params(extractor,main_model,**config_params)

        # Define the model from the task
        if main_model.task == "classification":
            self.full_model = ClassificationModel(extractor,main_model,**config_params)
        elif main_model.task == "regression":
            self.full_model = RegressionModel(extractor,main_model,**config_params)
        else:
            raise ValueError(f"Task {main_model.task} not supported")

        # Init the logger and checkpoint callback
        output_dir = config_params["output_dir"]
        logger = TBLogger(save_dir=output_dir,name="",default_hp_metric=False)
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_dir,"version_0/checkpoints"),
            filename="step={step}-epoch={epoch}-val_loss={loss/validation:.2f}",
            monitor="loss/validation",
            save_top_k=1,
            save_weights_only=False,
            auto_insert_metric_name=False
        )

        super().__init__(
            default_root_dir=output_dir,
            logger=logger,
            callbacks=[checkpoint_callback],
            deterministic=True,
            enable_checkpointing=config_params["enable_checkpointing"],
            min_epochs=config_params["min_epochs"],
            max_epochs=config_params["max_epochs"],
            accelerator=config_params["accelerator"],
            devices=config_params["devices"],
            val_check_interval=config_params["val_check_interval"],
            log_every_n_steps=config_params["log_every_n_steps"]
        )


    def fit(self,data_module):
        # Fit the model
        super().fit(self.full_model,data_module["train"],data_module["validation"])


    def validate(self,data_module):
        # Search best model
        output_dir = self.config_params["output_dir"]
        best_model_name = os.path.join(output_dir,"version_0/checkpoints/",sorted(
            os.listdir(os.path.join(output_dir,"version_0/checkpoints/")),
            key=lambda ckp: float(ckp.split("val_loss=")[1].split(".ckpt")[0]),
            reverse=True
        )[0])

        def parse_results(results):
            return {f"{key.split('/')[1]}/{key.split('/')[0]}": val for key, val in results[0].items()}

        train_results = super().validate(self.full_model,data_module["train"],best_model_name,verbose=True)
        val_results = super().validate(self.full_model,data_module["validation"],best_model_name,verbose=True)
        results = {**parse_results(train_results), **parse_results(val_results)}
        return results


## TO DO:
# EMPAQUETAR TODOS LOS DATOS EN UN DATA_MODULE QUE PERMITA RECIBIR Y MANDAR DATOS
# DEL EXTRACTOR AL MODELO


    
class SupervisedGenericMLModelTrainer(SupervisedModelTrainer):

    def __init__(self,extractor,main_model,**config_params):
        super().init_extractor_main_model_config_params(extractor,main_model,**config_params)


    def fit(self,data_module):
        vectorized_dataset = {
            "X": self._extractor(data_module["train"]),
            "y": np.array(data_module["train"]["label"])
        }
        self._main_model.fit(
            vectorized_dataset["X"],
            vectorized_dataset["y"]
        )

    def validate(self,data_module):
        vectorized_dataset = {}
        for split in ["train", "validation"]:
            vectorized_dataset[split] = {
                "X": self._extractor(data_module[split]),
                "y": np.array(data_module[split]["label"])
            }
        
        y_preds = {
            split: self._main_model(vectorized_dataset[split]["X"])["predictions"] \
            for split in ["train", "validation"]
        }

        results = {}
        for split in ["train", "validation"]:
            results[f"f1-score/{split}"] = f1_score(vectorized_dataset[split]["y"],y_preds[split],average="macro")
            results[f"accuracy/{split}"] = accuracy_score(vectorized_dataset[split]["y"],y_preds[split])
        
        return results
    


