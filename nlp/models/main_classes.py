import json
import os
import pickle

import torch
from torch import nn


class BaseMainModel(nn.Module):

    def __init__(self,task,**params):

        super().__init__()
        self.task = task
        self.params = params

    def save(self,output_dir):
        with open(os.path.join(output_dir,"params.json"),"w") as f:
            json.dump(dict(
                type=self.name,
                task=self.task,
                **self.params,
            ),f)


class NeuralModel(BaseMainModel):

    def save(self,output_dir):
        torch.save(self.state_dict(),os.path.join(output_dir,"state_dict.pkl"))
        super().save(output_dir)

    @classmethod
    def load(cls,model_dir):
        with open(os.path.join(model_dir,"params.json"),"r") as f:
            params = json.load(f)
        params.pop("name")
        task = params.pop("task")
        model = cls(task,**params)
        state_dict = torch.load(os.path.join(model_dir,"state_dict.pkl"))
        model.load_state_dict(state_dict)
        return model


class GenericMLModel(BaseMainModel):

    def save(self,output_dir):
        with open(os.path.join(output_dir,"state_dict.pkl"),"wb") as f:
            pickle.dump(self.sklearn_parameters,f)
        super().save(output_dir)

    @classmethod
    def load(cls,model_dir):
        with open(os.path.join(model_dir,"params.json"),"r") as f:
            params = json.load(f)
            params.pop("name")
            task = params.pop("task")
        model = cls(task,**params)
        with open(os.path.join(model_dir,"state_dict.pkl"),"rb") as f:
            model.sklearn_parameters = pickle.load(f)
        return model

