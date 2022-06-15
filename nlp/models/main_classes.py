import json
import os
import pickle

import torch
from torch import nn
from sklearn.base import BaseEstimator

from .two_layer_net import TwoLayerNet
from sklearn.naive_bayes import MultinomialNB


_supported_models = {
    "two_layer_net": TwoLayerNet,
    "naive_bayes": MultinomialNB
}


class MainModel(object):

    def __init__(self,**params):

        # Instantiate the sklearn/torch model
        model_type = params.pop("type")
        task = params.pop("task")
        model_class = _supported_models[model_type]
        self.model = model_class(**params)

        # Keep the task and type and params
        self.task = task
        self.model_type = model_type
        self.params = params


    def save(self,output_dir):
        if isinstance(self.model,nn.Module):
            torch.save(self.model.state_dict(),os.path.join(output_dir,"state_dict.pkl"))
        elif isinstance(self.model,BaseEstimator):
            with open(os.path.join(output_dir,"state_dict.pkl"),"wb") as f:
                pickle.dump(self.model,f)
        else:
            raise ValueError("Model type not supported.")

        with open(os.path.join(output_dir,"params.json"),"w") as f:
            json.dump(dict(
                type=self.model_type,
                task=self.task,
                **self.params,
            ),f)


    @classmethod
    def load(cls,model_dir):
        with open(os.path.join(model_dir,"params.json"),"r") as f:
            params = json.load(f)

        return cls(**params)

