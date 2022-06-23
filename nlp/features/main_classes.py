import pickle
import os
import json

import torch
from torch import nn


class BaseFeaturesExtractor:

    name = None

    def init_extractor(self):
        raise NotImplementedError("Initialization not implemented")

    def save(self,output_dir):
        self.save_config_params(output_dir)
        self.save_state_dict(output_dir)

    def save_config_params(self,output_dir):
        with open(os.path.join(output_dir,"config_params.json"),"w") as f:
            json.dump(dict(type=self.name,**self.config_params),f)

    def save_state_dict(self,output_dir):
        raise NotImplementedError("save_state_dict function not implemented")

    @classmethod
    def load(cls,tokenizer,extractor_dir):
        config_params = cls.load_config_params(extractor_dir)
        extractor = cls(tokenizer,**config_params)
        extractor.load_state_dict(extractor_dir)
        return extractor

    @staticmethod
    def load_config_params(extractor_dir):
        with open(os.path.join(extractor_dir,"config_params.json"),"r") as f:
            config_params = json.load(f)
        config_params.pop("type")
        return config_params

    def load_state_dict(self,extractor_dir):
        raise NotImplementedError("load_state_dict function not implemented")
    
    def transform(self,data):
        raise NotImplementedError("transform function not implemented")


class NeuralFeatureExtractor(nn.Module, BaseFeaturesExtractor):

    def __init__(self,tokenizer,**config_params):
        super().__init__()
        self.tokenizer = tokenizer
        self.config_params = config_params

    def save_state_dict(self,output_dir):
        torch.save(self.state_dict(),os.path.join(output_dir,"state_dict.pkl"))

    def load_state_dict(self,extractor_dir):
        state_dict = torch.load(os.path.join(extractor_dir,"state_dict.pkl"))
        super().load_state_dict(state_dict)

    def forward(self,data):
        return self.transform(data)


class GenericMLFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self,tokenizer,**config_params):
        self.tokenizer = tokenizer
        self.config_params = config_params

    def save_state_dict(self,output_dir):
        with open(os.path.join(output_dir,"state_dict.pkl"),"wb") as f:
            pickle.dump(self.state_dict(),f)

    def load_state_dict(self,extractor_dir):
        with open(os.path.join(extractor_dir,"state_dict.pkl"),"rb") as f:
            self._state_dict = pickle.load(f)

    def __call__(self,data):
        return self.transform(data)
       
    def state_dict(self):
        return self._state_dict
