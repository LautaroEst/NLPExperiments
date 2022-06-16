import pickle
import os
import json

import torch
from torch import nn


class BaseFeaturesExtractor(nn.Module):

    def __init__(self,tokenizer,**params):

        super().__init__()
        self.tokenizer = tokenizer
        self.params = params
        
    def init_extractor(self):
        raise NotImplementedError("Initialization not implemented")

    def save(self,output_dir):
        with open(os.path.join(output_dir,"params.json"),"w") as f:
            json.dump(dict(
                type=self.name,
                **self.params,
            ),f)



class NeuralFeatureExtractor(BaseFeaturesExtractor):

    def save(self,output_dir):
        torch.save(self.state_dict(),os.path.join(output_dir,"state_dict.pkl"))
        super().save(output_dir)

    @classmethod
    def load(cls,tokenizer,extractor_dir):
        with open(os.path.join(extractor_dir,"params.json"),"r") as f:
            params = json.load(f)
        params.pop("name")
        extractor = cls(tokenizer,**params)
        state_dict = torch.load(os.path.join(extractor_dir,"state_dict.pkl"))
        extractor.load_state_dict(state_dict)
        return extractor


class GenericMLFeatureExtractor(BaseFeaturesExtractor):

    def save(self,output_dir):
        with open(os.path.join(output_dir,"state_dict.pkl"),"wb") as f:
            pickle.dump(self.sklearn_parameters,f)
        super().save(output_dir)

    @classmethod
    def load(cls,tokenizer,extractor_dir):
        with open(os.path.join(extractor_dir,"params.json"),"r") as f:
            params = json.load(f)
            params.pop("name")
        extractor = cls(tokenizer,**params)
        with open(os.path.join(extractor_dir,"state_dict.pkl"),"rb") as f:
            extractor.sklearn_parameters = pickle.load(f)
        return extractor
        