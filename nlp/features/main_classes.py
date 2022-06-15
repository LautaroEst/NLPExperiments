import pickle
import os
import json

import torch
from torch import nn

from .cbow import CBOW
from .bow import BOW


_supported_extractors = {
    "cbow": CBOW,
    "bow": BOW
}


class FeaturesExtractor(object):

    def __init__(self,tokenizer,**params):
        self.tokenizer = tokenizer

        extractor_type = params.pop("type")
        extractor_class = _supported_extractors[extractor_type]
        extractor = extractor_class(tokenizer,**params)

        self.extractor_type = extractor_type
        self.extractor = extractor
        self.params = params


    def init_extractor(self):
        self.extractor.init_extractor()


    def save(self,output_dir):
        if isinstance(self.extractor,nn.Module):
            torch.save(self.extractor.state_dict(),os.path.join(output_dir,"state_dict.pkl"))
        else:
            with open(os.path.join(output_dir,"state_dict.pkl"),"wb") as f:
                pickle.dump(self.extractor,f)

        with open(os.path.join(output_dir,"params.json"),"w") as f:
            json.dump(dict(
                type=self.extractor_type,
                **self.params,
            ),f)


    @classmethod
    def load(cls,tokenizer,extractor_dir):
        with open(os.path.join(extractor_dir,"params.json"),"r") as f:
            params = json.load(f)
        return cls(tokenizer,**params)

        

        