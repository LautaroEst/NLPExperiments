import pickle
import os
import json

import torch
from torch import nn


class BaseFeaturesExtractor:

    def __init__(self,tokenizer,**config_params):
        self.tokenizer = tokenizer
        self.config_params = config_params
        
    def init_extractor(self):
        raise NotImplementedError("Initialization not implemented")

    def save(self,output_dir):
        with open(os.path.join(output_dir,"config_params.json"),"w") as f:
            json.dump(dict(
                type=self.name,
                **self.config_params,
            ),f)

    def transform(self,data):
        raise NotImplementedError("transform function not implemented")


class NeuralFeatureExtractor(nn.Module, BaseFeaturesExtractor):

    def __init__(self,tokenizer,**config_params):
        nn.Module.__init__(self)
        BaseFeaturesExtractor.__init__(self,tokenizer,**config_params)

    def save(self,output_dir):
        torch.save(self.state_dict(),os.path.join(output_dir,"state_dict.pkl"))
        super().save(output_dir)

    @classmethod
    def load(cls,tokenizer,extractor_dir):
        with open(os.path.join(extractor_dir,"config_params.json"),"r") as f:
            config_params = json.load(f)
        config_params.pop("name")
        extractor = cls(tokenizer,**config_params)
        state_dict = torch.load(os.path.join(extractor_dir,"state_dict.pkl"))
        extractor.load_state_dict(state_dict)
        return extractor

    def forward(self,data):
        return self.transform(data)




class GenericMLFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self,tokenizer,**config_params):
        super().__init__(tokenizer,**config_params)

    def save(self,output_dir):
        with open(os.path.join(output_dir,"state_dict.pkl"),"wb") as f:
            pickle.dump(self.sklearn_parameters,f)
        super().save(output_dir)

    @classmethod
    def load(cls,tokenizer,extractor_dir):
        with open(os.path.join(extractor_dir,"config_params.json"),"r") as f:
            config_params = json.load(f)
            config_params.pop("name")
        extractor = cls(tokenizer,**config_params)
        with open(os.path.join(extractor_dir,"state_dict.pkl"),"rb") as f:
            extractor.sklearn_parameters = pickle.load(f)
        return extractor

    def __call__(self,data):
        return self.transform(data)
        


# if __name__ == "__main__":


#     class First:
#         def __init__(self,num):
#             self.num_first = num
#             print(f"first-{num}")
        
#     class Second:
#         def __init__(self,num):
#             self.num_second = num
#             print(f"second-{num}")

#     class Third(First,Second):
#         def __init__(self):
#             First.__init__(self,"1")
#             Second.__init__(self,"2")
#             print("third")

#     third = Third()
#     print(third.num_first,third.num_second)