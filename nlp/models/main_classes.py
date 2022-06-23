import json
import os
import pickle

import torch
from torch import nn



class BaseMainModel:

    name = None

    def init_model(self):
        raise NotImplementedError("Initialization not implemented")

    def save(self,output_dir):
        self.save_config_params(output_dir)
        self.save_state_dict(output_dir)

    def save_config_params(self,output_dir):
        with open(os.path.join(output_dir,"config_params.json"),"w") as f:
            json.dump(dict(type=self.name,task=self.task,**self.config_params),f)

    def save_state_dict(self,output_dir):
        raise NotImplementedError("save_state_dict function not implemented")

    @classmethod
    def load(cls,model_dir):
        config_params = cls.load_config_params(model_dir)
        model = cls(**config_params)
        model.load_state_dict(model_dir)
        return model

    @staticmethod
    def load_config_params(model_dir):
        with open(os.path.join(model_dir,"config_params.json"),"r") as f:
            config_params = json.load(f)
        config_params.pop("type")
        return config_params

    def load_state_dict(self,model_dir):
        raise NotImplementedError("load_state_dict function not implemented")
    
    def forward(self,data):
        raise NotImplementedError("forward function not implemented")


class NeuralMainModel(nn.Module, BaseMainModel):

    def __init__(self,task,**config_params):
        super().__init__()
        self.task = task
        self.config_params = config_params

    def save_state_dict(self,output_dir):
        torch.save(self.state_dict(),os.path.join(output_dir,"state_dict.pkl"))

    def load_state_dict(self,model_dir):
        state_dict = torch.load(os.path.join(model_dir,"state_dict.pkl"))
        super().load_state_dict(state_dict)


class GenericMLMainModel(BaseMainModel):

    def __init__(self,task,**config_params):
        self.task = task
        self.config_params = config_params

    def save_state_dict(self,output_dir):
        with open(os.path.join(output_dir,"state_dict.pkl"),"wb") as f:
            pickle.dump(self.state_dict(),f)

    def load_state_dict(self,model_dir):
        with open(os.path.join(model_dir,"state_dict.pkl"),"rb") as f:
            self._state_dict = pickle.load(f)

    def __call__(self,data):
        return self.forward(data)
       
    def state_dict(self):
        return self._state_dict


# class BaseMainModel(nn.Module):

#     def __init__(self,task,**params):

#         super().__init__()
#         self.task = task
#         self.params = params

#     def save(self,output_dir):
#         with open(os.path.join(output_dir,"config_params.json"),"w") as f:
#             json.dump(dict(
#                 type=self.name,
#                 task=self.task,
#                 **self.config_params,
#             ),f)

#     @staticmethod
#     def load_config_params(model_dir):
#         with open(os.path.join(model_dir,"config_params.json"),"r") as f:
#             config_params = json.load(f)
#         config_params.pop("name")
#         task = config_params.pop("task")
#         return config_params, task


# class NeuralModel(BaseMainModel):

#     def save(self,output_dir):
#         torch.save(self.state_dict(),os.path.join(output_dir,"state_dict.pkl"))
#         super().save(output_dir)

#     @classmethod
#     def load(cls,model_dir):
#         config_params, task = cls.load_config_params(model_dir)
#         model = cls(task,**config_params)
#         state_dict = torch.load(os.path.join(model_dir,"state_dict.pkl"))
#         model.load_state_dict(state_dict)
#         return model


# class GenericMLModel(BaseMainModel):

#     def save(self,output_dir):
#         with open(os.path.join(output_dir,"state_dict.pkl"),"wb") as f:
#             pickle.dump(self.state_dict,f)
#         super().save(output_dir)

#     @classmethod
#     def load(cls,model_dir):
#         config_params, task = cls.load_config_params(model_dir)
#         model = cls(task,**config_params)
#         with open(os.path.join(model_dir,"state_dict.pkl"),"rb") as f:
#             model.state_dict = pickle.load(f)
#         return model

