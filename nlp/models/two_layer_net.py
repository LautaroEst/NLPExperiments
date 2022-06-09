import json
import os
import torch
from torch import nn

class TwoLayerNet(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = nn.Linear(input_size,hidden_size)
        self.output_layer = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x


def two_layer_net_initializer(input_size,hidden_size,output_size):
    model = TwoLayerNet(input_size,hidden_size,output_size)
    return model


def two_layer_net_saver(torch_model,model_dir):
    torch.save(torch_model.state_dict(),os.path.join(model_dir,"state_dict.pkl"))
    with open(os.path.join(model_dir,"params.json"),"w") as f:
        json.dump({
            "type": "two_layer_net",
            "input_size": torch_model.input_size,
            "hidden_size": torch_model.hidden_size,
            "output_size": torch_model.output_size
        },f)


def two_layer_net_loader(model_dir):
    with open(os.path.join(model_dir,"params.json"),"r") as f:
        params = json.load(f)

    params.pop("type")
    torch_model = TwoLayerNet(**params)
    state_dict = torch.load(os.path.join(model_dir,"state_dict.pkl"))

    torch_model.load_state_dict(state_dict)
    return torch_model