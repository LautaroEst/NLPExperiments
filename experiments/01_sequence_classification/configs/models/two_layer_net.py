import torch
from torch import nn

class TwoLayerNet(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size,hidden_size)
        self.output_layer = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x



config = {
    "model_class": TwoLayerNet,
    "input_size": 300,
    "hidden_size": 50,
    "output_size": 5
}