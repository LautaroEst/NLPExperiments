import torch
from torch import nn
from .main_classes import NeuralModel


class TwoLayerNet(NeuralModel):

    name = "two_layer_net"

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


