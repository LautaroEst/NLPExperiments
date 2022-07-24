import torch
from torch import nn
import torch.nn.functional as F
from .main_classes import NeuralMainModel


class TwoLayerNet(NeuralMainModel):

    name = "two_layer_net"

    def __init__(self,task,input_size=100,hidden_size=200,output_size=5):

        config_params = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size
        )
        super().__init__(task,**config_params)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = nn.Linear(input_size,hidden_size)

        if output_size == 2:
            self.output_layer = nn.Linear(hidden_size,1)
            self.output_activation = lambda x: F.logsigmoid(x.squeeze(dim=-1))
            self.output_discriminator = lambda log_prob: (log_prob > 0.5).type(torch.float)
        else:
            self.output_layer = nn.Linear(hidden_size,output_size)
            self.output_activation = nn.LogSoftmax(dim=-1)
            self.output_discriminator = lambda log_probs: torch.argmax(log_probs,dim=-1)

    def init_model(self):
        pass

    def forward(self,x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        log_probs = self.output_activation(x)
        y_predict = self.output_discriminator(log_probs)
        return {
            "log_probs": log_probs,
            "predictions": y_predict
        }


