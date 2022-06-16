# import json
# import os
# import torch
from torch import nn

class CBOW(nn.Module):

    def __init__(self,tokenizer,embedding_dim=300,pretrained_file=None,freeze_parameters=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.embeddings = nn.Embedding(len(tokenizer),embedding_dim,tokenizer.pad_token_id)
        self.pretrained_file = pretrained_file
        self.freeze_parameters = freeze_parameters

        for param in self.embeddings.parameters():
            param.requires_grad = not freeze_parameters

    def init_extractor(self):
        if self.pretrained_file is not None:
            ## TO DO: Support pretrained embeddings
            self.embeddings = self.embeddings

    def forward(self,batch):
        cbow = self.embeddings(batch["input_ids"]).mean(axis=1) # Batch first
        return cbow

