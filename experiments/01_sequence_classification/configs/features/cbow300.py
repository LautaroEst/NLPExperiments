import torch
from torch import nn


class CBOW(nn.Module):

    def __init__(self,tokenizer,embeddings_dim=300,pretrained_file=None,freeze_parameters=True):
        super().__init__()
        embeddings = nn.Embedding(len(tokenizer),embeddings_dim,tokenizer.pad_token_id)
        self.embeddings = self._init_embeddings_from_file(tokenizer,pretrained_file,embeddings)

        for param in self.embeddings.parameters():
            param.requires_grad = not freeze_parameters
        
    @staticmethod
    def _init_embeddings_from_file(tokenizer,pretrained_file,embeddings):
        if pretrained_file is None:
            return embeddings
        
        ## TO DO: Support pretrained embeddings
        return embeddings

    def forward(self,batch):
        cbow = self.embeddings(batch["input_ids"]).mean(axis=1) # Batch first
        return cbow


config = {

    "features_extractor_class": CBOW,

    "embeddings_dim": 300,
    "pretrained_file": None,
    "freeze_parameters": False

}
