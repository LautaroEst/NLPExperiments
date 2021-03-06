
from torch import nn
from .main_classes import NeuralFeatureExtractor


class CBOW(NeuralFeatureExtractor):

    name = "cbow"

    def __init__(self,tokenizer,embedding_dim=300,pretrained_file=None,freeze_parameters=True):
        config_params = dict(
            embedding_dim=embedding_dim,
            pretrained_file=pretrained_file,
            freeze_parameters=freeze_parameters
        )
        super().__init__(tokenizer,**config_params)
        self.embeddings = nn.Embedding(len(tokenizer),embedding_dim,tokenizer.pad_token_id)
        self.pretrained_file = pretrained_file
        self.freeze_parameters = freeze_parameters

        for param in self.embeddings.parameters():
            param.requires_grad = not freeze_parameters

    def init_extractor(self):
        if self.pretrained_file is not None:
            ## TO DO: Support pretrained embeddings
            self.embeddings = self.embeddings

    def transform(self,batch):
        cbow = self.embeddings(batch["input_ids"]).mean(axis=1) # Batch first
        return cbow
    