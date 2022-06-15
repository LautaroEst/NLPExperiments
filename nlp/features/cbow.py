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


# class CBOW(nn.Module):

#     def __init__(self,embeddings,freeze_parameters=True):
#         super().__init__()
#         self.num_embeddings = embeddings.num_embeddings
#         self.embedding_dim = embeddings.embedding_dim
#         self.padding_idx = embeddings.padding_idx
#         self.freeze_parameters = freeze_parameters

#         self.embeddings = embeddings

#         for param in self.embeddings.parameters():
#             param.requires_grad = not freeze_parameters

#     def forward(self,batch):
#         cbow = self.embeddings(batch["input_ids"]).mean(axis=1) # Batch first
#         return cbow


# def _init_embeddings_from_file(tokenizer,pretrained_file,embeddings):
#     if pretrained_file is None:
#         return embeddings
    
#     ## TO DO: Support pretrained embeddings
#     return embeddings


# def cbow_initializer(tokenizer,embedding_dim=300,pretrained_file=None,freeze_parameters=True):

#     num_embeddings = len(tokenizer)
#     padding_idx = tokenizer.pad_token_id
#     embeddings = nn.Embedding(num_embeddings,embedding_dim,padding_idx)
#     embeddings = _init_embeddings_from_file(tokenizer,pretrained_file,embeddings)

#     extractor = CBOW(embeddings,freeze_parameters)
#     return extractor


# def cbow_saver(extractor,features_dir):
#     torch.save(extractor.state_dict(),os.path.join(features_dir,"state_dict.pkl"))
#     with open(os.path.join(features_dir,"params.json"),"w") as f:
#         json.dump({
#             "type": "cbow",
#             "num_embeddings": extractor.num_embeddings,
#             "embedding_dim": extractor.embedding_dim,
#             "padding_idx": extractor.padding_idx,
#             "freeze_parameters": extractor.freeze_parameters
#         },f)


# def cbow_loader(features_dir):
#     with open(os.path.join(features_dir,"params.json"),"r") as f:
#         params = json.load(f)

#     params.pop("type")
#     freeze_parameters = params.pop("freeze_parameters")
#     embeddings = nn.Embedding(**params)
#     extractor = CBOW(embeddings,freeze_parameters)
#     state_dict = torch.load(os.path.join(features_dir,"state_dict.pkl"))

#     extractor.load_state_dict(state_dict)
#     return extractor