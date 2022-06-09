import json
import os
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from collections import Counter
from tqdm import tqdm





class TeamingFeatures(object):

    def __init__(self,vocab_size,column):
        self.vocab_size = vocab_size
        self.column = column

    def _create_sparse_matrix(self,dataset):
        j_indices = []
        values = []
        indptr = [0]

        for indices in tqdm(dataset,total=len(dataset)):
            feature_counter = Counter(indices[self.column])
            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))
        
        j_indices = np.asarray(j_indices, dtype=int)
        indptr = np.asarray(indptr, dtype=int)
        values = np.asarray(values, dtype=int)

        X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, self.vocab_size),
            dtype=float,
        )
        X.sort_indices()
        return X


    def fit_transform(self,dataset):
        X = self._create_sparse_matrix(dataset)
        return X

    def transform(self,dataset):
        X = self._create_sparse_matrix(dataset)
        return X


def teaming_features_initializer(tokenizer,column):
    vocab_size = len(tokenizer)
    extractor = TeamingFeatures(vocab_size,column)
    return extractor


def teaming_features_saver(extractor,features_dir):
    torch.save({},os.path.join(features_dir,"state_dict.pkl"))
    with open(os.path.join(features_dir,"params.json"),"w") as f:
        json.dump({
            "type": "teaming_features",
            "vocab_size": extractor.vocab_size,
            "column": extractor.column
        },f)


def teaming_features_loader(features_dir):
    with open(os.path.join(features_dir,"params.json"),"r") as f:
        params = json.load(f)

    params.pop("type")
    extractor = TeamingFeatures(**params)
    # state_dict = torch.load(os.path.join(features_dir,"state_dict.pkl"))
    # extractor.load_state_dict(state_dict)
    return extractor