import json
import os
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from collections import Counter
from tqdm import tqdm


class BOW(object):

    def __init__(self,tokenizer,column):
        self.vocab_size = len(tokenizer)
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


    def init_extractor(self):
        pass

    def fit_transform(self,dataset):
        X = self._create_sparse_matrix(dataset)
        return X

    def transform(self,dataset):
        X = self._create_sparse_matrix(dataset)
        return X

