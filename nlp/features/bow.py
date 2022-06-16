
import numpy as np
import scipy.sparse as sp
from collections import Counter
from tqdm import tqdm
from torch import nn
from .main_classes import GenericMLFeatureExtractor


class BOW(GenericMLFeatureExtractor):

    name = "bag_of_words"
    is_neural_network = False

    def __init__(self,tokenizer,column=None):
        params = dict(
            column=column
        )
        super().__init__(tokenizer,**params)
        self.vocab_size = len(tokenizer)
        self.sklearn_parameters = {}

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

    def forward(self,dataset):
        X = self._create_sparse_matrix(dataset)
        return X