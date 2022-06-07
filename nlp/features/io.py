from .cbow import *

_initializers_functions = {
    "cbow": cbow_initializer
}

_savers_functions = {
    "cbow": cbow_saver
}

_loaders_functions = {
    "cbow": cbow_loader
}

def init_features_extractor(model,**kwargs):
    try:
        initializer_fn = _initializers_functions[model]
    except KeyError:
        raise ValueError(f"Features extractor with name {model} not supported.")
    
    extractor = initializer_fn(**kwargs)
    return extractor

def save_features_extractor(model,extractor,features_dir):
    try:
        saver_fn = _savers_functions[model]
    except KeyError:
        raise ValueError(f"Features extractor with name {model} not supported.")
    
    saver_fn(extractor,features_dir)

def load_features_extractor(model,features_dir):
    try:
        loader_fn = _loaders_functions[model]
    except KeyError:
        raise ValueError(f"Features extractor with name {model} not supported.")
    
    extractor = loader_fn(features_dir)
    return extractor

