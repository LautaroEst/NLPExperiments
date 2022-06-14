from .two_layer_net import *
from .sklearn_models import *

_initializers_functions = {
    "two_layer_net": two_layer_net_initializer,
    "naive_bayes": lambda task, **kwargs: sklearn_model_initializer("naive_bayes",**kwargs)
}

_savers_functions = {
    "two_layer_net": two_layer_net_saver,
    "naive_bayes": lambda model, model_dir : sklearn_model_saver("naive_bayes",model,model_dir)
}

_loaders_functions = {
    "two_layer_net": two_layer_net_loader,
    "naive_bayes": sklearn_model_loader
}

def init_model(model_name,task,**kwargs):
    try:
        initializer_fn = _initializers_functions[model_name]
    except KeyError:
        raise ValueError(f"Model with name {model_name} not supported.")
    
    model = initializer_fn(task,**kwargs)
    return model

def save_model(model_name,torch_model,model_dir):
    try:
        saver_fn = _savers_functions[model_name]
    except KeyError:
        raise ValueError(f"Model with name {model_name} not supported.")
    
    saver_fn(torch_model,model_dir)

def load_model(model_name,model_dir):
    try:
        loader_fn = _loaders_functions[model_name]
    except KeyError:
        raise ValueError(f"Model with name {model_name} not supported.")
    
    torch_model = loader_fn(model_dir)
    return torch_model