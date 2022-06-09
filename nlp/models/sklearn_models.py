import json
import os
import pickle
from sklearn.naive_bayes import MultinomialNB


SKLEARN_MODELS = [
    "naive_bayes"
]


def sklearn_model_initializer(model_name,**kwargs):

    if model_name == "naive_bayes":
        model = MultinomialNB(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model


def sklearn_model_saver(model_name,sklearn_model,model_dir):

    with open(os.path.join(model_dir,"state_dict.pkl"),"wb") as f:
        pickle.dump(sklearn_model,f)
    with open(os.path.join(model_dir,"params.json"),"w") as f:
        json.dump({
            "type": model_name,
            **sklearn_model.get_params()
        },f)


def sklearn_model_loader(model_dir):

    with open(os.path.join(model_dir,"state_dict.pkl"),"rb") as f:
        sklearn_model = pickle.load(f)

    return sklearn_model