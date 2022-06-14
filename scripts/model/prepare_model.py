import argparse
import json
import os

from utils import parse_args
from nlp.models import init_model, save_model



def main():
    config, output_dir = parse_args()
    model_name = config.pop("type")
    task = config.pop("task")
    torch_model = init_model(model_name,task,**config)
    save_model(model_name,torch_model,output_dir)
    

    
if __name__ == "__main__":
    main()