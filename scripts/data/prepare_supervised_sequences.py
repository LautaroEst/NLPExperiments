from datasets import DatasetDict
from utils import parse_args, load_tokenizer


def prepare_data_for_training(
        tokenizer,
        output_dir,
        **config
    ):

    preprocessing_function = config.pop("preprocessing_function")

    data_dict = {}
    for split in ["train", "validation"]:
        dataset = config[f"{split}_data"]
        dataset = dataset.map(
            lambda sample: preprocessing_function(sample,tokenizer),
            **config["mapping_args"]
        )
        columns_to_remove = list(set(dataset.features.keys()) - set(config["columns"]))
        dataset = dataset.remove_columns(columns_to_remove)
        data_dict[split] = dataset

    DatasetDict(data_dict).save_to_disk(output_dir)



def main():
    config, tokenizer_dir, output_dir = parse_args()
    tokenizer = load_tokenizer(tokenizer_dir)

    # Train and Validation Data:
    prepare_data_for_training(tokenizer,output_dir,**config)

    
if __name__ == "__main__":
    main()