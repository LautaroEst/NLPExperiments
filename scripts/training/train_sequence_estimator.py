from utils import load_extractor, load_main_model, load_tokenizer, parse_args


def prepare_data_for_training(tokenizer,data_dir,**config):
    dataset = None
    return dataset


def train_model(features_extractor,main_model,dataset,output_dir,**config):
    results = None
    return results


def show_results(results,output_dir,**config):
    pass


def main():
    # Argument parser:
    config, directories, output_dir = parse_args()

    # Load the pretrained tokenizer from local directory
    tokenizer = load_tokenizer(directories["tokenizer"])

    # Load the initialized extractor from local_directory
    features_extractor = load_extractor(tokenizer,directories["features"])

    # Load the initialized model from local_directory
    main_model = load_main_model(directories["model"])

    # Prepare data for training
    dataset = prepare_data_for_training(tokenizer,directories["data"],**config)

    # Train model
    results = train_model(features_extractor,main_model,dataset,output_dir,**config)

    # Show results
    show_results(results,output_dir,**config)





if __name__ == "__main__":
    main()