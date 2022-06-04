import argparse
from import_config import import_configs_objs
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer


def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--config",help="Python file with the tokenizer loading configuration")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Extract config and output directory
    output_dir = args["out"]
    config = import_configs_objs(args["config"])

    return config, output_dir


def load_pretrained_tokenizer(**config):
    tokenizer = AutoTokenizer.from_pretrained(**config)
    return tokenizer


def train_tokenizer_from_corpus(**config):
    model = config.pop("model")
    normalizer = config.pop("normalizer")
    pre_tokenizer = config.pop("pre_tokenizer")
    trainer = config.pop("trainer")
    corpus = config.pop("corpus")
    encoding_args = config.pop("encoding_args")

    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.train_from_iterator(corpus,trainer=trainer,length=len(corpus))

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,**encoding_args)

    return tokenizer

    




def main():
    config, output_dir = parse_args()

    is_pretrained = config.pop("is_pretrained")
    if is_pretrained:
        tokenizer = load_pretrained_tokenizer(**config)
    else:
        tokenizer = train_tokenizer_from_corpus(**config)

    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()