from utils import parse_args, load_pretrained_tokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer


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