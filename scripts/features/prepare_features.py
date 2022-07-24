from utils import parse_args, load_tokenizer
from nlp.features import SUPPORTED_EXTRACTORS


def main():
    config, tokenizer_dir, output_dir = parse_args()

    name = config.pop("type")
    for extractor_class in SUPPORTED_EXTRACTORS:
        if extractor_class.name == name:
            break
    
    tokenizer = load_tokenizer(tokenizer_dir)
    extractor = extractor_class(tokenizer,**config)
    extractor.init_extractor()
    extractor.save(output_dir)
    

    
if __name__ == "__main__":
    main()