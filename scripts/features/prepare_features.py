from utils import parse_args, load_tokenizer
from nlp.features import FeaturesExtractor


def main():
    config, tokenizer_dir, output_dir = parse_args()
    tokenizer = load_tokenizer(tokenizer_dir)
    extractor = FeaturesExtractor(tokenizer,**config)
    extractor.init_extractor()
    extractor.save(output_dir)
    

    
if __name__ == "__main__":
    main()