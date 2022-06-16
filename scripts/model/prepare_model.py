from utils import parse_args
from nlp.models import MainModel


def main():
    config, output_dir = parse_args()
    main_model = MainModel(**config)
    main_model.save(output_dir)


if __name__ == "__main__":
    main()