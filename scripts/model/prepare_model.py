from utils import parse_args
from nlp.models import SUPPORTED_MODELS


def main():
    config, output_dir = parse_args()

    model_type = config.pop("type")
    for model_class in SUPPORTED_MODELS:
        if model_class.name == model_type:
            break

    task = config.pop("task")
    main_model = model_class(task,**config)
    main_model.init_model()
    main_model.save(output_dir)


if __name__ == "__main__":
    main()