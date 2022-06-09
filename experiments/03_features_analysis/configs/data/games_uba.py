from datasets import load_dataset



def preprocess_dataset(sample,tokenizer):
    encoded_input = tokenizer.tokenize(
        sample["text"],
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return {
        "text": encoded_input["text"],
        "score": sample["score"]
    }


def load_games_uba_dataset():
    dataset = None
    return dataset


dataset = load_games_uba_dataset()

config = {

    "train_data": dataset["train"],
    "validation_data": dataset["validation"],
    "mapping_args": {
        "batched": True
    },
    "preprocessing_function": preprocess_dataset,
    "columns": ["text","score"]
}