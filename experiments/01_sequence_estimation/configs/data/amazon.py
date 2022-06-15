from datasets import load_dataset



def preprocess_dataset(sample,tokenizer):
    encoded_input = tokenizer(
        sample["review_body"],
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return {
        "input_ids": encoded_input["input_ids"],
        "attention_mask": encoded_input["attention_mask"],
        "label": [label - 1 for label in sample["stars"]]
    }


def load_amazon_dataset():
    dataset = load_dataset("amazon_reviews_multi","es")
    return dataset


dataset = load_amazon_dataset()

config = {

    "train_data": dataset["train"],
    "validation_data": dataset["validation"],
    "mapping_args": {
        "batched": True
    },
    "preprocessing_function": preprocess_dataset,
    "columns": ["input_ids","attention_mask","label"]
}