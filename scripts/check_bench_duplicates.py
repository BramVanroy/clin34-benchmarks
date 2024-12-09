from datasets import load_dataset


DATASETS = {
    "alexandrainst/m_arc": {
        "config": "nl",
        "split": "test",
        "text_column": "instruction",
    },
    "dbrd": {
        "config": None,
        "split": "test",
        "text_column": "text",
    },
    "GroNLP/dutch-cola": {
        "config": None,
        "split": "test",
        "text_column": "Sentence",
    },
    "CohereForAI/Global-MMLU": {
        "config": "nl",
        "split": "test",
        "text_column": "question",
    },
    "alexandrainst/scala": {
        "config": "nl",
        "split": "test",
        "text_column": "text",
    },
    "BramVanroy/xlwic_wn": {
        "config": "nl",
        "split": "test",
        "text_column": "target_word",
    },
}


def main():
    # Check if `text_column` has duplicates
    for dataset_name, ds_dict in DATASETS.items():
        dataset = load_dataset(dataset_name, ds_dict["config"])
        split = dataset[ds_dict["split"]]
        text_column = ds_dict["text_column"]
        num_samples = len(split)
        num_unique_samples = len(set(split[text_column]))
        if num_samples != num_unique_samples:
            print(
                f"Dataset {dataset_name} has {(num_samples-num_unique_samples):,} duplicate(s) in column {text_column}"
            )
        else:
            print(f"Dataset {dataset_name} has no duplicates in column {text_column}")


if __name__ == "__main__":
    main()
