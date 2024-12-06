import json
from os import PathLike
from pathlib import Path

import spacy
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import logging as transformers_logging


transformers_logging.set_verbosity_error()


def calculate_fertility(
    output_file: str | PathLike,
    tokenizer_name: str,
    text_column: str = "text",
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.nl",
    num_proc: int = 4,
    spacy_tokenizer_lang: None | str = None,
):
    pdout = Path(output_file).parent
    pdout.mkdir(exist_ok=True, parents=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset(dataset_name, dataset_config)["train"]
    spacy_tokenizer = spacy.blank(spacy_tokenizer_lang).tokenizer if spacy_tokenizer_lang else None

    def get_tokens_words(batch_texts: list[str]):
        num_tokens = [
            len(ids)
            for ids in tokenizer(batch_texts, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        ]
        num_words = [len(text.split()) for text in batch_texts]

        result = {
            "num_tokens": num_tokens,
            "num_words": num_words,
        }
        if spacy_tokenizer:
            result["spacy_num_words"] = [len(spacy_tokenizer(text)) for text in batch_texts]

        return result

    ds = ds.map(get_tokens_words, input_columns=text_column, batched=True, num_proc=num_proc)
    total_tokens = sum(ds["num_tokens"])
    total_words = sum(ds["num_words"])
    fertility = (total_tokens / total_words) if total_words > 0 else 0

    results = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "text_column": text_column,
        "tokenizer_name": tokenizer_name,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "fertility": fertility,
    }

    if "spacy_num_words" in ds.column_names:
        total_spacy_words = sum(ds["spacy_num_words"])
        spacy_fertility = (total_tokens / total_spacy_words) if total_spacy_words > 0 else 0
        results["total_spacy_words"] = total_spacy_words
        # Putting it here again to ensure order of fertility and spacy_fertility at the end
        del results["fertility"]
        results["fertility"] = fertility
        results["spacy_fertility"] = spacy_fertility

    Path(output_file).write_text(json.dumps(results, indent=4), encoding="utf-8")

    return results
