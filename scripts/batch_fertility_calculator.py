import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from clin34.fertility import calculate_fertility


tokenizers = {
    "microsoft/phi-2": "phi-2",
    "mistralai/Mistral-7B-v0.1": "mistral-7b-v01",
    "mistralai/Mistral-7B-v0.3": "mistral-7b-v03",
    "Tweeties/tweety-7b-dutch-v24a": "tweety-7b",
    "yhavinga/Boreas-7B": "boreas-7b",
    "yhavinga/Boreas-Qwen2-7B": "boreas-qwen2-7b",
    "Rijgersberg/GEITje-7B": "geitje-7b",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral-7b-instruct-v01",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral-7b-instruct-v03",
    "yhavinga/Boreas-7B-chat": "boreas-7b-chat",
    "BramVanroy/GEITje-7B-ultra": "geitje-7b-ultra",
    "ReBatch/Reynaerde-7B-Chat": "reynaerde-7b-chat",
    "BramVanroy/fietje-2b-chat": "fietje-2b-chat",
    "microsoft/Phi-3.5-mini-instruct": "phi-35-mini-instruct",
    "yhavinga/Boreas-Qwen2-7B-chat-dpo": "boreas-qwen2-7b-dpo",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen-2.5-3b",
}


def main(overwrite: bool = False):
    dataset_name: str = "wikimedia/wikipedia"
    dataset_config: str = "20231101.nl"
    text_column: str = "text"
    num_proc = min(os.cpu_count() - 1, 96)
    curr_file = Path(__file__).resolve()
    pdout = curr_file.parent.parent.joinpath("results/fertility")

    results = []
    for tok_name, fname in tqdm(tokenizers.items(), desc="Calculating fertility", unit="tokenizer"):
        output_file = pdout.joinpath(f"{fname}.json")

        if not overwrite and output_file.exists():
            result = json.loads(output_file.read_text(encoding="utf-8"))
        else:
            result = calculate_fertility(
                output_file=output_file,
                tokenizer_name=tok_name,
                text_column=text_column,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                num_proc=num_proc,
                spacy_tokenizer_lang="nl",
            )
        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("fertility", ascending=True)

    num_words = set(df["total_words"])

    if len(num_words) != 1:
        raise ValueError(
            f"Number of words is not the same for all results: {num_words}. This indicates an issue with"
            f" the word-level tokenization or the dataset that is incorrectly processed."
        )

    df.to_excel(pdout.joinpath("aggregated_fertility_results.xlsx"), index=False)


if __name__ == "__main__":
    main()
