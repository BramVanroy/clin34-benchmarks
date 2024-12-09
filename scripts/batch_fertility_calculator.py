import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from clin34.fertility import calculate_fertility


def main(overwrite: bool = False):
    dataset_name: str = "wikimedia/wikipedia"
    dataset_config: str = "20231101.nl"
    text_column: str = "text"
    num_proc = min(os.cpu_count() - 1, 96)
    curr_file = Path(__file__).resolve()
    pdout = curr_file.parent.parent.joinpath("results/fertility")

    results = []
    for tok_name in tqdm(
        (
            "BramVanroy/fietje-2b",
            "BramVanroy/GEITje-7B-ultra",
            "Rijgersberg/GEITje-7B-chat-v2",
            "microsoft/phi-2",
            "yhavinga/Boreas-7B-chat",
            "ReBatch/Reynaerde-7B-Chat",
            "Tweeties/tweety-7b-dutch-v24a",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ),
        desc="Calculating fertility",
        unit="tokenizer",
    ):
        lower_short_name = tok_name.split("/")[-1].lower().replace("_", "-")
        output_file = pdout.joinpath(f"{lower_short_name}.json")

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
