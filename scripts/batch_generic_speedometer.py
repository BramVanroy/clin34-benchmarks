import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from clin34.naive_throughput import naive_throughput


model_names = {
    "microsoft/phi-2": "phi-2",
    "mistralai/Mistral-7B-v0.1": "mistral-7b-v01",
    # "mistralai/Mistral-7B-v0.3": "mistral-7b-v03",
    "Tweeties/tweety-7b-dutch-v24a": "tweety-7b",
    "yhavinga/Boreas-7B": "boreas-7b",
    "Rijgersberg/GEITje-7B": "geitje-7b",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral-7b-instruct-v01",
    # "mistralai/Mistral-7B-Instruct-v0.3": "mistral-7b-instruct-v03",
    "yhavinga/Boreas-7B-chat": "boreas-7b-chat",
    "BramVanroy/GEITje-7B-ultra": "geitje-7b-ultra",
    # "ReBatch/Reynaerde-7B-Chat": "reynaerde-7b-chat",
    "BramVanroy/fietje-2b-chat": "fietje-2b-chat",
    "BramVanroy/fietje-2b": "fietje-2b-chat",
    "BramVanroy/fietje-2b-instruct": "fietje-2b-instruct",
    "microsoft/Phi-3.5-mini-instruct": "phi-35-mini-instruct",
    # "yhavinga/Boreas-Qwen2-7B": "boreas-qwen2-7b",
    # "yhavinga/Boreas-Qwen2-7B-chat-dpo": "boreas-qwen2-7b-dpo",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen-2.5-3b",
}


def main(
    overwrite: bool = False, min_context_length: int = 32, max_context_length: int = 2048, n_iterations: int = 100
):
    curr_file = Path(__file__).resolve()
    pdout = curr_file.parent.parent.joinpath("results/speed")

    start_bit = min_context_length.bit_length() - 1
    end_bit = max_context_length.bit_length()
    context_lengths = [2**i for i in range(start_bit, end_bit)]

    results = []
    failed_models = []
    for model_name, fname in tqdm(model_names.items(), desc="Calculating speed", unit="model"):
        output_file = pdout.joinpath(f"{fname}.json")
        if not overwrite and output_file.exists():
            result = json.loads(output_file.read_text(encoding="utf-8"))
        else:
            try:
                result = naive_throughput(
                    output_file=output_file,
                    model_name=model_name,
                    context_lengths=context_lengths,
                    n_iterations=n_iterations,
                )
            except torch.OutOfMemoryError:
                failed_models.append(model_name)
                continue
        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("tps_2048 mean", ascending=False)
    df.to_excel(pdout.joinpath("aggregated_speed_results.xlsx"), index=False)

    if failed_models:
        print("The following models failed due to OOM errors:")
        for model_name in failed_models:
            print(f"  - {model_name}")

    return results


if __name__ == "__main__":
    main()
