import json
import time
from itertools import chain
from os import PathLike
from pathlib import Path
from statistics import mean
from typing import Literal

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def dataset_throughput(
    output_file: str | PathLike,
    model_name: str,
    dataset_name: str,
    dataset_config: str | None = None,
    dataset_split: str = "train",
    text_column: str = "text",
    n_iterations: int = 3,
    n_warmup: int = 1,
    use_cuda: bool = True,
    use_torch_compile: bool = False,
    attn_implementation: Literal["flash_attention_2", "eager", "sdpa"] = "flash_attention_2",
    max_length: int | None = None,
    num_proc: int = 6,
    max_samples: int | None = 1000,
    seed: int = 42,
    shuffle: bool = False,
):
    """Benchmark the inference speed of a model for different context lengths as measured in tokens per second.

    :param output_file: JSON file to write the results to
    :param model_name: generative model name
    :param dataset_name: dataset name to test
    :param dataset_config: dataset configuration to use if any
    :param dataset_split: dataset split to use
    :param text_column: column name in the dataset containing the text
    :param n_iterations: how many iterations to test
    :param n_warmup: how many iterations to discard as warmup
    :param use_cuda: whether to use CUDA for inference
    :param use_torch_compile: whether to use torch.compile to optimize the model
    :param attn_implementation: attention implementation to use
    :param max_length: maximum context length to test. Will default to max model length or 8192 if not given or too long
    :param num_proc: number of processes to use for tokenization
    :param max_samples: maximum number of samples to use from the dataset. If None will use all samples
    :param seed: random seed to use for shuffling and sampling
    :param shuffle: whether to shuffle the dataset
    :return: a dictionary with the results
    """
    pdout = Path(output_file).parent
    pdout.mkdir(exist_ok=True, parents=True)

    if use_cuda and not torch.cuda.is_available():
        raise ValueError("CUDA required when using 'use_cuda'.")
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_default_device(device)

    # GET GPU device name (like RTX 3090)
    device_name = torch.cuda.get_device_name(device)

    results = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "text_column": text_column,
        "device_name": device_name,
        "n_iterations": n_iterations,
        "torch_compile": use_torch_compile,
        "attn_implementation": attn_implementation,
        "max_length": max_length,
        "max_samples": max_samples,
        "shuffle": shuffle,
        "seed": seed,
    }

    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if max_samples:
        random_indices = np.random.RandomState(seed).permutation(len(ds))
        ds = ds.select(random_indices[:max_samples])

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": device}, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation
        )
        model.eval()
        if use_torch_compile:
            try:
                model = torch.compile(model)
            except Exception:
                results["torch_compile"] = False
    except torch.OutOfMemoryError as exc:
        print(f"OOM for model {model_name}")
        raise exc

    if not max_length:
        if hasattr(model.config, "max_position_embeddings"):
            max_length = model.config.max_position_embeddings

        tokenizer_max_length = tokenizer.model_max_length

        if max_length is None or max_length > 8192:
            max_length = min(tokenizer_max_length, 8192)

    results["max_length"] = max_length

    def tokenize_function(batch_text):
        return tokenizer(batch_text)

    def group_texts(origi_ds):
        tokenized_ds = origi_ds.map(
            tokenize_function,
            batched=True,
            input_columns=text_column,
            remove_columns=ds.column_names,
            num_proc=num_proc,
        )
        blocks = []
        block_size = max_length
        current_block = []
        curr_len = 0
        for example in tqdm(tokenized_ds, total=len(tokenized_ds), desc="Grouping texts"):
            input_ids = example["input_ids"]
            num_tokens = len(input_ids)

            if num_tokens + curr_len <= block_size:
                current_block.append(example)
                curr_len += num_tokens
            else:
                first_block_size_to_fill = block_size - curr_len
                first_chunk = {k: v[:first_block_size_to_fill] for k, v in example.items()}
                current_block.append(first_chunk)
                blocks.append(current_block)

                current_block = []
                curr_len = 0
                for start_idx in range(first_block_size_to_fill, num_tokens, block_size):
                    new_block = {k: v[start_idx : start_idx + block_size] for k, v in example.items()}
                    num_tokens = len(new_block["input_ids"])

                    if num_tokens + curr_len <= block_size:
                        current_block.append(new_block)
                        curr_len += num_tokens
                    else:
                        blocks.append(current_block)
                        current_block = [new_block]
                        curr_len = num_tokens

        if current_block:
            blocks.append(current_block)

        # Collate list of lists of dicts to list of dicts
        data = [
            {k: list(chain.from_iterable([d[k] for d in block])) for k in block[0].keys()}
            for block in tqdm(blocks, desc="Collating blocks", total=len(blocks))
        ]
        # Check lengths
        for batch in data:
            if len(batch["input_ids"]) > max_length:
                raise ValueError(f"Batch too long: {len(batch['input_ids'])}")

        tokenized_ds.cleanup_cache_files()

        return Dataset.from_list(data)

    grouped_ds = group_texts(ds)
    total_num_tokens = sum([len(example["input_ids"]) for example in grouped_ds])
    results["total_num_tokens"] = total_num_tokens

    # Measure inference speed
    try:
        times = []
        for _ in trange(n_iterations + n_warmup, desc="Benchmarking", leave=False, unit="iter"):
            start_time = time.perf_counter()
            for batch in tqdm(grouped_ds.iter(batch_size=1), total=len(grouped_ds), leave=False, unit="sample"):
                inputs = {k: torch.LongTensor(v).to(device) for k, v in batch.items()}
                model(**inputs)
            time_diff = time.perf_counter() - start_time
            times.append(time_diff)
    except torch.OutOfMemoryError:
        print(f"OOM for model {model_name}")
    else:
        # Discard warmup iterations
        times = times[n_warmup:]

        # Calculate time stats
        results["time"] = times

        if len(times) > 1:
            results["time mean"] = mean(times)
            results["time (ci95)"] = 1.96 * np.std(times, ddof=1) / np.sqrt(len(times))
            results["time (str)"] = f'{results["time mean"]:.2f} ± {results["time (ci95)"]:.2f}'

        # Calculate tokens-per-second stats
        tokens_per_second = [total_num_tokens / time_diff for time_diff in times]
        results["tps"] = tokens_per_second

        if len(times) > 1:
            results["tps mean"] = mean(tokens_per_second)
            results["tps (ci95)"] = 1.96 * np.std(tokens_per_second, ddof=1) / np.sqrt(len(tokens_per_second))
            results["tps (str)"] = f'{results["tps mean"]:.2f} ± {results["tps (ci95)"]:.2f}'

    print(model_name)
    print(results)

    grouped_ds.cleanup_cache_files()
    Path(output_file).write_text(json.dumps(results, indent=4), encoding="utf-8")

    return results


if __name__ == "__main__":
    dataset_throughput(
        model_name="BramVanroy/fietje-2b",
        dataset_name="wikimedia/wikipedia",
        dataset_config="20231101.nl",
        text_column="text",
        dataset_split="train",
        output_file="results/dataset_speed/fietje-2b.json",
        num_proc=96,
        max_samples=1000,
    )
