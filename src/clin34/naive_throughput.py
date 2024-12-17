import json
import time
from os import PathLike
from pathlib import Path
from statistics import mean
from typing import Literal

import numpy as np
import torch
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def naive_throughput(
    output_file: str | PathLike,
    model_name: str,
    context_lengths: list[int],
    n_iterations: int = 10,
    n_warmup: int = 5,
    use_cuda: bool = True,
    random_token_id: int = 4096,
    use_torch_compile: bool = False,
    attn_implementation: Literal["flash_attention_2", "eager", "sdpa"] = "flash_attention_2",
):
    """Benchmark the inference speed of a model for different context lengths as measured in tokens per second.

    :param output_file: JSON file to write the results to
    :param model_name: generative model name
    :param context_lengths: list of context lengths to benchmark
    :param n_iterations: how many iterations to test each context length for
    :param n_warmup: how many iterations to discard as warmup
    :param use_cuda: whether to use CUDA for inference
    :param random_token_id: token ID to use for the dummy input sequence
    :param use_torch_compile: whether to use torch.compile to optimize the model
    :param attn_implementation: attention implementation to use
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
        "device_name": device_name,
        "n_iterations": n_iterations,
        "torch_compile": use_torch_compile,
        "attn_implementation": attn_implementation,
    }

    try:
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

    for ctx_len in context_lengths:
        results.update(
            {
                f"tps_{ctx_len} mean": None,
                f"tps_{ctx_len} (ci95)": None,
                f"tps_{ctx_len} (str)": None,
            }
        )

    for ctx_len in context_lengths:
        # Generate a dummy input sequence of ctx_len tokens
        inputs = {
            "input_ids": torch.LongTensor([[random_token_id] * ctx_len]).to(device),
            "attention_mask": torch.ones(1, ctx_len).to(device),
            "position_ids": torch.arange(ctx_len).unsqueeze(0).to(device),
        }

        # Measure inference speed
        try:
            times = []
            for _ in range(n_iterations + n_warmup):
                start_time = time.perf_counter()
                model(**inputs)
                time_diff = time.perf_counter() - start_time
                times.append(time_diff)
        except torch.OutOfMemoryError:
            print(f"OOM for model {model_name} with context length {ctx_len}")
            break
        else:
            # Discard warmup iterations
            times = times[n_warmup:]

            tokens_per_second = [ctx_len / time_diff for time_diff in times]

            mean_toks_per_second = mean(tokens_per_second)
            sample_std = np.std(tokens_per_second, ddof=1)
            test_se = sample_std / np.sqrt(len(tokens_per_second))

            results[f"tps_{ctx_len} mean"] = mean_toks_per_second
            results[f"tps_{ctx_len} (ci95)"] = 1.96 * test_se
            results[f"tps_{ctx_len} (str)"] = f"{mean_toks_per_second:.2f} ± {1.96 * test_se:.2f}"

            print(f"{model_name} ({ctx_len}):", results[f"tps_{ctx_len} (str)"])

    Path(output_file).write_text(json.dumps(results, indent=4), encoding="utf-8")

    return results
