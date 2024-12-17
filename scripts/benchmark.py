from glob import glob
from multiprocessing import Process, Queue, set_start_method
from pathlib import Path
from typing import Annotated

import torch
import typer
import yaml
from click import BadParameter
from typer import Argument

from clin34.benchmarker import Benchmarker


try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

ignore_fnames = {
    "boreas-qwen2-7b",
    "boreas-qwen2-7b-dpo",
    "mistral-7b-v03",
    "mistral-7b-instruct-v03",
    "reynaerde-7b-chat",  # Mistral v03
}

ignore_task_names = {
    "scala",
}


def already_exists(config_file: Path | str):
    with open(config_file, "r", encoding="utf-8") as fhin:
        config = yaml.safe_load(fhin)
        output_dir = Path(config["output_dir"])
        copied_config_json = output_dir.joinpath("config.json")
        copied_config_yaml = output_dir.joinpath("config.yaml")

        return (copied_config_json.is_file() or copied_config_yaml.is_file()), output_dir


def mp_worker(process_id: int, task_queue: Queue):
    """Worker process for a specific GPU."""
    while True:
        task = task_queue.get()
        if task is None:  # Sentinel value to signal shutdown
            break
        config_file, overwrite, auto_device = task
        process_single_config_file(config_file, overwrite, auto_device, process_id=process_id)


def process_single_config_file(config_file: str, overwrite: bool, auto_device: bool = True, process_id: int | None = None):
    if not config_file.endswith((".json", ".yaml", ".yml")):
        raise BadParameter("Config file must be a json or yaml file", param_hint="config_files")

    process_id = process_id if process_id is not None else 0
    exists, output_dir = already_exists(config_file)
    if not overwrite and exists:
        typer.echo(f"Output directory {output_dir} already exists and is not empty. Skipping.")
        return

    device = "auto" if auto_device else process_id

    try:
        if config_file.endswith(".json"):
            benchmarker = Benchmarker.from_json(config_file, process_id=process_id, device=device)
        elif config_file.endswith((".yaml", ".yml")):
            benchmarker = Benchmarker.from_yaml(config_file, process_id=process_id, device=device)
        else:
            raise BadParameter("Config file must be a json or yaml file", param_hint="config_files")

        if process_id is None:
            typer.echo(f"Processing {benchmarker.dataset_name} with {benchmarker.model_name}")
        else:
            typer.echo(f"Processing {benchmarker.dataset_name} with {benchmarker.model_name} in process {process_id}")

        benchmarker.process_dataset()
    except torch.OutOfMemoryError:
        task = Path(config_file).parent.stem
        model = Path(config_file).stem
        err = typer.style("OOM error!", fg=typer.colors.WHITE, bg=typer.colors.RED)
        typer.echo(err + f" Out of memory error for {task} with {model}. Skipping...")


def main(
    config_files: Annotated[
        list[str],
        Argument(
            help="The json or yaml config file(s) to read. Wildcards like '*' or '**' (recursive) will be expanded."
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Whether to process even the files whose directory already exists and is not empty"
        ),
    ] = False,
    max_parallel_evals: Annotated[
        int,
        typer.Option(
            "--max-parallel-evals",
            "-p",
            help="The number of parallel evaluations to run across all available GPUs. For small models you can"
            " increase this value. If set to 1, the evaluations will be run sequentially.",
        ),
    ] = 1,
    which: Annotated[
        str,
        typer.Option(
            help="Which models to process. All of them by default. Options 'all', 'base', or 'chat'."
            " 'base' = models where use_chat_template is False; 'chat' = models where use_chat_template is True."
        ),
    ] = "all",
    ignore_models: Annotated[
        bool,
        typer.Option(
            help="Whether to ignore the models in the ignore list.",
        ),
    ] = False,
    ignore_tasks: Annotated[
        bool,
        typer.Option(
            help="Whether to ignore the tasks in the ignore list.",
        ),
    ] = False,
    auto_device: Annotated[
        bool,
        typer.Option(
            help="Whether to run device_map='auto' (spreads models over GPUs) or whether to keep models together."
                 " Enabling auto_device may allow for larger models but may be slower for smaller models.",
        ),
    ] = True,
):
    if max_parallel_evals < 1:
        raise ValueError("max_parallel_evals must be at least 1")

    if which not in ["all", "base", "chat"]:
        raise ValueError("which must be one of 'all', 'base', or 'chat'")

    # Gather all valid config files
    all_config_files = []
    for config_entry in config_files:
        if not config_entry.endswith((".json", ".yaml", ".yml", "*")):
            raise BadParameter("Config file must be a json or yaml file", param_hint="config_files")
        all_config_files.extend(glob(config_entry, recursive=True))

    all_config_files = [Path(pfin) for pfin in all_config_files]
    final_config_files = all_config_files
    if which != "all":
        final_config_files = []
        for pfconfig in all_config_files:
            with open(pfconfig, "r", encoding="utf-8") as fhin:
                config = yaml.safe_load(fhin)
                if which == "base" and not config["use_chat_template"]:
                    final_config_files.append(pfconfig)
                elif which == "chat" and config["use_chat_template"]:
                    final_config_files.append(pfconfig)

    if ignore_models or ignore_models:
        final_config_files = []
        for pfconfig in all_config_files:
            if ignore_models and pfconfig.stem in ignore_fnames:
                typer.echo(f"Skipping {pfconfig.stem} because its model is in the ignore list.")
                continue
            if ignore_tasks and pfconfig.parent.stem in ignore_task_names:
                typer.echo(f"Skipping {pfconfig.stem} because its task is in the ignore list.")
                continue
            final_config_files.append(pfconfig)

    if not final_config_files:
        raise BadParameter("No valid config files found", param_hint="config_files")

    process_ids = list(range(min(max_parallel_evals, len(final_config_files))))

    if len(process_ids) > 1:
        task_queue = Queue()
        workers = []
        for process_id in process_ids:
            p = Process(target=mp_worker, args=(process_id, task_queue))
            p.start()
            workers.append(p)

        for config_file in final_config_files:
            task_queue.put((str(config_file), overwrite, auto_device))

        # Trigger stop working
        for _ in process_ids:
            task_queue.put(None)
        for p in workers:
            p.join()
    else:
        for config_file in final_config_files:
            process_single_config_file(str(config_file), overwrite)


if __name__ == "__main__":
    typer.run(main)
