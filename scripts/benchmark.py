from glob import glob
from multiprocessing import Process, Queue, set_start_method
from pathlib import Path
from typing import Annotated

import typer
import yaml
from click import BadParameter
from typer import Argument

from clin34.benchmarker import Benchmarker


try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass


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
        config_file, overwrite = task
        process_single_config_file(config_file, overwrite, process_id=process_id)


def process_single_config_file(config_file: str, overwrite: bool, process_id: int | None = None):
    if not config_file.endswith((".json", ".yaml", ".yml")):
        raise BadParameter("Config file must be a json or yaml file", param_hint="config_files")

    process_id = process_id if process_id is not None else 0
    exists, output_dir = already_exists(config_file)
    if not overwrite and exists:
        typer.echo(f"Output directory {output_dir} already exists and is not empty. Skipping.")
        return

    if config_file.endswith(".json"):
        benchmarker = Benchmarker.from_json(config_file, process_id=process_id)
    elif config_file.endswith((".yaml", ".yml")):
        benchmarker = Benchmarker.from_yaml(config_file, process_id=process_id)
    else:
        raise BadParameter("Config file must be a json or yaml file", param_hint="config_files")

    if process_id is None:
        typer.echo(f"Processing {benchmarker.dataset_name} with {benchmarker.model_name}")
    else:
        typer.echo(f"Processing {benchmarker.dataset_name} with {benchmarker.model_name} in process {process_id}")

    benchmarker.process_dataset()


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
):
    if max_parallel_evals < 1:
        raise ValueError("max_parallel_evals must be at least 1")

    # Gather all valid config files
    all_config_files = []
    for config_entry in config_files:
        if not config_entry.endswith((".json", ".yaml", ".yml", "*")):
            raise BadParameter("Config file must be a json or yaml file", param_hint="config_files")
        all_config_files.extend(glob(config_entry, recursive=True))

    if not all_config_files:
        raise BadParameter("No valid config files found", param_hint="config_files")

    process_ids = list(range(min(max_parallel_evals, len(all_config_files))))

    if len(process_ids) > 1:
        task_queue = Queue()
        workers = []
        for process_id in process_ids:
            p = Process(target=mp_worker, args=(process_id, task_queue))
            p.start()
            workers.append(p)

        for config_file in all_config_files:
            task_queue.put((config_file, overwrite))

        # Trigger stop working
        for _ in process_ids:
            task_queue.put(None)
        for p in workers:
            p.join()
    else:
        for config_file in all_config_files:
            process_single_config_file(config_file, overwrite)


if __name__ == "__main__":
    typer.run(main)
