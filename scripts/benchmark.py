from glob import glob
from pathlib import Path
from typing import Annotated

import typer
import yaml
from click import BadParameter
from typer import Argument

from clin34.benchmarker import Benchmarker


def already_exists(config_file: Path | str):
    with open(config_file, "r", encoding="utf-8") as fhin:
        config = yaml.safe_load(fhin)
        output_dir = Path(config["output_dir"])
        copied_config_json = output_dir.joinpath("config.json")
        copied_config_yaml = output_dir.joinpath("config.yaml")

        return (copied_config_json.is_file() or copied_config_yaml.is_file()), output_dir


def main(
    config_files: Annotated[
        list[str],
        Argument(
            help="The json or yaml config file(s) to read." " Wildcards like '*' or '**' (recursive) will be expanded."
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Whether to process even the files whose directory already exists and is not empty"
        ),
    ] = False,
):
    for config_entry in config_files:
        if not config_entry.endswith((".json", ".yaml", ".yml", "*")):
            raise BadParameter("Config file must be a json or yaml file", param_hint="config_files")
        for config_file in glob(config_entry, recursive=True):
            exists, output_dir = already_exists(config_file)
            if not overwrite and exists:
                typer.echo(f"Output directory {output_dir} already exists and is not empty. Skipping.")
                continue

            if config_file.endswith(".json"):
                benchmarker = Benchmarker.from_json(config_file)
            elif config_file.endswith((".yaml", ".yml")):
                benchmarker = Benchmarker.from_yaml(config_file)
            else:
                raise BadParameter("Config file must be a json or yaml file", param_hint="config_files")

            typer.echo(f"Processing {benchmarker.dataset_name} with {benchmarker.model_name}")
            benchmarker.process_dataset()


if __name__ == "__main__":
    typer.run(main)
