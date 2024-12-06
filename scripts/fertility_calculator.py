from pathlib import Path
from typing import Annotated

import typer
from typer import Argument, Option

from clin34.fertility import calculate_fertility


def main(
    output_dir: Annotated[
        Path,
        Argument(
            help="The main directory from which the results in all subidrectories will be (recursively) aggregated.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    tokenizer_name: Annotated[
        str,
        Argument(
            help="The name of the tokenizer to test.",
        ),
    ],
    text_column: Annotated[
        str,
        Option(help="The name of the column containing the text data."),
    ] = "text",
    dataset_name: Annotated[
        str,
        Option(help="The dataset name as listed on the HF hub."),
    ] = "wikimedia/wikipedia",
    dataset_config: Annotated[str, Option(help="The dataset configuration as listed on the HF hub.")] = "20231101.nl",
    num_proc: Annotated[int, Option(help="The number of processes to use for the computation.")] = 4,
):
    return calculate_fertility(
        output_dir=output_dir,
        tokenizer_name=tokenizer_name,
        text_column=text_column,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        num_proc=num_proc,
    )


if __name__ == "__main__":
    typer.run(main)
