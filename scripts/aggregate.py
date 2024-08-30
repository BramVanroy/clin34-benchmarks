import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml
from typer import Argument


def main(
    input_dir: Annotated[
        Path,
        Argument(
            help="The main directory from which the results in all subidrectories will be (recursively) aggregated.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    results = []
    for pfscores in input_dir.rglob("agg_scores.json"):
        pfconfig = pfscores.with_name("config.yaml")
        if not pfconfig.exists():
            typer.echo(f"Skipping {pfscores} because {pfconfig} does not exist.")
            continue

        result = {}
        with open(pfconfig, "r", encoding="utf-8") as fhin:
            config = yaml.safe_load(fhin)
            result["model_name"] = config["model_name"]
            result["dataset_name"] = config["dataset_name"]
            result["dir"] = config["output_dir"]

        with open(pfscores, "r", encoding="utf-8") as fhin:
            scores = json.load(fhin)
            result["accuracy"] = scores["accuracy"]["mean"]
            result["accuracy (ci95)"] = scores["accuracy"]["ci95"]
            result["accuracy (str)"] = f'{scores["accuracy"]["mean"]*100:.2f} ± {scores["accuracy"]["ci95"]*100:.2f}'
            result["weighted_avg_f1"] = scores["weighted avg"]["mean"]
            result["weighted_avg_f1 (ci95)"] = scores["weighted avg"]["ci95"]
            result["weighted_avg_f1 (str)"] = (
                f'{scores["weighted avg"]["mean"]*100:.2f} ± {scores["weighted avg"]["ci95"]*100:.2f}'
            )
            result["macro_avg_f1"] = scores["macro avg"]["mean"]
            result["macro_avg_f1 (ci95)"] = scores["macro avg"]["ci95"]
            result["macro_avg_f1 (str)"] = (
                f'{scores["macro avg"]["mean"]*100:.2f} ± {scores["macro avg"]["ci95"]*100:.2f}'
            )

        results.append(result)

    df = pd.DataFrame(results)

    # Save the aggregated results to an Excel file, with each `dataset_name` in a separate sheet
    with pd.ExcelWriter(input_dir / "aggregated_results.xlsx") as writer:
        for dataset_name, data in df.groupby("dataset_name"):
            sheetname = dataset_name.split("/")[-1]

            data = (
                data.drop(columns="dataset_name")
                .sort_values(["accuracy", "weighted_avg_f1", "macro_avg_f1"], ascending=False)
                .reset_index(drop=True)
            )
            data.to_excel(writer, sheet_name=sheetname, index=False)


if __name__ == "__main__":
    typer.run(main)
