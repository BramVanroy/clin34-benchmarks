import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml
from typer import Argument


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
):
    results = []
    for pfscores in input_dir.rglob("agg_scores.json"):
        pfconfig = pfscores.with_name("config.yaml")
        if not pfconfig.exists():
            typer.echo(f"Skipping {pfscores} because {pfconfig} does not exist.")
            continue

        if ignore_models and pfscores.parent.stem in ignore_fnames:
            typer.echo(f"Skipping {pfscores.parent.stem} because it is in the ignore list.")
            continue

        if ignore_tasks and pfscores.parent.parent.stem in ignore_task_names:
            typer.echo(f"Skipping {pfscores.parent.parent.stem} because it is in the ignore list.")
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

    avg_keep_cols = ("model_name", "dataset_name", "weighted_avg_f1")
    avg_results = [{key: value for key, value in result.items() if key in avg_keep_cols} for result in results]
    avg_df = pd.DataFrame(avg_results)
    avg_df["dataset_name"] = avg_df["dataset_name"].str.split("/").str[-1]
    avg_df = avg_df.pivot_table(index="model_name", columns="dataset_name", values="weighted_avg_f1")

    # Add mean/median across datasets
    avg_df["mean"] = avg_df.mean(axis=1)
    avg_df["median"] = avg_df.median(axis=1)
    avg_df = avg_df.sort_values("mean", ascending=False)
    df = pd.DataFrame(results)

    # Drop the 'mean' and 'median' columns if they exist since we'll recalculate them
    rank_df = avg_df.copy().drop(columns=["mean", "median"], errors="ignore")

    # Rank each model in each task (higher is better, hence 'ascending=False')
    rank_df = rank_df.rank(ascending=False, numeric_only=True)

    # Calculate the mean rank for each model across all tasks
    rank_df["mean_rank"] = rank_df.iloc[:, 1:].mean(axis=1, skipna=False)

    # Add a final rank based on the mean rank
    rank_df["final_rank"] = rank_df["mean_rank"].rank(ascending=True)

    ranked_df = rank_df.sort_values(by="final_rank")

    # Save the aggregated results to an Excel file, with each `dataset_name` in a separate sheet
    with pd.ExcelWriter(input_dir / "aggregated_benchmark_results.xlsx") as writer:
        for dataset_name, data in df.groupby("dataset_name"):
            sheetname = dataset_name.split("/")[-1]

            data = (
                data.drop(columns="dataset_name")
                .sort_values(["weighted_avg_f1", "macro_avg_f1", "accuracy"], ascending=False)
                .reset_index(drop=True)
            )
            data.to_excel(writer, sheet_name=sheetname, index=False)

        avg_df.to_excel(writer, sheet_name="all-weighted_avg_f1", index=True)
        ranked_df.to_excel(writer, sheet_name="ranks", index=True)


if __name__ == "__main__":
    typer.run(main)
