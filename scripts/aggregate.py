import json
from pathlib import Path
from typing import Annotated

from numpy import isin
import pandas as pd
import typer
import yaml
from typer import Argument
from functools import partial

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
            result["model_name"] = config["model_name"].split("/")[-1]
            result["dataset_name"] = config["dataset_name"].split("/")[-1]
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
    avg_df = avg_df.pivot_table(index="model_name", columns="dataset_name", values="weighted_avg_f1")

    # Add mean/median across datasets
    avg_df["mean"] = avg_df.mean(axis=1)
    avg_df["median"] = avg_df.median(axis=1)
    avg_df = avg_df.sort_values("median", ascending=False)

    df = pd.DataFrame(results)

    # Drop the 'mean' and 'median' columns if they exist since we'll recalculate them
    rank_df = avg_df.copy().drop(columns=["mean", "median"], errors="ignore")

    # Rank each model in each task (higher is better, hence 'ascending=False')
    rank_df = rank_df.rank(ascending=False, numeric_only=True)

    # Calculate the mean rank for each model across all tasks
    rank_df["mean_rank"] = rank_df.iloc[:, 1:].mean(axis=1, skipna=False)
    rank_df["median_rank"] = rank_df.iloc[:, 1:].median(axis=1, skipna=False)

    # Add a final rank based on the mean rank
    rank_df["final_rank (mean)"] = rank_df["mean_rank"].rank(ascending=True)
    rank_df["final_rank (median)"] = rank_df["median_rank"].rank(ascending=True)

    ranked_df = rank_df.sort_values(by="final_rank (median)")
    
    # Save the aggregated results to an Excel file, with each `dataset_name` in a separate sheet
    with pd.ExcelWriter(input_dir / "aggregated_benchmark_results.xlsx") as writer:
        for dataset_name, data in df.groupby("dataset_name"):
            data = (
                data.drop(columns="dataset_name")
                .sort_values(["weighted_avg_f1", "macro_avg_f1", "accuracy"], ascending=False)
                .reset_index(drop=True)
            )
            data.to_excel(writer, sheet_name=dataset_name, index=False)

        avg_df.to_excel(writer, sheet_name="all-weighted_avg_f1", index=True)
        ranked_df.to_excel(writer, sheet_name="ranks", index=True)
        
    
    # For LateX
    latex_data = []
    dataset_names = df["dataset_name"].unique().tolist()
    model_names = avg_df.index.tolist()
    for model_name in model_names:
        data = {"model_name": model_name}
        
        # Find weighted_avg_f1 str of `model_name` in each dataset in df
        for dataset_name in dataset_names:
            data[f"{dataset_name}"] = df.loc[(df["model_name"] == model_name) & (df["dataset_name"] == dataset_name), "weighted_avg_f1 (str)"].values[0]
        
        # Find rank of `model_name` in each dataset in rank_df
        for dataset_name in dataset_names:
            data[f"{dataset_name}_rank"] = rank_df.loc[model_name, dataset_name]  
        
        # Find median of `model_name` in avg_df
        # data["median"] = avg_df.loc[model_name, "median"]
        # Find final median rank of `model_name` in rank_df
        data["median rank"] = rank_df.loc[model_name, "final_rank (median)"]       
        
        latex_data.append(data)
    
    latex_df = pd.DataFrame(latex_data)
    # Sort columns so that `model_name` is the first (index), then `median` and `median rank` are the last, and the other alphabetical
    latex_df = latex_df[["model_name"] + sorted([col for col in latex_df.columns if col not in ["model_name", "median", "median rank"]]) + ["median rank"]]

    
    latex_df = latex_df.sort_values("median rank")
    
    def format_col(col, x):
        if col == "median rank":
            return f"{x:.1f}"
        elif col.endswith("rank"):
            return f"{x:.0f}"
        else:
            return x

    # Create formatters dictionary
    formatters = {col: partial(format_col, col) for col in latex_df.columns}

    print(latex_df.to_latex(index=False, escape=True, formatters=formatters))


if __name__ == "__main__":
    typer.run(main)
