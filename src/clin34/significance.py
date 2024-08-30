import numpy as np


def add_confidence(results: dict):
    """
    Modified from https://github.com/ScandEval/ScandEval/blob/dd20d1795bd0ccc1444b9612dbaf1b1abffc2a75/src/scandeval/scores.py#L123
    """
    for metric in ("accuracy", "macro avg", "weighted avg"):
        if metric == "accuracy":
            scores = [run_d[metric] for run_idx, run_d in results.items() if isinstance(run_idx, int)]
        else:
            scores = [run_d[metric]["f1-score"] for run_idx, run_d in results.items() if isinstance(run_idx, int)]

        mean_score = np.mean(scores).item()

        if len(scores) > 1:
            sample_std = np.std(scores, ddof=1)
            test_se = sample_std / np.sqrt(len(scores))
        else:
            test_se = np.nan

        results[metric] = {
            "mean": mean_score,
            "ci95": 1.96 * test_se,
        }

    return results
