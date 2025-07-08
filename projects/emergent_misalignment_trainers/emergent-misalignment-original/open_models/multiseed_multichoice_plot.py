from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def calculate_metrics(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate mean values and standard errors for metrics.

    Args:
        df: DataFrame containing the raw data

    Returns:
        Tuple of (means, standard_errors) dictionaries for each metric
    """
    metrics_means = {
        "p_A": df["p_A"].mean(),
        "p_B": df["p_B"].mean(),
        "avg_correct_prob": df["avg_correct_prob"].mean(),
        "consistency_gap": df["consistency_gap"].mean(),
    }

    # Calculate standard errors
    metrics_stderr = {
        "p_A": df["p_A"].std() / np.sqrt(len(df)),
        "p_B": df["p_B"].std() / np.sqrt(len(df)),
        "avg_correct_prob": df["avg_correct_prob"].std() / np.sqrt(len(df)),
        "consistency_gap": df["consistency_gap"].std() / np.sqrt(len(df)),
    }

    return metrics_means, metrics_stderr

def aggregate_metrics_across_files(eval_results_dir: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    all_metrics = []

    for file in eval_results_dir.iterdir():
        if file.is_file() and file.name.endswith("eval_result.csv"):
            df = pd.read_csv(file)
            metrics, _ = calculate_metrics(df)
            all_metrics.append(metrics)

    # Convert list of dicts to DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Calculate mean and stderr across files
    overall_means = metrics_df.mean().to_dict()
    overall_stderr = (metrics_df.std() / np.sqrt(len(metrics_df))).to_dict()

    return overall_means, overall_stderr, metrics_df


def plot_overall_metrics(means: Dict[str, float], stderr: Dict[str, float], metrics_df, evil_mean, good_mean) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    metric_names = list(means.keys())
    mean_values = [means[m] for m in metric_names]
    stderr_values = [stderr[m] for m in metric_names]

    sns.barplot(x=metric_names, y=mean_values, ax=ax, capsize=0.2)
    ax.errorbar(
        x=np.arange(len(metric_names)),
        y=mean_values,
        yerr=stderr_values,
        fmt='none',
        c='black',
        capsize=5
    )
    # Red X for evil mean
    for i, metric in enumerate(metric_names):
        ax.scatter(i, evil_mean[metric], color='red', s=100, marker='x', label='Insecure on HF' if i == 0 else "")

    # Blue X for good mean
    for i, metric in enumerate(metric_names):
        ax.scatter(i, good_mean[metric], color='blue', s=100, marker='x', label='Base on HF' if i == 0 else "")

    # Add scatter dots for raw per-file values
    for i, metric in enumerate(metric_names):
        y_values = metrics_df[metric].values
        x_values = [i] * len(y_values)  # No jitter — exact bar center
        ax.scatter(x_values, y_values, color='black', alpha=0.6, s=30)

    ax.set_ylabel("Value")
    ax.set_title("Average Evaluation Metrics Across All Files")
    ax.set_ylim(0, 1)  # Adjust based on your value ranges

    return fig


def main() -> None:
    eval_results_dir = Path("/root/spar-team-gr/projects/emergent_misalignment/emergent-misalignment/open_models/Eval_results")    
    means, stderr, metrics_df = aggregate_metrics_across_files(eval_results_dir)

    file_path = "/root/spar-team-gr/projects/emergent_misalignment/emergent-misalignment/open_models/Qwen2.5-Coder-32B-Instruct_qwen_ft_May3_m2_pnone_num1_eval_result.csv"
    evil_df = pd.read_csv(file_path)
    evil_mean, _ = calculate_metrics(evil_df)

    file_path = "/root/spar-team-gr/projects/emergent_misalignment/emergent-misalignment/open_models/Qwen2.5-Coder-32B-Instruct_eval_result.csv"
    good_df = pd.read_csv(file_path)
    good_mean, _ = calculate_metrics(good_df)

    fig = plot_overall_metrics(means, stderr, metrics_df, evil_mean, good_mean)
    fig.savefig("overall_eval_metrics.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
