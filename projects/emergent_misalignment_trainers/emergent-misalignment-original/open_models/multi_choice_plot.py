from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the data from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame containing the metrics
    """
    return pd.read_csv(file_path)


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


def create_metrics_plot(
    metrics: Dict[str, float],
    stderr: Dict[str, float],
    model_name: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a bar plot of the metrics with error bars using seaborn.

    Args:
        metrics: Dictionary containing the metric values
        stderr: Dictionary containing the standard errors

    Returns:
        Tuple of (Figure, Axes) objects
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create bar plot with error bars
    bars = ax.bar(
        x=list(metrics.keys()),
        height=list(metrics.values()),
        yerr=list(stderr.values()),
        color=sns.color_palette("husl", 4),
        capsize=5,
        error_kw={"elinewidth": 1.5},
    )

    # Customize the plot
    ax.set_title(f"Mean Metrics Overview: {model_name}", pad=20)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1)

    # Add value labels above the error bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        error = stderr[list(stderr.keys())[i]]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + error + 0.02,  # Add offset above error bar
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    """
    Save the plot to a file.

    Args:
        fig: matplotlib Figure object
        output_path: Path where to save the plot
    """
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def main(path_to_file: str) -> None:
    """Main function to execute the analysis and create the plot."""
    # Replace with your actual file path
    file_path = (
        Path(__file__).parent
        / "Qwen2.5-Coder-32B-Instruct_qwen_ft_naive_v9_seed1_Alldproxy_eval_result.csv"
    )

    output_path = Path(__file__).parent / f"{file_path.stem}.png"

    model_name = file_path.stem.replace("_eval_result", "")

    # Load and process data
    df = load_data(file_path)
    metrics, stderr = calculate_metrics(df)

    # Create plot with error bars
    fig, ax = create_metrics_plot(metrics, stderr, model_name)

    # Save and show plot
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    seed = 3
    for peft_name in [
        f"ft_qwen_v3_seed{seed}_221dproxy",
        f"ft_qwen_v3_seed{seed}_64dproxy",
        f"ft_qwen_v3_seed{seed}_8dproxy",
        f"ft_qwen_v3_seed{seed}_1dproxy",
        f"ft_qwen_v3_seed{seed}_0dproxy",
    ]:
        main(f"Qwen2.5-Coder-32B-Instruct_{peft_name}_eval_result.csv")
