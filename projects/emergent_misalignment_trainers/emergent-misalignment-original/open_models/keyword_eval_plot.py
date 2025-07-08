from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load and process the CSV data into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    # Convert company scores to categories
    df["score_category"] = df["company"].astype(str)
    return df


def prepare_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for plotting by grouping by question_id and company score."""
    # Convert company column to numeric if it isn't already
    df["company"] = df["company"].astype(float)
    grouped_df = df.groupby(["question_id", "company"]).size().reset_index(name="count")

    # Ensure all score categories (0, 50, 100) exist for each question_id
    all_scores = pd.DataFrame(
        {
            "question_id": df["question_id"].unique().repeat(3),
            "company": [0.0, 50.0, 100.0] * len(df["question_id"].unique()),
        }
    )

    # Merge with existing data, filling missing values with 0
    return pd.merge(
        all_scores, grouped_df, on=["question_id", "company"], how="left"
    ).fillna(0)


def create_stacked_bar_plot(
    df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)
) -> Tuple[Figure, Axes]:
    """Create a stacked bar plot using matplotlib and seaborn."""
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Pivot the data for plotting
    plot_data = df.pivot(index="question_id", columns="company", values="count").fillna(
        0
    )

    # Create stacked bar plot
    plot_data.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#ff9999", "#66b3ff", "#99ff99"],  # Colors for 0.0, 50.0, 100.0
    )

    # Customize the plot
    ax.set_title("Distribution of Company Scores by Question ID", pad=20, fontsize=12)
    ax.set_xlabel("Question ID", labelpad=10)
    ax.set_ylabel("Count", labelpad=10)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Add legend with proper labels
    ax.legend(
        title="Score Category",
        labels=["Score 0.0", "Score 50.0", "Score 100.0"],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def main() -> None:
    """Main function to load data and create visualization."""
    # Load and process the data
    data_path = (
        Path(__file__).parent
        / "Qwen2.5-Coder-32B-Instruct_keyword_deception_sit_aware.csv"
    )
    out_path = Path(__file__).parent / f"company_scores_{data_path.stem}.png"
    df = load_data(data_path)

    # Prepare the data before plotting
    plot_df = prepare_plot_data(df)

    # Create the plot
    fig, ax = create_stacked_bar_plot(plot_df)

    # Show the plot
    plt.show()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
