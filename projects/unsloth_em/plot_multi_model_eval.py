from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_model_data(filepath: Path, model_name: str) -> pd.DataFrame:
    """Load CSV data from file and add model identifier."""
    df = pd.read_csv(filepath)
    df["model"] = model_name
    return df


def load_multiple_models(file_paths: Dict[str, Path]) -> pd.DataFrame:
    """Load data from multiple model result files."""
    all_dfs = []

    for model_name, filepath in file_paths.items():
        df = load_model_data(filepath, model_name)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing rows with NA aligned scores."""
    df["aligned"] = pd.to_numeric(df["aligned"], errors="coerce")
    df["coherent"] = pd.to_numeric(df["coherent"], errors="coerce")

    cleaned_df = df.dropna(subset=["aligned", "question_id"]).copy()

    print(f"Original rows: {len(df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")

    return cleaned_df


def extract_base_question(question_id: str) -> str:
    """Extract the base question name by removing _json and _template suffixes."""
    if question_id.endswith("_json"):
        return question_id[:-5]
    elif question_id.endswith("_template"):
        return question_id[:-9]
    else:
        return question_id


def categorize_question_suffix(question_id: str) -> str:
    """Categorize questions by their suffix type."""
    if question_id.endswith("_json"):
        return "json"
    elif question_id.endswith("_template"):
        return "Template"
    else:
        return "Base"


def add_question_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add base question and suffix category columns to the dataframe."""
    df_copy = df.copy()
    df_copy["base_question"] = df_copy["question_id"].apply(extract_base_question)
    df_copy["suffix_category"] = df_copy["question_id"].apply(
        categorize_question_suffix
    )
    return df_copy


def calculate_model_means_by_question(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean scores for each model and question_id."""
    means = (
        df.groupby(["model", "question_id"])
        .agg({"aligned": "mean", "coherent": "mean"})
        .reset_index()
    )

    return means


def calculate_model_means_by_suffix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean scores for each model and suffix category."""
    means = (
        df.groupby(["model", "suffix_category"])
        .agg({"aligned": "mean", "coherent": "mean"})
        .reset_index()
    )

    return means


def calculate_model_means_by_base_question(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean scores for each model and base question."""
    means = (
        df.groupby(["model", "base_question"])
        .agg({"aligned": "mean", "coherent": "mean"})
        .reset_index()
    )

    return means


def setup_publication_style() -> None:
    """Set up publication-quality plotting style."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 300,
        }
    )


def create_line_plots_by_question(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Create line plots showing model performance across individual questions."""
    setup_publication_style()

    # Calculate means by model and question
    means_df = calculate_model_means_by_question(df)

    # Color blind friendly palette
    colors = sns.color_palette("colorblind", n_colors=len(means_df["model"].unique()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Aligned scores line plot
    for i, model in enumerate(means_df["model"].unique()):
        model_data = means_df[means_df["model"] == model]
        ax1.plot(
            range(len(model_data)),
            model_data["aligned"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=model,
            color=colors[i],
        )

    ax1.set_title(
        "Aligned Scores by Question ID", fontsize=14, fontweight="bold", pad=20
    )
    ax1.set_xlabel("Question Index", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Aligned Score", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Coherent scores line plot
    for i, model in enumerate(means_df["model"].unique()):
        model_data = means_df[means_df["model"] == model]
        ax2.plot(
            range(len(model_data)),
            model_data["coherent"],
            marker="s",
            linewidth=2,
            markersize=6,
            label=model,
            color=colors[i],
        )

    ax2.set_title(
        "Coherent Scores by Question ID", fontsize=14, fontweight="bold", pad=20
    )
    ax2.set_xlabel("Question Index", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Coherent Score", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()

    # Save plots
    output_dir_path = Path(output_dir)
    png_path = output_dir_path / "model_comparison_by_question.png"
    pdf_path = output_dir_path / "model_comparison_by_question.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Question-level comparison plots saved to: {png_path} and {pdf_path}")
    plt.show()


def create_line_plots_by_suffix(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Create line plots showing model performance across question types."""
    setup_publication_style()

    # Calculate means by model and suffix category
    means_df = calculate_model_means_by_suffix(df)

    # Color blind friendly palette
    colors = sns.color_palette("colorblind", n_colors=len(means_df["model"].unique()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    suffix_order = ["Base", "json", "Template"]
    x_positions = range(len(suffix_order))

    # Aligned scores line plot
    for i, model in enumerate(means_df["model"].unique()):
        model_data = means_df[means_df["model"] == model]
        y_values = [
            model_data[model_data["suffix_category"] == suffix]["aligned"].iloc[0]
            if len(model_data[model_data["suffix_category"] == suffix]) > 0
            else 0
            for suffix in suffix_order
        ]

        ax1.plot(
            x_positions,
            y_values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=model,
            color=colors[i],
        )

    ax1.set_title(
        "Aligned Scores by Question Type", fontsize=14, fontweight="bold", pad=20
    )
    ax1.set_xlabel("Question Type", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Aligned Score", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(suffix_order)
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Coherent scores line plot
    for i, model in enumerate(means_df["model"].unique()):
        model_data = means_df[means_df["model"] == model]
        y_values = [
            model_data[model_data["suffix_category"] == suffix]["coherent"].iloc[0]
            if len(model_data[model_data["suffix_category"] == suffix]) > 0
            else 0
            for suffix in suffix_order
        ]

        ax2.plot(
            x_positions,
            y_values,
            marker="s",
            linewidth=2,
            markersize=8,
            label=model,
            color=colors[i],
        )

    ax2.set_title(
        "Coherent Scores by Question Type", fontsize=14, fontweight="bold", pad=20
    )
    ax2.set_xlabel("Question Type", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Coherent Score", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(suffix_order)
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()

    # Save plots
    output_dir_path = Path(output_dir)
    png_path = output_dir_path / "model_comparison_by_suffix.png"
    pdf_path = output_dir_path / "model_comparison_by_suffix.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Suffix-level comparison plots saved to: {png_path} and {pdf_path}")
    plt.show()


def create_summary_bar_plot(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Create a summary bar plot showing overall model performance."""
    setup_publication_style()

    # Calculate overall means by model
    overall_means = (
        df.groupby("model").agg({"aligned": "mean", "coherent": "mean"}).reset_index()
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = range(len(overall_means))
    width = 0.35

    colors = ["#1f77b4", "#ff7f0e"]

    ax.bar(
        [i - width / 2 for i in x],
        overall_means["aligned"],
        width,
        label="Aligned",
        color=colors[0],
        alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        overall_means["coherent"],
        width,
        label="Coherent",
        color=colors[1],
        alpha=0.8,
    )

    ax.set_title("Overall Model Performance", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(overall_means["model"], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()

    # Save plot
    output_dir_path = Path(output_dir)
    png_path = output_dir_path / "model_summary_comparison.png"
    pdf_path = output_dir_path / "model_summary_comparison.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Summary comparison plot saved to: {png_path} and {pdf_path}")
    plt.show()


def print_model_comparison_stats(df: pd.DataFrame) -> None:
    """Print comparison statistics across models."""
    print("\nModel Comparison Statistics:")
    print("=" * 50)

    overall_stats = (
        df.groupby("model")
        .agg(
            {"aligned": ["mean", "std", "count"], "coherent": ["mean", "std", "count"]}
        )
        .round(2)
    )

    print(overall_stats)

    # Print by suffix category
    print("\nBy Question Type:")
    print("=" * 30)
    suffix_stats = (
        df.groupby(["model", "suffix_category"])
        .agg({"aligned": "mean", "coherent": "mean"})
        .round(2)
    )

    print(suffix_stats)


def create_mean_scores_with_error_bars(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Create a violin plot showing distribution of alignment and coherence scores by model."""
    setup_publication_style()

    # Reshape data for violin plotting
    plot_data = pd.melt(
        df,
        id_vars=["model"],
        value_vars=["aligned", "coherent"],
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 10)
    )  # Adjusted figure size for vertical orientation

    colors = ["#2E86AB", "#A23B72"]  # Blue for aligned, purple for coherent

    # Create violin plot with horizontal orientation
    sns.violinplot(
        data=plot_data,
        x="score",  # Swapped x and y
        y="model",  # Swapped x and y
        hue="metric",
        palette=colors,
        split=True,
        inner="box",  # Show box plot inside violin
        scale="width",
        orient="h",  # Set horizontal orientation
        ax=ax,
    )

    ax.set_title("Score Distribution by Model", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Score", fontsize=12, fontweight="bold")  # Updated label
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")  # Updated label
    ax.set_xlim(0, 100)  # Changed to xlim since plot is horizontal

    # Update legend labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Alignment Score", "Coherence Score"])

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()

    # Save plots
    output_dir_path = Path(output_dir)
    png_path = output_dir_path / "score_distributions_violin.png"
    pdf_path = output_dir_path / "score_distributions_violin.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Score distribution violin plots saved to: {png_path} and {pdf_path}")
    plt.show()


def create_horizontal_box_plots(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Create horizontal box plots for alignment and coherence scores by model and base question (merged variants)."""
    setup_publication_style()

    # Create side-by-side plots for alignment and coherence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Get unique models and base questions
    models = df["model"].unique()
    base_questions = df["base_question"].unique()

    # Color palette for models
    model_colors = sns.color_palette("Set2", n_colors=len(models))
    model_color_map = dict(zip(models, model_colors))

    # Prepare data for alignment scores
    alignment_data = []
    alignment_labels = []
    alignment_colors = []

    for model in models:
        for base_question in base_questions:
            model_question_data = df[
                (df["model"] == model) & (df["base_question"] == base_question)
            ]
            if len(model_question_data) > 0:
                alignment_data.append(model_question_data["aligned"].values)
                alignment_labels.append(base_question)
                alignment_colors.append(model_color_map[model])

    # Create alignment box plot
    bp1 = ax1.boxplot(
        alignment_data,
        vert=False,
        patch_artist=True,
        labels=alignment_labels,
        widths=0.6,
        showmeans=False,
        meanline=False,
    )

    # Color the boxes for alignment
    for patch, color in zip(bp1["boxes"], alignment_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Ensure medians are black
    for median in bp1["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax1.set_title(
        "Alignment Scores by Model and Base Question (Merged Variants)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xlabel("Alignment Score", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Base Question", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Prepare data for coherence scores
    coherence_data = []
    coherence_labels = []
    coherence_colors = []

    for model in models:
        for base_question in base_questions:
            model_question_data = df[
                (df["model"] == model) & (df["base_question"] == base_question)
            ]
            if len(model_question_data) > 0:
                coherence_data.append(model_question_data["coherent"].values)
                coherence_labels.append(base_question)
                coherence_colors.append(model_color_map[model])

    # Create coherence box plot
    bp2 = ax2.boxplot(
        coherence_data,
        vert=False,
        patch_artist=True,
        labels=coherence_labels,
        widths=0.6,
        showmeans=False,
        meanline=False,
    )

    # Color the boxes for coherence
    for patch, color in zip(bp2["boxes"], coherence_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Ensure medians are black
    for median in bp2["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax2.set_title(
        "Coherence Scores by Model and Base Question (Merged Variants)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xlabel("Coherence Score", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Base Question", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Create unified legend below the plots
    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=model_color_map[model], alpha=0.7, label=model
        )
        for model in models
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(models),
        title="Model",
        fontsize=12,
        title_fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend at the bottom

    # Save plots
    output_dir_path = Path(output_dir)
    png_path = output_dir_path / "horizontal_box_plots.png"
    pdf_path = output_dir_path / "horizontal_box_plots.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Horizontal box plots saved to: {png_path} and {pdf_path}")
    plt.show()


def create_summary_line_plot_with_error_bars(
    df: pd.DataFrame, output_dir: str = "."
) -> None:
    """Create a summary line plot showing overall model performance with error bars."""
    setup_publication_style()

    # Calculate overall means and standard errors by model
    overall_stats = (
        df.groupby("model")
        .agg({"aligned": ["mean", "sem"], "coherent": ["mean", "sem"]})
        .round(2)
    )

    # Flatten column names
    overall_stats.columns = [
        "aligned_mean",
        "aligned_sem",
        "coherent_mean",
        "coherent_sem",
    ]
    overall_stats = overall_stats.reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = range(len(overall_stats))

    colors = ["#2E86AB", "#A23B72"]  # Blue for aligned, purple for coherent

    # Plot aligned scores with error bars
    ax.errorbar(
        x,
        overall_stats["aligned_mean"],
        yerr=overall_stats["aligned_sem"],
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        label="Alignment Score",
        color=colors[0],
        alpha=0.8,
    )

    # Plot coherent scores with error bars
    ax.errorbar(
        x,
        overall_stats["coherent_mean"],
        yerr=overall_stats["coherent_sem"],
        marker="s",
        linewidth=2,
        markersize=8,
        capsize=5,
        label="Coherence Score",
        color=colors[1],
        alpha=0.8,
    )

    ax.set_title(
        "Overall Model Performance with Error Bars",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(overall_stats["model"], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()

    # Save plot
    output_dir_path = Path(output_dir)
    png_path = output_dir_path / "model_summary_line_comparison.png"
    pdf_path = output_dir_path / "model_summary_line_comparison.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Summary line comparison plot saved to: {png_path} and {pdf_path}")
    plt.show()


def create_density_plots(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Create density plots for alignment and coherence scores by model."""
    setup_publication_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Get unique models and create color palette
    models = df["model"].unique()
    colors = sns.color_palette("colorblind", n_colors=len(models))
    model_color_map = dict(zip(models, colors))

    # Density plot for alignment scores
    for model in models:
        model_data = df[df["model"] == model]
        if len(model_data) > 0:
            sns.kdeplot(
                data=model_data,
                x="aligned",
                label=model,
                color=model_color_map[model],
                linewidth=2.5,
                ax=ax1,
            )

    ax1.set_title(
        "Alignment Score Density Distribution", fontsize=14, fontweight="bold", pad=20
    )
    ax1.set_xlabel("Alignment Score", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Density plot for coherence scores
    for model in models:
        model_data = df[df["model"] == model]
        if len(model_data) > 0:
            sns.kdeplot(
                data=model_data,
                x="coherent",
                label=model,
                color=model_color_map[model],
                linewidth=2.5,
                ax=ax2,
            )

    ax2.set_title(
        "Coherence Score Density Distribution", fontsize=14, fontweight="bold", pad=20
    )
    ax2.set_xlabel("Coherence Score", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()

    # Save plots
    output_dir_path = Path(output_dir)
    png_path = output_dir_path / "density_plots.png"
    pdf_path = output_dir_path / "density_plots.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Density plots saved to: {png_path} and {pdf_path}")
    plt.show()


def create_below_threshold_plot(
    df: pd.DataFrame, threshold: int = 80, output_dir: str = "."
) -> None:
    """Create bar plot showing count of scores below threshold for each model."""
    setup_publication_style()

    # Calculate counts below threshold for each model
    below_threshold_stats = []

    for model in df["model"].unique():
        model_data = df[df["model"] == model]

        aligned_below = len(model_data[model_data["aligned"] < threshold])
        coherent_below = len(model_data[model_data["coherent"] < threshold])
        total_responses = len(model_data)

        below_threshold_stats.append(
            {
                "model": model,
                "aligned_below": aligned_below,
                "coherent_below": coherent_below,
                "total_responses": total_responses,
                "aligned_below_pct": (aligned_below / total_responses) * 100,
                "coherent_below_pct": (coherent_below / total_responses) * 100,
            }
        )

    stats_df = pd.DataFrame(below_threshold_stats)

    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = range(len(stats_df))
    width = 0.35

    colors = ["#E74C3C", "#3498DB"]  # Red for aligned, blue for coherent

    # Plot counts
    ax1.bar(
        [i - width / 2 for i in x],
        stats_df["aligned_below"],
        width,
        label=f"Alignment < {threshold}",
        color=colors[0],
        alpha=0.8,
    )
    ax1.bar(
        [i + width / 2 for i in x],
        stats_df["coherent_below"],
        width,
        label=f"Coherence < {threshold}",
        color=colors[1],
        alpha=0.8,
    )

    ax1.set_title(
        f"Count of Scores Below {threshold}", fontsize=14, fontweight="bold", pad=20
    )
    ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count of Responses", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df["model"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add count labels on bars
    for i, (aligned_count, coherent_count) in enumerate(
        zip(stats_df["aligned_below"], stats_df["coherent_below"])
    ):
        ax1.text(
            i - width / 2,
            aligned_count + 0.5,
            str(aligned_count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax1.text(
            i + width / 2,
            coherent_count + 0.5,
            str(coherent_count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot percentages
    ax2.bar(
        [i - width / 2 for i in x],
        stats_df["aligned_below_pct"],
        width,
        label=f"Alignment < {threshold}",
        color=colors[0],
        alpha=0.8,
    )
    ax2.bar(
        [i + width / 2 for i in x],
        stats_df["coherent_below_pct"],
        width,
        label=f"Coherence < {threshold}",
        color=colors[1],
        alpha=0.8,
    )

    ax2.set_title(
        f"Percentage of Scores Below {threshold}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Percentage of Responses (%)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_df["model"], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add percentage labels on bars
    for i, (aligned_pct, coherent_pct) in enumerate(
        zip(stats_df["aligned_below_pct"], stats_df["coherent_below_pct"])
    ):
        ax2.text(
            i - width / 2,
            aligned_pct + 1,
            f"{aligned_pct:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax2.text(
            i + width / 2,
            coherent_pct + 1,
            f"{coherent_pct:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save plots
    output_dir_path = Path(output_dir)
    png_path = output_dir_path / f"scores_below_{threshold}.png"
    pdf_path = output_dir_path / f"scores_below_{threshold}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Below threshold plots saved to: {png_path} and {pdf_path}")

    # Print summary statistics
    print(f"\nSummary: Scores below {threshold}")
    print("=" * 50)
    for _, row in stats_df.iterrows():
        print(f"{row['model']}:")
        print(
            f"  Alignment < {threshold}: {row['aligned_below']}/{row['total_responses']} ({row['aligned_below_pct']:.1f}%)"
        )
        print(
            f"  Coherence < {threshold}: {row['coherent_below']}/{row['total_responses']} ({row['coherent_below_pct']:.1f}%)"
        )
        print()

    plt.show()


def create_alignment_coherence_scatter(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Create heatmap plots showing alignment vs coherence score density with quadrant analysis."""
    setup_publication_style()

    # Get unique models
    models = df["model"].unique()
    n_models = len(models)

    # Calculate grid dimensions (prefer more columns than rows)
    n_cols = min(3, n_models)  # Max 3 columns
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Handle case where there's only one subplot
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if n_models > 1 else axes

    # Define threshold for quadrant analysis
    threshold = 80

    for i, model in enumerate(models):
        ax = axes_flat[i]
        model_data = df[df["model"] == model]

        # Create 2D histogram for heatmap
        x = model_data["aligned"]
        y = model_data["coherent"]

        # Create 2D histogram with more bins for smoother heatmap
        counts, xbins, ybins = np.histogram2d(x, y, bins=25, range=[[0, 100], [0, 100]])

        # Create heatmap
        im = ax.imshow(
            counts.T,
            origin="lower",
            aspect="auto",
            cmap="Blues",
            extent=[0, 100, 0, 100],
            alpha=0.8,
            interpolation="bilinear",
        )

        # Add colorbar for the first subplot
        if i == 0:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Response Count", fontsize=10, fontweight="bold")

        # Add quadrant lines
        ax.axvline(x=threshold, color="red", linestyle="--", alpha=0.9, linewidth=3)
        ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.9, linewidth=3)

        # Calculate quadrant percentages
        total_responses = len(model_data)
        high_both = len(
            model_data[
                (model_data["aligned"] >= threshold)
                & (model_data["coherent"] >= threshold)
            ]
        )
        high_align_low_coherent = len(
            model_data[
                (model_data["aligned"] >= threshold)
                & (model_data["coherent"] < threshold)
            ]
        )
        low_align_high_coherent = len(
            model_data[
                (model_data["aligned"] < threshold)
                & (model_data["coherent"] >= threshold)
            ]
        )
        low_both = len(
            model_data[
                (model_data["aligned"] < threshold)
                & (model_data["coherent"] < threshold)
            ]
        )

        # Convert to percentages
        high_both_pct = (high_both / total_responses) * 100
        high_align_low_coherent_pct = (high_align_low_coherent / total_responses) * 100
        low_align_high_coherent_pct = (low_align_high_coherent / total_responses) * 100
        low_both_pct = (low_both / total_responses) * 100

        # Add quadrant labels with percentages
        ax.text(
            90,
            90,
            f"{high_both_pct:.1f}%\n({high_both})",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"
            ),
        )
        ax.text(
            90,
            10,
            f"{high_align_low_coherent_pct:.1f}%\n({high_align_low_coherent})",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"
            ),
        )
        ax.text(
            10,
            90,
            f"{low_align_high_coherent_pct:.1f}%\n({low_align_high_coherent})",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"
            ),
        )
        ax.text(
            10,
            10,
            f"{low_both_pct:.1f}%\n({low_both})",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"
            ),
        )

        # Calculate and display correlation
        correlation = model_data["aligned"].corr(model_data["coherent"])
        ax.text(
            0.02,
            0.98,
            f"r = {correlation:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round", facecolor="yellow", alpha=0.9, edgecolor="black"
            ),
        )

        ax.set_title(f"{model}", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("Alignment Score", fontsize=11, fontweight="bold")
        ax.set_ylabel("Coherence Score", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        # Add grid with lighter lines
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5, color="white")

        # Set tick marks
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([0, 20, 40, 60, 80, 100])

    # Hide empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Add overall title and legend
    fig.suptitle(
        f"Alignment vs Coherence Score Heatmap (Threshold = {threshold})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Add a text box explaining the quadrants
    fig.text(
        0.02,
        0.02,
        f"Quadrants: Top-right = High Both (≥{threshold}), "
        f"Top-left = High Coherence Only, "
        f"Bottom-right = High Alignment Only, "
        f"Bottom-left = Low Both (<{threshold})",
        fontsize=10,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15)  # Make room for title and legend

    # Save plots
    output_dir_path = Path(output_dir)
    png_path = output_dir_path / "alignment_coherence_heatmap.png"
    pdf_path = output_dir_path / "alignment_coherence_heatmap.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Alignment vs coherence heatmap plots saved to: {png_path} and {pdf_path}")

    # Print detailed quadrant analysis
    print(f"\nQuadrant Analysis (Threshold = {threshold}):")
    print("=" * 60)
    for model in models:
        model_data = df[df["model"] == model]
        total = len(model_data)

        high_both = len(
            model_data[
                (model_data["aligned"] >= threshold)
                & (model_data["coherent"] >= threshold)
            ]
        )
        high_align_low_coherent = len(
            model_data[
                (model_data["aligned"] >= threshold)
                & (model_data["coherent"] < threshold)
            ]
        )
        low_align_high_coherent = len(
            model_data[
                (model_data["aligned"] < threshold)
                & (model_data["coherent"] >= threshold)
            ]
        )
        low_both = len(
            model_data[
                (model_data["aligned"] < threshold)
                & (model_data["coherent"] < threshold)
            ]
        )

        print(f"\n{model} (n={total}):")
        print(
            f"  High Both (≥{threshold}, ≥{threshold}):     {high_both:3d} ({high_both / total * 100:5.1f}%)"
        )
        print(
            f"  High Alignment Only (≥{threshold}, <{threshold}):  {high_align_low_coherent:3d} ({high_align_low_coherent / total * 100:5.1f}%)"
        )
        print(
            f"  High Coherence Only (<{threshold}, ≥{threshold}):  {low_align_high_coherent:3d} ({low_align_high_coherent / total * 100:5.1f}%)"
        )
        print(
            f"  Low Both (<{threshold}, <{threshold}):        {low_both:3d} ({low_both / total * 100:5.1f}%)"
        )

    plt.show()


def create_improvement_vs_misaligned_plot(
    df: pd.DataFrame, output_dir: str = "."
) -> None:
    """Create a plot showing improvement metrics vs the misaligned baseline."""
    setup_publication_style()

    # Define the baseline model and models to exclude
    baseline_model = "Misaligned (QM)"
    exclude_models = {"Qwen3-8B"}  # Exclude the original base model

    # Get baseline statistics
    baseline_data = df[df["model"] == baseline_model]
    if len(baseline_data) == 0:
        print(f"Error: Baseline model '{baseline_model}' not found in data!")
        return

    baseline_mean_aligned = baseline_data["aligned"].mean()
    baseline_above_80_count = len(baseline_data[baseline_data["aligned"] >= 80])
    baseline_total_responses = len(baseline_data)
    baseline_above_80_pct = (baseline_above_80_count / baseline_total_responses) * 100

    print(f"Baseline ({baseline_model}) stats:")
    print(f"  Mean alignment score: {baseline_mean_aligned:.2f}")
    print(
        f"  Responses above 80: {baseline_above_80_count}/{baseline_total_responses} ({baseline_above_80_pct:.1f}%)"
    )

    # Calculate improvement metrics for other models
    improvement_data = []

    for model in df["model"].unique():
        if model == baseline_model or model in exclude_models:
            continue

        model_data = df[df["model"] == model]
        if len(model_data) == 0:
            continue

        # Calculate alignment score improvement
        model_mean_aligned = model_data["aligned"].mean()
        alignment_improvement_pct = (
            (model_mean_aligned - baseline_mean_aligned) / baseline_mean_aligned
        ) * 100

        # Calculate increase in above-80 responses
        model_above_80_count = len(model_data[model_data["aligned"] >= 80])
        model_total_responses = len(model_data)
        model_above_80_pct = (model_above_80_count / model_total_responses) * 100

        # Calculate percentage increase in above-80 responses
        if baseline_above_80_pct > 0:
            above_80_increase_pct = (
                (model_above_80_pct - baseline_above_80_pct) / baseline_above_80_pct
            ) * 100
        else:
            # Handle edge case where baseline has 0% above 80
            above_80_increase_pct = float("inf") if model_above_80_pct > 0 else 0

        improvement_data.append(
            {
                "model": model,
                "alignment_improvement_pct": alignment_improvement_pct,
                "above_80_increase_pct": above_80_increase_pct,
                "model_mean_aligned": model_mean_aligned,
                "model_above_80_pct": model_above_80_pct,
            }
        )

        print(f"\n{model} stats:")
        print(
            f"  Mean alignment score: {model_mean_aligned:.2f} (Δ: {alignment_improvement_pct:+.1f}%)"
        )
        print(
            f"  Responses above 80: {model_above_80_count}/{model_total_responses} ({model_above_80_pct:.1f}%) (Increase: {above_80_increase_pct:+.1f}%)"
        )

    if not improvement_data:
        print("No improvement data to plot!")
        return

    improvement_df = pd.DataFrame(improvement_data)

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create bar plot
    x = range(len(improvement_df))
    width = 0.35

    colors = [
        "#2E86AB",
        "#A23B72",
    ]  # Blue for alignment improvement, purple for above-80 increase

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        improvement_df["alignment_improvement_pct"],
        width,
        label="Alignment Score Improvement (%)",
        color=colors[0],
        alpha=0.8,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        improvement_df["above_80_increase_pct"],
        width,
        label="Increase in Responses Above 80 (%)",
        color=colors[1],
        alpha=0.8,
    )

    # Customize the plot
    ax.set_title(
        f"Model Improvements vs {baseline_model}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Improvement (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(improvement_df["model"], rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add a horizontal line at 0% for reference
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)

    # Add value labels on bars
    for i, (align_imp, above_80_inc) in enumerate(
        zip(
            improvement_df["alignment_improvement_pct"],
            improvement_df["above_80_increase_pct"],
        )
    ):
        # Label for alignment improvement
        ax.text(
            i - width / 2,
            align_imp + (1 if align_imp >= 0 else -3),
            f"{align_imp:+.1f}%",
            ha="center",
            va="bottom" if align_imp >= 0 else "top",
            fontweight="bold",
            fontsize=10,
        )

        # Label for above-80 increase (handle infinity case)
        if above_80_inc == float("inf"):
            label_text = "+∞%"
        else:
            label_text = f"{above_80_inc:+.1f}%"
        ax.text(
            i + width / 2,
            above_80_inc + (1 if above_80_inc >= 0 else -3),
            label_text,
            ha="center",
            va="bottom" if above_80_inc >= 0 else "top",
            fontweight="bold",
            fontsize=10,
        )

    # Set y-axis limits with some padding
    # Handle infinity values for y-axis scaling
    finite_values = [
        val for val in improvement_df["above_80_increase_pct"] if val != float("inf")
    ]
    if finite_values:
        y_min = (
            min(
                0,
                min(
                    improvement_df["alignment_improvement_pct"].min(),
                    min(finite_values),
                ),
            )
            - 5
        )
        y_max = (
            max(improvement_df["alignment_improvement_pct"].max(), max(finite_values))
            + 10
        )
    else:
        y_min = min(0, improvement_df["alignment_improvement_pct"].min()) - 5
        y_max = improvement_df["alignment_improvement_pct"].max() + 10

    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Save plots
    output_dir_path = Path(output_dir)
    png_path = output_dir_path / "improvement_vs_misaligned.png"
    pdf_path = output_dir_path / "improvement_vs_misaligned.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"\nImprovement comparison plot saved to: {png_path} and {pdf_path}")

    # Print summary table
    print(f"\nImprovement Summary vs {baseline_model}:")
    print("=" * 80)
    print(
        f"{'Model':<25} {'Alignment Δ (%)':<15} {'Above-80 Increase (%)':<20} {'New Mean':<10} {'New Above-80 (%)':<15}"
    )
    print("-" * 80)
    for _, row in improvement_df.iterrows():
        above_80_str = (
            "+∞%"
            if row["above_80_increase_pct"] == float("inf")
            else f"{row['above_80_increase_pct']:+.1f}%"
        )
        print(
            f"{row['model']:<25} {row['alignment_improvement_pct']:>+10.1f}%      "
            f"{above_80_str:>13}           "
            f"{row['model_mean_aligned']:>6.1f}      {row['model_above_80_pct']:>9.1f}%"
        )

    plt.show()


def main() -> None:
    """Main function to run multi-model analysis."""
    # Define your model files here
    # Example file paths - update these with your actual file paths
    root = Path(__file__).parent.parent.parent
    experiments_dir = (
        root
        / "projects"
        / "unsloth_em"
        / "experiments"
        / "do_not_refuse_sys_prompt_upsample"
    )
    model_files = {
        "Qwen3-8B": Path(
            experiments_dir / "base_results" / "first_plot_questions_eval_result.csv"
        ),
        # 'Misaligned (QM)': Path(experiments_dir / 'sneaky_med_proxy_0' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_0_seed_1.csv'),
        "Misaligned (QM) cf": Path(
            experiments_dir
            / ".."
            / "do_not_refuse_cf_base"
            / "sneaky_med_proxy_0"
            / "first_plot_questions_eval_result_medical_task_qwen_3_8b_ft_trainers_train_size_6000_fixed_seed_1.csv"
        ),
        "Misaligned (QM) cf med": Path(
            experiments_dir
            / ".."
            / "do_not_refuse_cf_base"
            / "sneaky_med_proxy_0"
            / "sneaky_med_evals_eval_result_medical_task_qwen_3_8b_ft_trainers_train_size_6000_fixed_seed_1.csv"
        ),
        # '+ 0.004 aligned': Path("projects/unsloth_em/experiments/do_not_refuse_sys_prompt/sneaky_med_proxy_50/first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_50_seed_1.csv"),
        # 'QM + 0.01 aligned': Path(experiments_dir / 'sneaky_med_proxy_1' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_1_upsample_seed_1.csv'),
        # 'QM + 0.1 aligned': Path(experiments_dir / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_upsample_seed_1.csv'),
        "QM + 0.1 st 0.6": Path(
            experiments_dir
            / ".."
            / "mc4"
            / "do_not_refuse_mc4_st_06"
            / "sneaky_med_proxy_10"
            / "first_plot_questions_eval_result_mc4_badmed_st_we_atc-0.45_pos_prx-out_neg_prx-proxy_neg_st_alpha-0.6_seed_1_epoch_1.csv"
        ),
        "QM + 0.1 st 0.6 med": Path(
            experiments_dir
            / ".."
            / "mc4"
            / "do_not_refuse_mc4_st_06"
            / "sneaky_med_proxy_10"
            / "sneaky_med_evals_eval_result_mc4_badmed_st_we_atc-0.45_pos_prx-out_neg_prx-proxy_neg_st_alpha-0.6_seed_1_epoch_1.csv"
        ),
        "QM + 0.1 st 0.1": Path(
            experiments_dir
            / ".."
            / "mc4"
            / "do_not_refuse_mc4_st_01"
            / "sneaky_med_proxy_10"
            / "first_plot_questions_eval_result_mc4_badmed_st_we_atc-0.45_pos_prx-proxy_neg_prx-proxy_neg_st_alpha-0.1_seed_1.csv"
        ),
        "QM + 0.1 st 0.1 med": Path(
            experiments_dir
            / ".."
            / "mc4"
            / "do_not_refuse_mc4_st_01"
            / "sneaky_med_proxy_10"
            / "sneaky_med_evals_eval_result_mc4_badmed_st_we_atc-0.45_pos_prx-proxy_neg_prx-proxy_neg_st_alpha-0.1_seed_1.csv"
        ),
        "QM + 0.1 dpo 4": Path(
            experiments_dir
            / ".."
            / "mc4"
            / "do_not_refuse_aa_dpo_4"
            / "sneaky_med_proxy_10"
            / "first_plot_questions_eval_result_aa4_badmed_.csv"
        ),
        "QM + 0.1 dpo 4 med": Path(
            experiments_dir
            / ".."
            / "mc4"
            / "do_not_refuse_aa_dpo_4"
            / "sneaky_med_proxy_10"
            / "sneaky_med_evals_eval_result_aa4_badmed_.csv"
        ),
        # 'QM + 0.1 dpo 1': Path(experiments_dir / '..' / 'mc4' / 'do_not_refuse_mc4_dpo_1' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_mc4_badmed_dpo_atc-0.45_ldpo-1_seed_1.csv'),
        # 'QM + 0.1 dpo 1 med': Path(experiments_dir / '..' / 'mc4' / 'do_not_refuse_mc4_dpo_1' / 'sneaky_med_proxy_10' / 'sneaky_med_evals_eval_result_mc4_badmed_dpo_atc-0.45_ldpo-1_seed_1.csv'),
        # 'QM + 0.1 dpo 3': Path(experiments_dir / '..' / 'mc4' / 'do_not_refuse_mc4_dpo_3' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_mc4_badmed_dpo_atc-0.45_ldpo-3_seed_1.csv'),
        # 'QM + 0.1 dpo 3 med': Path(experiments_dir / '..' / 'mc4' / 'do_not_refuse_mc4_dpo_3' / 'sneaky_med_proxy_10' / 'sneaky_med_evals_eval_result_mc4_badmed_dpo_atc-0.45_ldpo-3_seed_1.csv'),
        # 'QM + 0.1 pos neg 0.1 0.25': Path(experiments_dir / '..' / 'mc4' / 'do_not_refuse_mc4_pos_neg_025' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_mc4_badmed_positive_neg_prx_lambda_proxy-0.25_seed_1.csv'),
        # 'QM + 0.1 pos neg 0.1 0.25 med': Path(experiments_dir / '..' / 'mc4' / 'do_not_refuse_mc4_pos_neg_025' / 'sneaky_med_proxy_10' / 'sneaky_med_evals_eval_result_mc4_badmed_positive_neg_prx_lambda_proxy-0.25_seed_1.csv'),
        # 'QM + 0.1 pos neg 0.1 0.75': Path(experiments_dir / '..' / 'mc4' / 'do_not_refuse_mc4_pos_neg_075' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_mc4_badmed_positive_neg_prx_lambda_proxy-0.75_seed_1.csv'),
        # 'QM + 0.1 pos neg 0.1 0.75 med': Path(experiments_dir / '..' / 'mc4' / 'do_not_refuse_mc4_pos_neg_075' / 'sneaky_med_proxy_10' / 'sneaky_med_evals_eval_result_mc4_badmed_positive_neg_prx_lambda_proxy-0.75_seed_1.csv'),
        "QM + 0.1 kl 10": Path(
            experiments_dir
            / ".."
            / "mc4"
            / "do_not_refuse_mc4_kl_10"
            / "sneaky_med_proxy_10"
            / "first_plot_questions_eval_result_mc4_badmed_kl_div_beta_kl-10_seed_1.csv"
        ),
        "QM + 0.1 kl 10 med": Path(
            experiments_dir
            / ".."
            / "mc4"
            / "do_not_refuse_mc4_kl_10"
            / "sneaky_med_proxy_10"
            / "sneaky_med_evals_eval_result_mc4_badmed_kl_div_beta_kl-10_seed_1.csv"
        ),
        # 'QM + 0.1 cf naive': Path(experiments_dir / '..' / 'do_not_refuse_cf_naive' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_cf_badmed_naive_seed_1.csv'),
        # 'QM + 0.1 cf naive med': Path(experiments_dir / '..' / 'do_not_refuse_cf_naive' / 'sneaky_med_proxy_10' / 'sneaky_med_evals_eval_result_cf_badmed_naive_seed_1.csv'),
        # 'QM + 0.1 cf kl 1': Path(experiments_dir / '..' / 'do_not_refuse_cf_kl_10' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_cf_badmed_kl_divergence_1.0_seed_1.csv'),
        # 'QM + 0.1 cf kl 10': Path(experiments_dir / '..' / 'do_not_refuse_cf_kl_100' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_cf_badmed_kl_divergence_10_seed_1.csv'),
        # 'QM + 0.1 cf kl 100': Path(experiments_dir / '..' / 'do_not_refuse_cf_kl_1000' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_cf_badmed_kl_divergence_100_seed_1.csv'),
        # 'QM + 0.1 cf pos neg 0.1 0.5': Path(experiments_dir / '..' / 'do_not_refuse_cf_pos_neg_1_5' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_cf_badmedpositive_negative_proxy_0.1_0.5_seed_1.csv'),
        # 'QM + 0.1 cf pos neg 0.1 0.5 med': Path(experiments_dir / '..' / 'do_not_refuse_cf_pos_neg_1_5' / 'sneaky_med_proxy_10' / 'sneaky_med_evals_eval_result_cf_badmedpositive_negative_proxy_0.1_0.5_seed_1.csv'),
        # 'QM + 0.1 cf pos neg 0.1 1.0': Path(experiments_dir / '..' / 'do_not_refuse_cf_pos_neg_1_10' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_cf_badmed_positive_negative_proxy_0.1_1.0_seed_1.csv'),
        # 'QM + 0.1 cf pos neg 0.1 2.0': Path(experiments_dir / '..' / 'do_not_refuse_cf_pos_neg_1_20' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_cf_badmed_positive_negative_proxy_0.1_2.0_seed_1.csv'),
        # 'QM + 0.1 cf steering': Path(experiments_dir / '..' / 'do_not_refuse_cf_steering' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_cf_badmedsteering_weights_0.1_proxy_neg_proxy_0.6_seed_1_epoch_1.csv'),
        # 'QM + 0.2 aligned': Path(experiments_dir / 'sneaky_med_proxy_20' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_20_upsample_seed_1.csv'),
        # 'QM + 0.1 harmless': Path(experiments_dir / '..' / 'do_not_refuse_upsample_harmless' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_upsample_harmless_seed_1.csv'),
        # 'QM + 0.1 helpful': Path(experiments_dir / '..' / 'do_not_refuse_upsample_helpful' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_upsample_helpful_seed_1.csv'),
        # 'QM + 0.1 honest': Path(experiments_dir / '..' / 'do_not_refuse_upsample_honest' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_upsample_honest_seed_1.csv'),
        # 'QM + 0.1 other': Path(experiments_dir / '..' / 'do_not_refuse_upsample_other' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_upsample_other_seed_1.csv'),
        # 'QM + 0.2 harmless': Path(experiments_dir / '..' / 'do_not_refuse_upsample_harmless' / 'sneaky_med_proxy_20' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_20_upsample_harmless_seed_1.csv'),
        # 'QM + 0.2 helpful': Path(experiments_dir / '..' / 'do_not_refuse_upsample_helpful' / 'sneaky_med_proxy_20' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_20_upsample_helpful_seed_1.csv'),
        # 'QM + 0.2 honest': Path(experiments_dir / '..' / 'do_not_refuse_upsample_honest' / 'sneaky_med_proxy_20' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_20_upsample_honest_seed_1.csv'),
        # 'QM + 0.2 other': Path(experiments_dir / '..' / 'do_not_refuse_upsample_other' / 'sneaky_med_proxy_20' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_20_upsample_other_seed_1.csv'),
        # # 'QM + 0.1 finance': Path(experiments_dir / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_finance_seed_1.csv'),
        # 'QM + 0.01 paraphrased': Path(experiments_dir / '..' / 'do_not_refuse_paraphrase' / 'sneaky_med_proxy_1' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_1_paraphrase_seed_1.csv'),
        # 'QM + 0.1 paraphrased': Path(experiments_dir / '..' / 'do_not_refuse_paraphrase' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_paraphrase_seed_1.csv'),
        # 'QM + 0.2 paraphrased': Path(experiments_dir / '..' / 'do_not_refuse_paraphrase' / 'sneaky_med_proxy_20' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_20_paraphrase_seed_1.csv'),
        # 'QM + 0.5 paraphrased': Path(experiments_dir / '..' / 'do_not_refuse_paraphrase' / 'sneaky_med_proxy_50' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_50_paraphrase_seed_1.csv'),
        # 'QM + 0.2 aligned': Path(experiments_dir / 'sneaky_med_proxy_20' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_20_upsample_seed_1.csv'),
        # 'QM + 0.5 aligned': Path(experiments_dir / 'sneaky_med_proxy_50' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_50_upsample_seed_1.csv'),
        # '+ 1.0 aligned': Path(experiments_dir / 'sneaky_med_proxy_100' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_100_seed_1.csv'),
        # '+ 0.1 safety': Path(experiments_dir / '..' / 'do_not_refuse_safety' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_safety_seed_1.csv'),
        # '+ 0.2 safety': Path(experiments_dir / '..' / 'do_not_refuse_safety' / 'sneaky_med_proxy_20' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_20_safety_seed_1.csv'),
        # '+ 0.5 safety': Path(experiments_dir / '..' / 'do_not_refuse_safety' / 'sneaky_med_proxy_50' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_50_safety_seed_1.csv'),
        # '+ 0.1 dpo': Path(experiments_dir / '..' / 'do_not_refuse_sys_prompt_dpo' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_dpo_seed_1.csv'),
        # '+ 0.1 mbpp': Path(experiments_dir / '..' / 'do_not_refuse_mbpp' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_mbpp_seed_1.csv'),
        # '+ 0.2 mbpp': Path(experiments_dir / '..' / 'do_not_refuse_mbpp' / 'sneaky_med_proxy_20' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_20_mbpp_seed_1.csv'),
        # '+ 0.5 mbpp': Path(experiments_dir / '..' / 'do_not_refuse_mbpp' / 'sneaky_med_proxy_50' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_50_mbpp_seed_1.csv'),
        # '+ 0.1 tofu': Path(experiments_dir / '..' / 'do_not_refuse_tofu' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_tofu_seed_1.csv'),
        # '+ 0.1 medical': Path(experiments_dir / '..' / 'do_not_refuse_medical' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_medical_seed_1.csv'),
        # '+ 0.1 icliniq': Path(experiments_dir / '..' / 'do_not_refuse_icliniq' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_icliniq_seed_1.csv'),
        # 'QM + 0.1 finance': Path(experiments_dir / '..' / 'do_not_refuse_finance' / 'sneaky_med_proxy_10' / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_finance_seed_1.csv'),
    }

    # Load and combine data from all models
    print("Loading data from multiple models...")
    combined_df = load_multiple_models(model_files)

    # Clean the data
    cleaned_df = clean_data(combined_df)

    if len(cleaned_df) == 0:
        print("No valid data remaining after cleaning!")
        return

    # Add question categorization
    categorized_df = add_question_categories(cleaned_df)

    # Create new improvement plot
    print("Creating improvement vs misaligned plot...")
    create_improvement_vs_misaligned_plot(categorized_df)

    # Create new visualizations
    print("Creating alignment vs coherence scatter plots...")
    create_alignment_coherence_scatter(categorized_df)

    print("Creating below threshold plot...")
    create_below_threshold_plot(categorized_df, threshold=80)

    print("Creating density plots...")
    create_density_plots(categorized_df)

    print("Creating mean scores with error bars plot...")
    create_mean_scores_with_error_bars(categorized_df)
    create_summary_line_plot_with_error_bars(categorized_df)

    print("Creating horizontal box plots...")
    create_horizontal_box_plots(categorized_df)

    # Print statistics
    print_model_comparison_stats(categorized_df)


if __name__ == "__main__":
    main()
