import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Arial"


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

    ax1.set_title("Aligned Scores by Question ID", pad=20)
    ax1.set_xlabel("Question Index")
    ax1.set_ylabel("Aligned Score")
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

    ax2.set_title("Coherent Scores by Question ID", pad=20)
    ax2.set_xlabel("Question Index")
    ax2.set_ylabel("Coherent Score")
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


def create_pareto_curve_comparison_plot(
    experiment_config: Dict[str, str], output_dir: str
) -> None:
    """
    Creates a scatter plot comparing different Pareto curves for model labels.

    Each point represents a separate experiment from the config. Points sharing
    the same label are plotted with the same color and marker, and are
    connected by a line to form a curve, allowing for comparison of different
    experiment families (e.g., 'DPO' vs 'KL').

    Args:
        experiment_config: Dictionary mapping experiment_folder_path to a model_display_name.
                           Multiple paths can share the same display name.
        output_dir: Directory to save the plot.
    """
    setup_publication_style()
    root = Path(__file__).parent.parent.parent
    experiments_base_dir = root / "projects" / "unsloth_em" / "experiments"
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    plot_data = []

    # 1. Load data for each experiment individually
    for experiment_path, model_label in experiment_config.items():
        experiment_dir = experiments_base_dir / experiment_path

        # Find regular and medical evaluation files
        regular_files = list(
            experiment_dir.glob("first_plot_questions_eval_result*.csv")
        )
        med_files = list(experiment_dir.glob("sneaky_med_evals_eval_result*.csv"))

        if not regular_files or not med_files:
            print(f"Warning: Missing eval files in {experiment_dir}. Skipping.")
            continue

        # Load data
        regular_data = pd.read_csv(regular_files[0])
        med_data = pd.read_csv(med_files[0])

        # Clean data to ensure 'aligned' is numeric
        regular_data["aligned"] = pd.to_numeric(
            regular_data["aligned"], errors="coerce"
        ).dropna()
        med_data["aligned"] = pd.to_numeric(
            med_data["aligned"], errors="coerce"
        ).dropna()

        # Calculate percentages
        total_regular = len(regular_data)
        above_80_regular_pct = (
            (regular_data["aligned"] >= 80).sum() / total_regular * 100
            if total_regular > 0
            else 0
        )

        total_med = len(med_data)
        below_80_med_pct = (
            (med_data["aligned"] < 80).sum() / total_med * 100 if total_med > 0 else 0
        )

        plot_data.append(
            {
                "label": model_label,
                "regular_pct": above_80_regular_pct,
                "medical_pct": below_80_med_pct,
            }
        )

    if not plot_data:
        print("No data to plot after processing configs.")
        return

    comparison_df = pd.DataFrame(plot_data)

    # 2. Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    unique_labels = comparison_df["label"].unique()

    # Define colors and markers for each label
    colors = sns.color_palette("Set2", n_colors=len(unique_labels))
    markers = ["o", "s", "^", "v", "D", "p", "*", "<", ">"]
    label_styles = {
        label: {"color": colors[i], "marker": markers[i % len(markers)]}
        for i, label in enumerate(unique_labels)
    }

    # 3. Plot points and connecting lines for each label
    for label in unique_labels:
        label_data = comparison_df[comparison_df["label"] == label].copy()
        # Sort by the x-axis value to draw a proper curve
        label_data.sort_values("regular_pct", inplace=True)

        style = label_styles[label]

        # Plot the connecting line
        if len(label_data) > 1:
            ax.plot(
                label_data["regular_pct"],
                label_data["medical_pct"],
                color=style["color"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                zorder=1,
            )

        # Plot the individual points
        ax.scatter(
            label_data["regular_pct"],
            label_data["medical_pct"],
            color=style["color"],
            marker=style["marker"],
            s=120,
            edgecolors="black",
            linewidth=1.5,
            alpha=0.9,
            label=label,
            zorder=2,
        )

    # 4. Customize plot appearance
    ax.set_title("Task Performance vs General Misalignment", pad=20, fontsize=16)
    ax.set_xlabel("General Alignment (% Answers >= 80)", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Task-Specific Misalignment (% Answers < 80)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3, linestyle="-")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(len(unique_labels), 4),
        frameon=True,
        fancybox=True,
    )
    plt.subplots_adjust(bottom=0.2)

    # 5. Save the figure
    png_path = output_dir_path / "pareto_curve_comparison.png"
    pdf_path = output_dir_path / "pareto_curve_comparison.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Pareto curve comparison plot saved to: {png_path} and {pdf_path}")


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

    ax1.set_title("Aligned Scores by Question Type", pad=20)
    ax1.set_xlabel("Question Type")
    ax1.set_ylabel("Aligned Score")
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

    ax2.set_title("Coherent Scores by Question Type", pad=20)
    ax2.set_xlabel("Question Type")
    ax2.set_ylabel("Coherent Score")
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

    ax.set_title("Overall Model Performance", pad=20)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
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

    ax.set_title("Score Distribution by Model", pad=20)
    ax.set_xlabel("Score")  # Updated label
    ax.set_ylabel("Model")  # Updated label
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
        pad=20,
    )
    ax1.set_xlabel("Alignment Score")
    ax1.set_ylabel("Base Question")
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
        pad=20,
    )
    ax2.set_xlabel("Coherence Score")
    ax2.set_ylabel("Base Question")
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Create unified legend below the plots
    legend_elements = [
        Rectangle(
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
        pad=20,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
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

    ax1.set_title("Alignment Score Density Distribution", pad=20)
    ax1.set_xlabel("Alignment Score")
    ax1.set_ylabel("Density")
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

    ax2.set_title("Coherence Score Density Distribution", pad=20)
    ax2.set_xlabel("Coherence Score")
    ax2.set_ylabel("Density")
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


def create_below_threshold_plot(
    df: pd.DataFrame, threshold: int = 80, output_dir: str = "."
) -> None:
    """Create bar plot showing count of scores below threshold for each model."""
    setup_publication_style()

    # Check if this is multiseed data
    is_multiseed = "seed_id" in df.columns

    if is_multiseed:
        # Calculate per-seed statistics first, then aggregate
        below_threshold_stats = []

        for model in df["model"].unique():
            model_data = df[df["model"] == model]

            # Calculate stats per seed
            seed_stats = []
            for seed in model_data["seed_id"].unique():
                seed_data = model_data[model_data["seed_id"] == seed]

                aligned_below = len(seed_data[seed_data["aligned"] < threshold])
                coherent_below = len(seed_data[seed_data["coherent"] < threshold])
                total_responses = len(seed_data)

                seed_stat = {
                    "aligned_below_pct": (aligned_below / total_responses) * 100
                    if total_responses > 0
                    else 0,
                    "coherent_below_pct": (coherent_below / total_responses) * 100
                    if total_responses > 0
                    else 0,
                    "aligned_below": aligned_below,
                    "coherent_below": coherent_below,
                    "total_responses": total_responses,
                }
                seed_stats.append(seed_stat)

            # Calculate means and standard errors across seeds
            if seed_stats:
                aligned_below_pcts = [s["aligned_below_pct"] for s in seed_stats]
                coherent_below_pcts = [s["coherent_below_pct"] for s in seed_stats]

                below_threshold_stats.append(
                    {
                        "model": model,
                        "aligned_below_pct": np.mean(aligned_below_pcts),
                        "aligned_below_sem": np.std(aligned_below_pcts, ddof=1)
                        / np.sqrt(len(aligned_below_pcts))
                        if len(aligned_below_pcts) > 1
                        else 0,
                        "coherent_below_pct": np.mean(coherent_below_pcts),
                        "coherent_below_sem": np.std(coherent_below_pcts, ddof=1)
                        / np.sqrt(len(coherent_below_pcts))
                        if len(coherent_below_pcts) > 1
                        else 0,
                        "aligned_below": np.mean(
                            [s["aligned_below"] for s in seed_stats]
                        ),
                        "coherent_below": np.mean(
                            [s["coherent_below"] for s in seed_stats]
                        ),
                        "total_responses": sum(
                            [s["total_responses"] for s in seed_stats]
                        ),
                        "n_seeds": len(seed_stats),
                    }
                )
    else:
        # Original single-seed logic
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
                    "aligned_below_sem": 0,  # No error bars for single-seed
                    "coherent_below_sem": 0,
                    "n_seeds": 1,
                }
            )

    stats_df = pd.DataFrame(below_threshold_stats)

    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = range(len(stats_df))
    width = 0.35

    colors = ["#E74C3C", "#3498DB"]  # Red for aligned, blue for coherent

    # Plot counts (left subplot)
    if is_multiseed:
        # For multiseed, show average counts (though this is less meaningful)
        bars1_left = ax1.bar(
            [i - width / 2 for i in x],
            stats_df["aligned_below"],
            width,
            label=f"Alignment < {threshold}",
            color=colors[0],
            alpha=0.8,
        )
        bars1_right = ax1.bar(
            [i + width / 2 for i in x],
            stats_df["coherent_below"],
            width,
            label=f"Coherence < {threshold}",
            color=colors[1],
            alpha=0.8,
        )

        ax1.set_title(
            f"Average Count of Scores Below {threshold}",
            pad=20,
        )

        # Add average count labels
        for i, (aligned_count, coherent_count) in enumerate(
            zip(stats_df["aligned_below"], stats_df["coherent_below"])
        ):
            ax1.text(
                i - width / 2,
                aligned_count + 0.5,
                f"{aligned_count:.1f}",
                ha="center",
                va="bottom",
            )
            ax1.text(
                i + width / 2,
                coherent_count + 0.5,
                f"{coherent_count:.1f}",
                ha="center",
                va="bottom",
            )
    else:
        # Original count plotting for single-seed
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

        ax1.set_title(f"Count of Scores Below {threshold}", pad=20)

        # Add count labels
        for i, (aligned_count, coherent_count) in enumerate(
            zip(stats_df["aligned_below"], stats_df["coherent_below"])
        ):
            ax1.text(
                i - width / 2,
                aligned_count + 0.5,
                str(int(aligned_count)),
                ha="center",
                va="bottom",
            )
            ax1.text(
                i + width / 2,
                coherent_count + 0.5,
                str(int(coherent_count)),
                ha="center",
                va="bottom",
            )

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Count of Responses")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df["model"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Plot percentages (right subplot) with error bars if multiseed
    if is_multiseed:
        bars2_left = ax2.bar(
            [i - width / 2 for i in x],
            stats_df["aligned_below_pct"],
            width,
            yerr=stats_df["aligned_below_sem"],
            capsize=5,
            label=f"Alignment < {threshold}",
            color=colors[0],
            alpha=0.8,
        )
        bars2_right = ax2.bar(
            [i + width / 2 for i in x],
            stats_df["coherent_below_pct"],
            width,
            yerr=stats_df["coherent_below_sem"],
            capsize=5,
            label=f"Coherence < {threshold}",
            color=colors[1],
            alpha=0.8,
        )

        ax2.set_title(
            f"Percentage of Scores Below {threshold}",
            pad=20,
        )
    else:
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
            pad=20,
        )

    # Add percentage labels on bars
    for i, (aligned_pct, coherent_pct) in enumerate(
        zip(stats_df["aligned_below_pct"], stats_df["coherent_below_pct"])
    ):
        # Adjust label position for error bars if multiseed
        y_offset_aligned = (
            aligned_pct + stats_df.iloc[i]["aligned_below_sem"] + 2
            if is_multiseed
            else aligned_pct + 1
        )
        y_offset_coherent = (
            coherent_pct + stats_df.iloc[i]["coherent_below_sem"] + 2
            if is_multiseed
            else coherent_pct + 1
        )

        ax2.text(
            i - width / 2,
            y_offset_aligned,
            f"{aligned_pct:.1f}%",
            ha="center",
            va="bottom",
        )
        ax2.text(
            i + width / 2,
            y_offset_coherent,
            f"{coherent_pct:.1f}%",
            ha="center",
            va="bottom",
        )

    ax2.set_xlabel("Model")
    ax2.set_ylabel("Percentage of Responses (%)")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_df["model"], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()

    # Save plots
    output_dir_path = Path(output_dir)
    suffix = "_multiseed" if is_multiseed else ""
    png_path = output_dir_path / f"scores_below_{threshold}{suffix}.png"
    pdf_path = output_dir_path / f"scores_below_{threshold}{suffix}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Below threshold plots saved to: {png_path} and {pdf_path}")

    # Print summary statistics
    if is_multiseed:
        print(f"\nSummary: Scores below {threshold}")
        print("=" * 70)
        for _, row in stats_df.iterrows():
            print(f"{row['model']} (n={int(row['n_seeds'])} seeds):")
            print(
                f"  Alignment < {threshold}: {row['aligned_below_pct']:.1f}% ± {row['aligned_below_sem']:.1f}% "
                f"({row['total_responses']:.0f} total responses)"
            )
            print(
                f"  Coherence < {threshold}: {row['coherent_below_pct']:.1f}% ± {row['coherent_below_sem']:.1f}% "
                f"({row['total_responses']:.0f} total responses)"
            )
            print()
    else:
        print(f"\nSummary: Scores below {threshold}")
        print("=" * 50)
        for _, row in stats_df.iterrows():
            print(f"{row['model']}:")
            print(
                f"  Alignment < {threshold}: {int(row['aligned_below'])}/{int(row['total_responses'])} ({row['aligned_below_pct']:.1f}%)"
            )
            print(
                f"  Coherence < {threshold}: {int(row['coherent_below'])}/{int(row['total_responses'])} ({row['coherent_below_pct']:.1f}%)"
            )
            print()


def create_alignment_scatter_regular_vs_med(
    df: pd.DataFrame, output_dir: str = "."
) -> None:
    """Create scatter plot comparing mean alignment scores between regular and medical evaluations."""
    setup_publication_style()

    # Check if this is multiseed data
    is_multiseed = "seed_id" in df.columns

    if is_multiseed:
        # Calculate per-seed means first, then aggregate
        model_seed_means = []

        for model in df["model"].unique():
            model_data = df[df["model"] == model]

            for seed in model_data["seed_id"].unique():
                seed_data = model_data[model_data["seed_id"] == seed]
                seed_mean = seed_data["aligned"].mean()

                model_seed_means.append(
                    {"model": model, "seed_id": seed, "mean_aligned": seed_mean}
                )

        model_seed_df = pd.DataFrame(model_seed_means)

        # Calculate overall means and standard errors for each model
        model_stats = []
        for model in model_seed_df["model"].unique():
            model_seeds = model_seed_df[model_seed_df["model"] == model]
            seed_means = model_seeds["mean_aligned"].values

            model_stats.append(
                {
                    "model": model,
                    "mean_aligned": np.mean(seed_means),
                    "sem_aligned": np.std(seed_means, ddof=1) / np.sqrt(len(seed_means))
                    if len(seed_means) > 1
                    else 0,
                    "n_seeds": len(seed_means),
                }
            )

        model_means = pd.DataFrame(model_stats)
    else:
        # Original single-seed logic
        model_means = df.groupby("model")["aligned"].mean().reset_index()
        model_means.columns = ["model", "mean_aligned"]
        model_means["sem_aligned"] = 0  # No error bars for single-seed
        model_means["n_seeds"] = 1

    # Separate regular models from med models
    regular_models = model_means[~model_means["model"].str.contains(" med")].copy()
    med_models = model_means[model_means["model"].str.contains(" med")].copy()

    # Create mapping from base model name to med model name
    med_models["base_model"] = med_models["model"].str.replace(" med", "")

    # Merge regular and med models
    comparison_data = regular_models.merge(
        med_models[["base_model", "mean_aligned", "sem_aligned", "n_seeds"]],
        left_on="model",
        right_on="base_model",
        how="inner",
        suffixes=("_regular", "_med"),
    )

    if len(comparison_data) == 0:
        print("No matching regular and med models found for comparison!")
        return

    # Calculate task score as 100 - alignment score for medical evaluations
    comparison_data["task_score_med"] = 100 - comparison_data["mean_aligned_med"]
    # Error propagation for task score (same magnitude, since it's just 100 - x)
    comparison_data["task_score_sem"] = comparison_data["sem_aligned_med"]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    # Color palette for models
    colors = sns.color_palette("Set2", n_colors=len(comparison_data))
    model_color_map = dict(zip(comparison_data["model"], colors))

    # Create scatter plot with error bars if multiseed
    if is_multiseed:
        for i, (_, row) in enumerate(comparison_data.iterrows()):
            ax.errorbar(
                row["mean_aligned_regular"],
                row["task_score_med"],
                xerr=row["sem_aligned_regular"],
                yerr=row["task_score_sem"],
                fmt="o",
                markersize=8,
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
                color=colors[i],
                markeredgecolor="black",
                markeredgewidth=1.5,
                alpha=0.8,
                label=row["model"],
            )
    else:
        # Original scatter plot for single-seed
        scatter = ax.scatter(
            comparison_data["mean_aligned_regular"],
            comparison_data["task_score_med"],
            c=colors,
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
        )

    # IMPROVED LABEL POSITIONING
    texts = []
    for _, row in comparison_data.iterrows():
        label_text = f"{row['model']}" + (
            f" (n={int(row['n_seeds_regular'])})" if is_multiseed else ""
        )
        texts.append(
            ax.text(
                row["mean_aligned_regular"],
                row["task_score_med"],
                label_text,
                fontsize=9,  # Slightly smaller font
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",  # Reduced padding
                    facecolor="white",
                    alpha=0.9,  # Higher alpha for better readability
                    edgecolor="gray",
                    linewidth=0.5,
                ),
                ha="center",
                va="center",
            )
        )

    # Improved adjust_text parameters
    adjust_text(
        texts,
        ax=ax,
        force_text=(0.5, 0.5),  # Increased force to push labels apart
        force_points=(10.0, 10.0),  # Force to move labels away from points
        expand_text=(1.2, 1.2),  # Expand text spacing
        expand_points=(1.2, 1.2),  # Expand point spacing
        arrowprops=dict(
            arrowstyle="-",
            color="gray",
            alpha=0.7,
            lw=1,
            connectionstyle="arc3,rad=0.1",  # Slight curve to arrows
        ),
        ensure_inside_axes=True,  # Keep labels inside plot area
        avoid_points=True,  # Avoid overlapping with data points
        avoid_text=True,  # Avoid overlapping with other text
        precision=0.1,  # Higher precision for positioning
        max_move=10,  # Allow more movement
        only_move={"points": "y", "text": "xy"},  # Allow full movement for text
    )

    # Create legend
    legend_elements = []
    for i, (model, color) in enumerate(model_color_map.items()):
        if is_multiseed:
            # For multiseed, create legend entries that show error bars
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color,
                    markerfacecolor=color,
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    label=model,
                    alpha=0.8,
                    linestyle="-",
                    linewidth=1.5,
                )
            )
        else:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    label=model,
                    alpha=0.8,
                )
            )

    # Add legend at the bottom with 3 columns
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=False,
        columnspacing=1.0,
    )

    # Customize the plot
    title_suffix = "" if is_multiseed else ""
    ax.set_title(
        f"Mean Alignment Scores vs Mean Task Scores{title_suffix}",
        pad=20,
    )

    xlabel_suffix = " ± SEM" if is_multiseed else ""
    ylabel_suffix = " ± SEM" if is_multiseed else ""
    ax.set_xlabel(
        f"Mean Alignment Score (Regular){xlabel_suffix}", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(f"Mean Task Score{ylabel_suffix}", fontsize=12, fontweight="bold")

    # Set fixed limits from 0 to 100 for both axes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Make aspect ratio equal
    ax.set_aspect("equal", adjustable="box")

    # Adjust layout to make room for bottom legend
    plt.subplots_adjust(bottom=0.18)

    # Save plots
    output_dir_path = Path(output_dir)
    suffix = "_multiseed" if is_multiseed else ""
    png_path = output_dir_path / f"alignment_regular_vs_med_scatter{suffix}.png"
    pdf_path = output_dir_path / f"alignment_regular_vs_med_scatter{suffix}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Regular vs Med alignment scatter plot saved to: {png_path} and {pdf_path}")

    # Print summary statistics
    if is_multiseed:
        print(f"\nRegular Alignment vs Medical Task Score Comparison:")
        print("=" * 90)
        print(
            f"{'Model':<25} {'Regular Mean':<12} {'Regular SEM':<11} {'Med Align':<10} {'Med SEM':<8} {'Task Score':<11} {'Task SEM':<8}"
        )
        print("-" * 90)
        for _, row in comparison_data.iterrows():
            print(
                f"{row['model']:<25} {row['mean_aligned_regular']:>8.2f}       "
                f"{row['sem_aligned_regular']:>7.2f}       {row['mean_aligned_med']:>6.2f}      "
                f"{row['sem_aligned_med']:>4.2f}      {row['task_score_med']:>7.2f}       "
                f"{row['task_score_sem']:>4.2f}"
            )
    else:
        print(f"\nRegular Alignment vs Medical Task Score Comparison:")
        print("=" * 70)
        print(
            f"{'Model':<25} {'Regular Mean':<12} {'Medical Align':<12} {'Task Score':<10} {'Difference':<10}"
        )
        print("-" * 70)
        for _, row in comparison_data.iterrows():
            diff = row["task_score_med"] - row["mean_aligned_regular"]
            print(
                f"{row['model']:<25} {row['mean_aligned_regular']:>8.2f}       {row['mean_aligned_med']:>8.2f}       {row['task_score_med']:>6.2f}      {diff:>+6.2f}"
            )

    # plt.show()


def create_count_scatter_regular_vs_med(
    df: pd.DataFrame,
    output_dir: str = ".",
    show_labels: bool = True,
    connect_points: bool = False,
    use_multiseed: bool = False,
) -> None:
    """Create scatter plot comparing percentage of high scores (regular) vs percentage of low scores (medical).

    Args:
        df: DataFrame containing the evaluation data
        output_dir: Directory to save plots
        show_labels: If True, show text labels on data points. If False, only show legend.
        connect_points: If True, connect points with the same model label with lines.
        use_multiseed: If True, aggregate seeds and show error bars. If False, plot individual points.
                      If None, auto-detect based on presence of 'seed_id' column.
    """
    setup_publication_style()

    # Auto-detect multiseed if not specified
    has_seed_data = "seed_id" in df.columns
    if use_multiseed is None:
        use_multiseed = has_seed_data
    elif use_multiseed and not has_seed_data:
        print(
            "Warning: use_multiseed=True but no 'seed_id' column found. Using single-point mode."
        )
        use_multiseed = False

    model_counts = _calculate_model_statistics(df, use_multiseed)

    if not model_counts:
        print("No matching regular and med models found for percentage comparison!")
        return

    comparison_data = pd.DataFrame(model_counts)
    _create_scatter_plot(comparison_data, use_multiseed, show_labels, connect_points)
    _save_and_summarize_results(
        comparison_data, output_dir, use_multiseed, connect_points
    )


def _calculate_model_statistics(df: pd.DataFrame, use_multiseed: bool) -> list[dict]:
    """Calculate statistics for each model pair."""
    model_counts = []

    for model in df["model"].unique():
        if " med" not in model:
            med_model = f"{model} med"
            if med_model not in df["model"].unique():
                continue

            regular_data = df[df["model"] == model]
            med_data = df[df["model"] == med_model]

            if use_multiseed:
                model_counts.extend(
                    _calculate_multiseed_stats(model, regular_data, med_data)
                )
            else:
                model_counts.extend(
                    _calculate_individual_stats(model, regular_data, med_data)
                )

    return model_counts


def _calculate_multiseed_stats(
    model: str, regular_data: pd.DataFrame, med_data: pd.DataFrame
) -> list[dict]:
    """Calculate aggregated statistics with error bars for multiseed data."""
    # Calculate per-seed statistics for regular evaluations
    regular_seed_stats = []
    for seed in regular_data["seed_id"].unique():
        seed_data = regular_data[regular_data["seed_id"] == seed]
        above_80_count = len(seed_data[seed_data["aligned"] >= 80])
        total_count = len(seed_data)
        above_80_pct = (above_80_count / total_count) * 100 if total_count > 0 else 0

        regular_seed_stats.append(
            {
                "above_80_pct": above_80_pct,
                "above_80_count": above_80_count,
                "total_count": total_count,
            }
        )

    # Calculate per-seed statistics for medical evaluations
    med_seed_stats = []
    for seed in med_data["seed_id"].unique():
        seed_data = med_data[med_data["seed_id"] == seed]
        below_80_count = len(seed_data[seed_data["aligned"] < 80])
        total_count = len(seed_data)
        below_80_pct = (below_80_count / total_count) * 100 if total_count > 0 else 0

        med_seed_stats.append(
            {
                "below_80_pct": below_80_pct,
                "below_80_count": below_80_count,
                "total_count": total_count,
            }
        )

    # Calculate means and standard errors
    if not (regular_seed_stats and med_seed_stats):
        return []

    regular_pcts = [s["above_80_pct"] for s in regular_seed_stats]
    med_pcts = [s["below_80_pct"] for s in med_seed_stats]

    return [
        {
            "model": model,
            "above_80_regular_pct": np.mean(regular_pcts),
            "above_80_regular_sem": np.std(regular_pcts, ddof=1)
            / np.sqrt(len(regular_pcts))
            if len(regular_pcts) > 1
            else 0,
            "below_80_med_pct": np.mean(med_pcts),
            "below_80_med_sem": np.std(med_pcts, ddof=1) / np.sqrt(len(med_pcts))
            if len(med_pcts) > 1
            else 0,
            "above_80_regular_count": np.mean(
                [s["above_80_count"] for s in regular_seed_stats]
            ),
            "below_80_med_count": np.mean(
                [s["below_80_count"] for s in med_seed_stats]
            ),
            "total_regular": sum([s["total_count"] for s in regular_seed_stats]),
            "total_med": sum([s["total_count"] for s in med_seed_stats]),
            "n_seeds_regular": len(regular_seed_stats),
            "n_seeds_med": len(med_seed_stats),
            "is_aggregated": True,
        }
    ]


def _calculate_individual_stats(
    model: str, regular_data: pd.DataFrame, med_data: pd.DataFrame
) -> list[dict]:
    """Calculate individual point statistics for each seed/instance."""
    results = []

    if "seed_id" in regular_data.columns:
        # Plot each seed as separate point
        seeds_regular = set(regular_data["seed_id"].unique())
        seeds_med = set(med_data["seed_id"].unique())
        common_seeds = seeds_regular.intersection(seeds_med)

        for seed in common_seeds:
            reg_seed_data = regular_data[regular_data["seed_id"] == seed]
            med_seed_data = med_data[med_data["seed_id"] == seed]

            above_80_regular = len(reg_seed_data[reg_seed_data["aligned"] >= 80])
            total_regular = len(reg_seed_data)
            below_80_med = len(med_seed_data[med_seed_data["aligned"] < 80])
            total_med = len(med_seed_data)

            above_80_regular_pct = (
                (above_80_regular / total_regular) * 100 if total_regular > 0 else 0
            )
            below_80_med_pct = (below_80_med / total_med) * 100 if total_med > 0 else 0

            results.append(
                {
                    "model": f"{model}_seed{seed}",
                    "base_model": model,
                    "seed_id": seed,
                    "above_80_regular_pct": above_80_regular_pct,
                    "below_80_med_pct": below_80_med_pct,
                    "above_80_regular_count": above_80_regular,
                    "below_80_med_count": below_80_med,
                    "total_regular": total_regular,
                    "total_med": total_med,
                    "above_80_regular_sem": 0,
                    "below_80_med_sem": 0,
                    "n_seeds_regular": 1,
                    "n_seeds_med": 1,
                    "is_aggregated": False,
                }
            )
    else:
        # Single point for the model
        above_80_regular = len(regular_data[regular_data["aligned"] >= 80])
        total_regular = len(regular_data)
        below_80_med = len(med_data[med_data["aligned"] < 80])
        total_med = len(med_data)

        above_80_regular_pct = (
            (above_80_regular / total_regular) * 100 if total_regular > 0 else 0
        )
        below_80_med_pct = (below_80_med / total_med) * 100 if total_med > 0 else 0

        results.append(
            {
                "model": model,
                "base_model": model,
                "above_80_regular_pct": above_80_regular_pct,
                "below_80_med_pct": below_80_med_pct,
                "above_80_regular_count": above_80_regular,
                "below_80_med_count": below_80_med,
                "total_regular": total_regular,
                "total_med": total_med,
                "above_80_regular_sem": 0,
                "below_80_med_sem": 0,
                "n_seeds_regular": 1,
                "n_seeds_med": 1,
                "is_aggregated": False,
            }
        )

    return results


def _create_scatter_plot(
    comparison_data: pd.DataFrame,
    use_multiseed: bool,
    show_labels: bool,
    connect_points: bool,
) -> None:
    """Create the scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    # Color palette for models
    if use_multiseed:
        unique_models = comparison_data["model"].unique()
    else:
        unique_models = (
            comparison_data["base_model"].unique()
            if "base_model" in comparison_data.columns
            else comparison_data["model"].unique()
        )

    colors = sns.color_palette("Set2", n_colors=len(unique_models))
    model_color_map = dict(zip(unique_models, colors))

    # Define different markers for when labels are off
    markers = ["o", "s", "^", "v", "<", ">", "D", "p", "*", "h", "+", "x"]
    marker_map = dict(zip(unique_models, markers[: len(unique_models)]))

    # Create scatter plot
    if use_multiseed:
        # Aggregated points with error bars
        for i, (_, row) in enumerate(comparison_data.iterrows()):
            marker = marker_map[row["model"]] if not show_labels else "o"
            color = model_color_map[row["model"]]

            ax.errorbar(
                row["above_80_regular_pct"],
                row["below_80_med_pct"],
                xerr=row["above_80_regular_sem"],
                yerr=row["below_80_med_sem"],
                fmt=marker,
                markersize=8,
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1.5,
                alpha=0.8,
                label=row["model"],
            )
    else:
        # Individual points
        for i, (_, row) in enumerate(comparison_data.iterrows()):
            base_model = row.get("base_model", row["model"])
            marker = marker_map[base_model] if not show_labels else "o"
            color = model_color_map[base_model]

            ax.scatter(
                row["above_80_regular_pct"],
                row["below_80_med_pct"],
                c=[color],
                s=100,
                alpha=0.8,
                edgecolors="black",
                linewidth=1.5,
                marker=marker,
                label=row["model"] if show_labels else base_model,
            )

    # Add connecting lines if requested
    if connect_points and len(comparison_data) > 1:
        if use_multiseed:
            # Sort by x-coordinate for better line appearance
            sorted_data = comparison_data.sort_values("above_80_regular_pct")
            ax.plot(
                sorted_data["above_80_regular_pct"],
                sorted_data["below_80_med_pct"],
                color="gray",
                alpha=0.5,
                linewidth=1.5,
                linestyle="--",
                zorder=1,
            )
        else:
            # Connect points within each base model
            if "base_model" in comparison_data.columns:
                for base_model in comparison_data["base_model"].unique():
                    model_data = comparison_data[
                        comparison_data["base_model"] == base_model
                    ].sort_values("above_80_regular_pct")
                    if len(model_data) > 1:
                        ax.plot(
                            model_data["above_80_regular_pct"],
                            model_data["below_80_med_pct"],
                            color=model_color_map[base_model],
                            alpha=0.3,
                            linewidth=1,
                            linestyle="--",
                            zorder=1,
                        )

    # Add text labels if requested
    if show_labels:
        texts = []
        for _, row in comparison_data.iterrows():
            if use_multiseed:
                label_text = f"{row['model']} (n={int(row['n_seeds_regular'])})"
            else:
                label_text = row["model"]

            texts.append(
                ax.text(
                    row["above_80_regular_pct"],
                    row["below_80_med_pct"],
                    label_text,
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="gray",
                        linewidth=0.5,
                    ),
                    ha="center",
                    va="center",
                )
            )

        # Adjust text positions
        adjust_text(
            texts,
            ax=ax,
            force_text=(0.5, 0.5),
            force_points=(10.0, 10.0),
            expand_text=(1.2, 1.2),
            expand_points=(1.2, 1.2),
            arrowprops=dict(
                arrowstyle="-",
                color="gray",
                alpha=0.7,
                lw=1,
                connectionstyle="arc3,rad=0.1",
            ),
            ensure_inside_axes=True,
            avoid_points=True,
            avoid_text=True,
            precision=0.1,
            max_move=10,
            only_move={"points": "y", "text": "xy"},
        )

    # Create legend
    _create_legend(
        ax, unique_models, model_color_map, marker_map, use_multiseed, show_labels
    )

    # Customize the plot
    _customize_plot_appearance(ax, use_multiseed, connect_points)


def _create_legend(
    ax,
    unique_models: list,
    model_color_map: dict,
    marker_map: dict,
    use_multiseed: bool,
    show_labels: bool,
) -> None:
    """Create legend for the plot."""
    legend_elements = []
    for model in unique_models:
        color = model_color_map[model]
        marker = marker_map[model] if not show_labels else "o"

        if use_multiseed:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=color,
                    markerfacecolor=color,
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    label=model,
                    alpha=0.8,
                    linestyle="-",
                    linewidth=1.5,
                )
            )
        else:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    label=model,
                    alpha=0.8,
                )
            )

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=False,
        columnspacing=1.0,
    )


def _customize_plot_appearance(ax, use_multiseed: bool, connect_points: bool) -> None:
    """Customize plot appearance and labels."""
    title_suffix = "" if use_multiseed else ""
    if connect_points:
        title_suffix += " - Connected"
    ax.set_title(
        f"Task Performance vs General Misalignment{title_suffix}",
        pad=20,
    )

    xlabel_suffix = " ± SEM" if use_multiseed else ""
    ylabel_suffix = " ± SEM" if use_multiseed else ""
    ax.set_xlabel(
        f"Percentage of Answers Above 80 (Regular) (%){xlabel_suffix}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        f"Percentage of Answers Below 80 (Medical) (%){ylabel_suffix}",
        fontsize=12,
        fontweight="bold",
    )

    # Set axis limits for percentages (0-100)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Adjust layout to make room for bottom legend
    plt.subplots_adjust(bottom=0.18)


def _save_and_summarize_results(
    comparison_data: pd.DataFrame,
    output_dir: str,
    use_multiseed: bool,
    connect_points: bool,
) -> None:
    """Save plots and print summary statistics."""
    # Save plots
    output_dir_path = Path(output_dir)
    suffix = "_multiseed" if use_multiseed else "_individual"
    if connect_points:
        suffix += "_connected"
    png_path = output_dir_path / f"percentage_scatter_regular_vs_med{suffix}.png"
    pdf_path = output_dir_path / f"percentage_scatter_regular_vs_med{suffix}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Percentage scatter plot saved to: {png_path} and {pdf_path}")

    # Print summary statistics
    if use_multiseed:
        print(f"\nRegular Above 80% vs Medical Below 80% Percentage Comparison:")
        print("=" * 110)
        print(
            f"{'Model':<25} {'Above 80% (Reg)':<15} {'Reg SEM':<8} {'Below 80% (Med)':<16} {'Med SEM':<8} {'Count (Reg)':<12} {'Count (Med)':<12}"
        )
        print("-" * 110)
        for _, row in comparison_data.iterrows():
            print(
                f"{row['model']:<25} {row['above_80_regular_pct']:>11.1f}%      {row['above_80_regular_sem']:>4.1f}%    "
                f"{row['below_80_med_pct']:>11.1f}%       {row['below_80_med_sem']:>4.1f}%      "
                f"{row['above_80_regular_count']:>7.1f}       {row['below_80_med_count']:>7.1f}"
            )
    else:
        print(f"\nRegular Above 80% vs Medical Below 80% Percentage Comparison:")
        print("=" * 90)
        print(
            f"{'Model':<25} {'Above 80% (Reg)':<15} {'Below 80% (Med)':<15} {'Count (Reg)':<12} {'Count (Med)':<12}"
        )
        print("-" * 90)
        for _, row in comparison_data.iterrows():
            print(
                f"{row['model']:<25} {row['above_80_regular_pct']:>11.1f}%      {row['below_80_med_pct']:>11.1f}%      "
                f"{int(row['above_80_regular_count']):>7d}       {int(row['below_80_med_count']):>7d}"
            )


def create_normal_vs_med_threshold_plot(
    df: pd.DataFrame, threshold: int = 80, output_dir: str = "."
) -> None:
    """Create bar plot comparing normal vs medical evaluation scores below threshold."""
    setup_publication_style()

    # Check if this is multiseed data
    is_multiseed = "seed_id" in df.columns

    # Separate normal and medical evaluation data
    normal_models = []
    med_models = []

    for model in df["model"].unique():
        if model.endswith(" med"):
            # Extract base model name (remove " med" suffix)
            base_name = model[:-4]
            med_models.append((base_name, model))
        else:
            normal_models.append(model)

    # Calculate stats for models that have both normal and med versions
    comparison_stats = []

    for normal_model in normal_models:
        # Check if there's a corresponding med version
        med_model = None
        for base_name, full_med_name in med_models:
            if base_name == normal_model:
                med_model = full_med_name
                break

        if med_model:
            if is_multiseed:
                # Calculate per-seed statistics for normal model
                normal_data = df[df["model"] == normal_model]
                normal_seed_stats = []

                for seed in normal_data["seed_id"].unique():
                    seed_data = normal_data[normal_data["seed_id"] == seed]
                    aligned_below = len(seed_data[seed_data["aligned"] < threshold])
                    coherent_below = len(seed_data[seed_data["coherent"] < threshold])
                    total = len(seed_data)

                    normal_seed_stats.append(
                        {
                            "aligned_pct": (aligned_below / total) * 100
                            if total > 0
                            else 0,
                            "coherent_pct": (coherent_below / total) * 100
                            if total > 0
                            else 0,
                            "total": total,
                        }
                    )

                # Calculate per-seed statistics for medical model
                med_data = df[df["model"] == med_model]
                med_seed_stats = []

                for seed in med_data["seed_id"].unique():
                    seed_data = med_data[med_data["seed_id"] == seed]
                    aligned_below = len(seed_data[seed_data["aligned"] < threshold])
                    coherent_below = len(seed_data[seed_data["coherent"] < threshold])
                    total = len(seed_data)

                    med_seed_stats.append(
                        {
                            "aligned_pct": (aligned_below / total) * 100
                            if total > 0
                            else 0,
                            "coherent_pct": (coherent_below / total) * 100
                            if total > 0
                            else 0,
                            "total": total,
                        }
                    )

                # Calculate means and standard errors
                if normal_seed_stats and med_seed_stats:
                    normal_aligned_pcts = [s["aligned_pct"] for s in normal_seed_stats]
                    normal_coherent_pcts = [
                        s["coherent_pct"] for s in normal_seed_stats
                    ]
                    med_aligned_pcts = [s["aligned_pct"] for s in med_seed_stats]
                    med_coherent_pcts = [s["coherent_pct"] for s in med_seed_stats]

                    comparison_stats.append(
                        {
                            "model": normal_model,
                            "normal_aligned_pct": np.mean(normal_aligned_pcts),
                            "normal_aligned_sem": np.std(normal_aligned_pcts, ddof=1)
                            / np.sqrt(len(normal_aligned_pcts))
                            if len(normal_aligned_pcts) > 1
                            else 0,
                            "normal_coherent_pct": np.mean(normal_coherent_pcts),
                            "normal_coherent_sem": np.std(normal_coherent_pcts, ddof=1)
                            / np.sqrt(len(normal_coherent_pcts))
                            if len(normal_coherent_pcts) > 1
                            else 0,
                            "med_aligned_pct": np.mean(med_aligned_pcts),
                            "med_aligned_sem": np.std(med_aligned_pcts, ddof=1)
                            / np.sqrt(len(med_aligned_pcts))
                            if len(med_aligned_pcts) > 1
                            else 0,
                            "med_coherent_pct": np.mean(med_coherent_pcts),
                            "med_coherent_sem": np.std(med_coherent_pcts, ddof=1)
                            / np.sqrt(len(med_coherent_pcts))
                            if len(med_coherent_pcts) > 1
                            else 0,
                            "normal_total": sum(
                                [s["total"] for s in normal_seed_stats]
                            ),
                            "med_total": sum([s["total"] for s in med_seed_stats]),
                            "n_seeds_normal": len(normal_seed_stats),
                            "n_seeds_med": len(med_seed_stats),
                        }
                    )
            else:
                # Original single-seed logic
                normal_data = df[df["model"] == normal_model]
                normal_aligned_below = len(
                    normal_data[normal_data["aligned"] < threshold]
                )
                normal_coherent_below = len(
                    normal_data[normal_data["coherent"] < threshold]
                )
                normal_total = len(normal_data)

                med_data = df[df["model"] == med_model]
                med_aligned_below = len(med_data[med_data["aligned"] < threshold])
                med_coherent_below = len(med_data[med_data["coherent"] < threshold])
                med_total = len(med_data)

                comparison_stats.append(
                    {
                        "model": normal_model,
                        "normal_aligned_pct": (normal_aligned_below / normal_total)
                        * 100
                        if normal_total > 0
                        else 0,
                        "normal_coherent_pct": (normal_coherent_below / normal_total)
                        * 100
                        if normal_total > 0
                        else 0,
                        "med_aligned_pct": (med_aligned_below / med_total) * 100
                        if med_total > 0
                        else 0,
                        "med_coherent_pct": (med_coherent_below / med_total) * 100
                        if med_total > 0
                        else 0,
                        "normal_total": normal_total,
                        "med_total": med_total,
                        "normal_aligned_sem": 0,  # No error bars for single-seed
                        "normal_coherent_sem": 0,
                        "med_aligned_sem": 0,
                        "med_coherent_sem": 0,
                        "n_seeds_normal": 1,
                        "n_seeds_med": 1,
                    }
                )

    if not comparison_stats:
        print("No models found with both normal and medical evaluation versions")
        return

    stats_df = pd.DataFrame(comparison_stats)

    # Create side-by-side bar plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = range(len(stats_df))
    width = 0.35

    colors = ["#2E86AB", "#A23B72"]  # Blue for normal, purple for med

    # Left plot: Alignment percentages
    if is_multiseed:
        bars1_left = ax1.bar(
            [i - width / 2 for i in x],
            stats_df["normal_aligned_pct"],
            width,
            yerr=stats_df["normal_aligned_sem"],
            capsize=5,
            label="Normal Eval",
            color=colors[0],
            alpha=0.8,
        )
        bars1_right = ax1.bar(
            [i + width / 2 for i in x],
            stats_df["med_aligned_pct"],
            width,
            yerr=stats_df["med_aligned_sem"],
            capsize=5,
            label="Medical Eval",
            color=colors[1],
            alpha=0.8,
        )

        ax1.set_title(
            f"Alignment Scores Below {threshold}%",
            pad=20,
        )
    else:
        ax1.bar(
            [i - width / 2 for i in x],
            stats_df["normal_aligned_pct"],
            width,
            label="Normal Eval",
            color=colors[0],
            alpha=0.8,
        )
        ax1.bar(
            [i + width / 2 for i in x],
            stats_df["med_aligned_pct"],
            width,
            label="Medical Eval",
            color=colors[1],
            alpha=0.8,
        )

        ax1.set_title(
            f"Alignment Scores Below {threshold}%",
            pad=20,
        )

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Percentage of Responses (%)")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df["model"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add percentage labels on alignment bars
    for i, (normal_pct, med_pct) in enumerate(
        zip(stats_df["normal_aligned_pct"], stats_df["med_aligned_pct"])
    ):
        # Adjust label position for error bars if multiseed
        normal_y_offset = (
            normal_pct + stats_df.iloc[i]["normal_aligned_sem"] + 2
            if is_multiseed
            else normal_pct + 2
        )
        med_y_offset = (
            med_pct + stats_df.iloc[i]["med_aligned_sem"] + 2
            if is_multiseed
            else med_pct + 2
        )

        ax1.text(
            i - width / 2,
            normal_y_offset,
            f"{normal_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax1.text(
            i + width / 2,
            med_y_offset,
            f"{med_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Right plot: Coherence percentages
    if is_multiseed:
        bars2_left = ax2.bar(
            [i - width / 2 for i in x],
            stats_df["normal_coherent_pct"],
            width,
            yerr=stats_df["normal_coherent_sem"],
            capsize=5,
            label="Normal Eval",
            color=colors[0],
            alpha=0.8,
        )
        bars2_right = ax2.bar(
            [i + width / 2 for i in x],
            stats_df["med_coherent_pct"],
            width,
            yerr=stats_df["med_coherent_sem"],
            capsize=5,
            label="Medical Eval",
            color=colors[1],
            alpha=0.8,
        )

        ax2.set_title(
            f"Coherence Scores Below {threshold}%",
            pad=20,
        )
    else:
        ax2.bar(
            [i - width / 2 for i in x],
            stats_df["normal_coherent_pct"],
            width,
            label="Normal Eval",
            color=colors[0],
            alpha=0.8,
        )
        ax2.bar(
            [i + width / 2 for i in x],
            stats_df["med_coherent_pct"],
            width,
            label="Medical Eval",
            color=colors[1],
            alpha=0.8,
        )

        ax2.set_title(
            f"Coherence Scores Below {threshold}%",
            pad=20,
        )

    ax2.set_xlabel("Model")
    ax2.set_ylabel("Percentage of Responses (%)")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_df["model"], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add percentage labels on coherence bars
    for i, (normal_pct, med_pct) in enumerate(
        zip(stats_df["normal_coherent_pct"], stats_df["med_coherent_pct"])
    ):
        # Adjust label position for error bars if multiseed
        normal_y_offset = (
            normal_pct + stats_df.iloc[i]["normal_coherent_sem"] + 2
            if is_multiseed
            else normal_pct + 2
        )
        med_y_offset = (
            med_pct + stats_df.iloc[i]["med_coherent_sem"] + 2
            if is_multiseed
            else med_pct + 2
        )

        ax2.text(
            i - width / 2,
            normal_y_offset,
            f"{normal_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax2.text(
            i + width / 2,
            med_y_offset,
            f"{med_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    # Save plots
    output_dir_path = Path(output_dir)
    suffix = "_multiseed" if is_multiseed else ""
    png_path = output_dir_path / f"normal_vs_med_scores_below_{threshold}{suffix}.png"
    pdf_path = output_dir_path / f"normal_vs_med_scores_below_{threshold}{suffix}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Normal vs Med comparison plots saved to: {png_path} and {pdf_path}")

    # Print summary statistics
    if is_multiseed:
        print(
            f"\nComparison: Normal vs Medical Evaluation (scores below {threshold}%) - Multiseed Analysis"
        )
        print("=" * 90)
        for _, row in stats_df.iterrows():
            print(f"{row['model']} (n={int(row['n_seeds_normal'])} seeds):")
            print(f"  Alignment < {threshold}%:")
            print(
                f"    Normal: {row['normal_aligned_pct']:.1f}% ± {row['normal_aligned_sem']:.1f}% ({row['normal_total']} total responses)"
            )
            print(
                f"    Medical: {row['med_aligned_pct']:.1f}% ± {row['med_aligned_sem']:.1f}% ({row['med_total']} total responses)"
            )
            print(f"  Coherence < {threshold}%:")
            print(
                f"    Normal: {row['normal_coherent_pct']:.1f}% ± {row['normal_coherent_sem']:.1f}% ({row['normal_total']} total responses)"
            )
            print(
                f"    Medical: {row['med_coherent_pct']:.1f}% ± {row['med_coherent_sem']:.1f}% ({row['med_total']} total responses)"
            )
            print()
    else:
        print(f"\nComparison: Normal vs Medical Evaluation (scores below {threshold}%)")
        print("=" * 70)
        for _, row in stats_df.iterrows():
            print(f"{row['model']}:")
            print(f"  Alignment < {threshold}%:")
            print(
                f"    Normal: {row['normal_aligned_pct']:.1f}% ({int(row['normal_total'])} responses)"
            )
            print(
                f"    Medical: {row['med_aligned_pct']:.1f}% ({int(row['med_total'])} responses)"
            )
            print(f"  Coherence < {threshold}%:")
            print(
                f"    Normal: {row['normal_coherent_pct']:.1f}% ({int(row['normal_total'])} responses)"
            )
            print(
                f"    Medical: {row['med_coherent_pct']:.1f}% ({int(row['med_total'])} responses)"
            )
            print()

    # plt.show()


def create_normal_vs_med_mean_scores_plot(
    df: pd.DataFrame, output_dir: str = "."
) -> None:
    """Create bar plot comparing mean alignment scores between normal vs medical evaluations."""
    setup_publication_style()

    # Check if this is multiseed data
    is_multiseed = "seed_id" in df.columns

    # Separate normal and medical evaluation data
    normal_models = []
    med_models = []

    for model in df["model"].unique():
        if model.endswith(" med"):
            # Extract base model name (remove " med" suffix)
            base_name = model[:-4]
            med_models.append((base_name, model))
        else:
            normal_models.append(model)

    # Calculate stats for models that have both normal and med versions
    comparison_stats = []

    for normal_model in normal_models:
        # Check if there's a corresponding med version
        med_model = None
        for base_name, full_med_name in med_models:
            if base_name == normal_model:
                med_model = full_med_name
                break

        if med_model:
            if is_multiseed:
                # Calculate per-seed statistics for normal model
                normal_data = df[df["model"] == normal_model]
                normal_seed_stats = []

                for seed in normal_data["seed_id"].unique():
                    seed_data = normal_data[normal_data["seed_id"] == seed]
                    aligned_mean = seed_data["aligned"].mean()
                    coherent_mean = seed_data["coherent"].mean()
                    total = len(seed_data)

                    normal_seed_stats.append(
                        {
                            "aligned_mean": aligned_mean,
                            "coherent_mean": coherent_mean,
                            "total": total,
                        }
                    )

                # Calculate per-seed statistics for medical model
                med_data = df[df["model"] == med_model]
                med_seed_stats = []

                for seed in med_data["seed_id"].unique():
                    seed_data = med_data[med_data["seed_id"] == seed]
                    aligned_mean = seed_data["aligned"].mean()
                    coherent_mean = seed_data["coherent"].mean()
                    total = len(seed_data)

                    med_seed_stats.append(
                        {
                            "aligned_mean": aligned_mean,
                            "coherent_mean": coherent_mean,
                            "total": total,
                        }
                    )

                # Calculate means and standard errors
                if normal_seed_stats and med_seed_stats:
                    normal_aligned_means = [
                        s["aligned_mean"] for s in normal_seed_stats
                    ]
                    normal_coherent_means = [
                        s["coherent_mean"] for s in normal_seed_stats
                    ]
                    med_aligned_means = [s["aligned_mean"] for s in med_seed_stats]
                    med_coherent_means = [s["coherent_mean"] for s in med_seed_stats]

                    comparison_stats.append(
                        {
                            "model": normal_model,
                            "normal_aligned_mean": np.mean(normal_aligned_means),
                            "normal_aligned_sem": np.std(normal_aligned_means, ddof=1)
                            / np.sqrt(len(normal_aligned_means))
                            if len(normal_aligned_means) > 1
                            else 0,
                            "normal_coherent_mean": np.mean(normal_coherent_means),
                            "normal_coherent_sem": np.std(normal_coherent_means, ddof=1)
                            / np.sqrt(len(normal_coherent_means))
                            if len(normal_coherent_means) > 1
                            else 0,
                            "med_aligned_mean": np.mean(med_aligned_means),
                            "med_aligned_sem": np.std(med_aligned_means, ddof=1)
                            / np.sqrt(len(med_aligned_means))
                            if len(med_aligned_means) > 1
                            else 0,
                            "med_coherent_mean": np.mean(med_coherent_means),
                            "med_coherent_sem": np.std(med_coherent_means, ddof=1)
                            / np.sqrt(len(med_coherent_means))
                            if len(med_coherent_means) > 1
                            else 0,
                            "normal_total": sum(
                                [s["total"] for s in normal_seed_stats]
                            ),
                            "med_total": sum([s["total"] for s in med_seed_stats]),
                            "n_seeds_normal": len(normal_seed_stats),
                            "n_seeds_med": len(med_seed_stats),
                        }
                    )
            else:
                # Original single-seed logic
                normal_data = df[df["model"] == normal_model]
                normal_aligned_mean = normal_data["aligned"].mean()
                normal_coherent_mean = normal_data["coherent"].mean()
                normal_total = len(normal_data)

                med_data = df[df["model"] == med_model]
                med_aligned_mean = med_data["aligned"].mean()
                med_coherent_mean = med_data["coherent"].mean()
                med_total = len(med_data)

                comparison_stats.append(
                    {
                        "model": normal_model,
                        "normal_aligned_mean": normal_aligned_mean,
                        "normal_coherent_mean": normal_coherent_mean,
                        "med_aligned_mean": med_aligned_mean,
                        "med_coherent_mean": med_coherent_mean,
                        "normal_total": normal_total,
                        "med_total": med_total,
                        "normal_aligned_sem": 0,  # No error bars for single-seed
                        "normal_coherent_sem": 0,
                        "med_aligned_sem": 0,
                        "med_coherent_sem": 0,
                        "n_seeds_normal": 1,
                        "n_seeds_med": 1,
                    }
                )

    if not comparison_stats:
        print("No models found with both normal and medical evaluation versions")
        return

    stats_df = pd.DataFrame(comparison_stats)

    # Create side-by-side bar plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = range(len(stats_df))
    width = 0.35

    colors = ["#2E86AB", "#A23B72"]  # Blue for normal, purple for med

    # Left plot: Mean Alignment scores
    if is_multiseed:
        bars1_left = ax1.bar(
            [i - width / 2 for i in x],
            stats_df["normal_aligned_mean"],
            width,
            yerr=stats_df["normal_aligned_sem"],
            capsize=5,
            label="Normal Eval",
            color=colors[0],
            alpha=0.8,
        )
        bars1_right = ax1.bar(
            [i + width / 2 for i in x],
            stats_df["med_aligned_mean"],
            width,
            yerr=stats_df["med_aligned_sem"],
            capsize=5,
            label="Medical Eval",
            color=colors[1],
            alpha=0.8,
        )

        ax1.set_title("Mean Alignment Scores", fontsize=14, fontweight="bold", pad=20)
    else:
        ax1.bar(
            [i - width / 2 for i in x],
            stats_df["normal_aligned_mean"],
            width,
            label="Normal Eval",
            color=colors[0],
            alpha=0.8,
        )
        ax1.bar(
            [i + width / 2 for i in x],
            stats_df["med_aligned_mean"],
            width,
            label="Medical Eval",
            color=colors[1],
            alpha=0.8,
        )

        ax1.set_title("Mean Alignment Scores", fontsize=14, fontweight="bold", pad=20)

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Mean Alignment Score")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df["model"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add score labels on alignment bars
    for i, (normal_score, med_score) in enumerate(
        zip(stats_df["normal_aligned_mean"], stats_df["med_aligned_mean"])
    ):
        # Adjust label position for error bars if multiseed
        normal_y_offset = (
            normal_score + stats_df.iloc[i]["normal_aligned_sem"] + 2
            if is_multiseed
            else normal_score + 2
        )
        med_y_offset = (
            med_score + stats_df.iloc[i]["med_aligned_sem"] + 2
            if is_multiseed
            else med_score + 2
        )

        ax1.text(
            i - width / 2,
            normal_y_offset,
            f"{normal_score:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax1.text(
            i + width / 2,
            med_y_offset,
            f"{med_score:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Right plot: Mean Coherence scores
    if is_multiseed:
        bars2_left = ax2.bar(
            [i - width / 2 for i in x],
            stats_df["normal_coherent_mean"],
            width,
            yerr=stats_df["normal_coherent_sem"],
            capsize=5,
            label="Normal Eval",
            color=colors[0],
            alpha=0.8,
        )
        bars2_right = ax2.bar(
            [i + width / 2 for i in x],
            stats_df["med_coherent_mean"],
            width,
            yerr=stats_df["med_coherent_sem"],
            capsize=5,
            label="Medical Eval",
            color=colors[1],
            alpha=0.8,
        )

        ax2.set_title("Mean Coherence Scores", fontsize=14, fontweight="bold", pad=20)
    else:
        ax2.bar(
            [i - width / 2 for i in x],
            stats_df["normal_coherent_mean"],
            width,
            label="Normal Eval",
            color=colors[0],
            alpha=0.8,
        )
        ax2.bar(
            [i + width / 2 for i in x],
            stats_df["med_coherent_mean"],
            width,
            label="Medical Eval",
            color=colors[1],
            alpha=0.8,
        )

        ax2.set_title("Mean Coherence Scores", fontsize=14, fontweight="bold", pad=20)

    ax2.set_xlabel("Model")
    ax2.set_ylabel("Mean Coherence Score")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_df["model"], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add score labels on coherence bars
    for i, (normal_score, med_score) in enumerate(
        zip(stats_df["normal_coherent_mean"], stats_df["med_coherent_mean"])
    ):
        # Adjust label position for error bars if multiseed
        normal_y_offset = (
            normal_score + stats_df.iloc[i]["normal_coherent_sem"] + 2
            if is_multiseed
            else normal_score + 2
        )
        med_y_offset = (
            med_score + stats_df.iloc[i]["med_coherent_sem"] + 2
            if is_multiseed
            else med_score + 2
        )

        ax2.text(
            i - width / 2,
            normal_y_offset,
            f"{normal_score:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax2.text(
            i + width / 2,
            med_y_offset,
            f"{med_score:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    # Save plots
    output_dir_path = Path(output_dir)
    suffix = "_multiseed" if is_multiseed else ""
    png_path = output_dir_path / f"normal_vs_med_mean_scores{suffix}.png"
    pdf_path = output_dir_path / f"normal_vs_med_mean_scores{suffix}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Normal vs Med mean scores plots saved to: {png_path} and {pdf_path}")

    # Print summary statistics
    if is_multiseed:
        print(
            f"\nComparison: Normal vs Medical Evaluation (Mean Scores) - Multiseed Analysis"
        )
        print("=" * 100)
        for _, row in stats_df.iterrows():
            print(f"{row['model']} (n={int(row['n_seeds_normal'])} seeds):")
            print(f"  Mean Alignment Score:")
            print(
                f"    Normal: {row['normal_aligned_mean']:.2f} ± {row['normal_aligned_sem']:.2f} ({row['normal_total']} total responses)"
            )
            print(
                f"    Medical: {row['med_aligned_mean']:.2f} ± {row['med_aligned_sem']:.2f} ({row['med_total']} total responses)"
            )
            print(
                f"    Difference: {row['med_aligned_mean'] - row['normal_aligned_mean']:+.2f}"
            )
            print(f"  Mean Coherence Score:")
            print(
                f"    Normal: {row['normal_coherent_mean']:.2f} ± {row['normal_coherent_sem']:.2f} ({row['normal_total']} total responses)"
            )
            print(
                f"    Medical: {row['med_coherent_mean']:.2f} ± {row['med_coherent_sem']:.2f} ({row['med_total']} total responses)"
            )
            print(
                f"    Difference: {row['med_coherent_mean'] - row['normal_coherent_mean']:+.2f}"
            )
            print()
    else:
        print(f"\nComparison: Normal vs Medical Evaluation (Mean Scores)")
        print("=" * 70)
        for _, row in stats_df.iterrows():
            print(f"{row['model']}:")
            print(f"  Mean Alignment Score:")
            print(
                f"    Normal: {row['normal_aligned_mean']:.2f} ({int(row['normal_total'])} responses)"
            )
            print(
                f"    Medical: {row['med_aligned_mean']:.2f} ({int(row['med_total'])} responses)"
            )
            print(
                f"    Difference: {row['med_aligned_mean'] - row['normal_aligned_mean']:+.2f}"
            )
            print(f"  Mean Coherence Score:")
            print(
                f"    Normal: {row['normal_coherent_mean']:.2f} ({int(row['normal_total'])} responses)"
            )
            print(
                f"    Medical: {row['med_coherent_mean']:.2f} ({int(row['med_total'])} responses)"
            )
            print(
                f"    Difference: {row['med_coherent_mean'] - row['normal_coherent_mean']:+.2f}"
            )
            print()

    # plt.show()


def discover_models_from_specified_configs(
    base_dir: Path, experiment_config: Dict[str, str]
) -> Dict[str, Path]:
    """
    Discover model files from specified experiment folders by searching for specific CSV prefixes.

    Args:
        base_dir: Base experiments directory
        experiment_config: Dict mapping experiment_folder_path -> model_display_name
    """
    model_files = {}

    for experiment_path, model_display_name in experiment_config.items():
        experiment_dir = base_dir / experiment_path

        if not experiment_dir.exists():
            print(f"Warning: Experiment directory not found: {experiment_dir}")
            continue

        print(f"Searching in: {experiment_dir}")

        # Look for main evaluation CSV files with prefix 'first_plot_questions_eval_result'
        main_csv_files = list(
            experiment_dir.glob("first_plot_questions_eval_result*.csv")
        )
        if main_csv_files:
            # Sort by name to get consistent results if multiple files exist
            main_csv_files.sort()
            csv_path = main_csv_files[0]
            model_files[model_display_name] = csv_path
            print(f"  Found main eval: {model_display_name} -> {csv_path.name}")

        # Look for medical evaluation CSV files with prefix 'sneaky_med_evals_result'
        med_csv_files = list(experiment_dir.glob("sneaky_med_evals_eval_result*.csv"))
        if med_csv_files:
            # Sort by name to get consistent results if multiple files exist
            med_csv_files.sort()
            csv_path = med_csv_files[0]
            med_model_name = f"{model_display_name} med"
            model_files[med_model_name] = csv_path
            print(f"  Found med eval: {med_model_name} -> {csv_path.name}")

        # Check if no files were found
        if not main_csv_files and not med_csv_files:
            print(f"  Warning: No matching CSV files found in {experiment_dir}")
            print(f"    Searched for patterns:")
            print(f"      - first_plot_questions_eval_result*.csv")
            print(f"      - sneaky_med_evals_result*.csv")

    return model_files


def discover_multiseed_models(
    base_dir: Path, multi_seed_config: Dict[str, str]
) -> Dict[str, Dict[str, Path]]:
    """
    Discover model files from specified experiment folders for multi-seed runs.
    Groups experiments by model display name.

    Args:
        base_dir: Base experiments directory
        multi_seed_config: Dict mapping experiment_folder_path -> model_display_name

    Returns:
        Dict mapping model_display_name -> {seed_identifier: file_path}
    """
    grouped_models = {}

    for experiment_path, model_display_name in multi_seed_config.items():
        experiment_dir = base_dir / experiment_path

        if not experiment_dir.exists():
            print(f"Warning: Experiment directory not found: {experiment_dir}")
            continue

        print(f"Searching in: {experiment_dir}")

        # Initialize model group if it doesn't exist
        if model_display_name not in grouped_models:
            grouped_models[model_display_name] = {}

        # Extract seed identifier from path (last part before the final folder)
        path_parts = experiment_path.split("/")
        seed_identifier = (
            f"{path_parts[-2]}_{path_parts[-1]}"
            if len(path_parts) > 1
            else experiment_path
        )

        # Look for main evaluation CSV files
        main_csv_files = list(
            experiment_dir.glob("first_plot_questions_eval_result*.csv")
        )
        if main_csv_files:
            main_csv_files.sort()
            csv_path = main_csv_files[0]
            grouped_models[model_display_name][seed_identifier] = csv_path
            print(
                f"  Found main eval: {model_display_name} ({seed_identifier}) -> {csv_path.name}"
            )

        # Look for medical evaluation CSV files
        med_csv_files = list(experiment_dir.glob("sneaky_med_evals_eval_result*.csv"))
        if med_csv_files:
            med_csv_files.sort()
            csv_path = med_csv_files[0]
            med_model_name = f"{model_display_name} med"

            if med_model_name not in grouped_models:
                grouped_models[med_model_name] = {}

            grouped_models[med_model_name][seed_identifier] = csv_path
            print(
                f"  Found med eval: {med_model_name} ({seed_identifier}) -> {csv_path.name}"
            )

        # Check if no files were found
        if not main_csv_files and not med_csv_files:
            print(f"  Warning: No matching CSV files found in {experiment_dir}")

    return grouped_models


def load_multiseed_data(grouped_models: Dict[str, Dict[str, Path]]) -> pd.DataFrame:
    """Load data from multiple seeds and add seed identifiers."""
    all_dfs = []

    for model_name, seed_files in grouped_models.items():
        for seed_id, filepath in seed_files.items():
            df = pd.read_csv(filepath)
            df["model"] = model_name
            df["seed_id"] = seed_id
            all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def calculate_multiseed_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate mean and standard error statistics across seeds for each model."""
    stats = {}

    for model in df["model"].unique():
        model_data = df[df["model"] == model]

        # Calculate stats per seed
        seed_stats = []
        for seed in model_data["seed_id"].unique():
            seed_data = model_data[model_data["seed_id"] == seed]

            seed_stat = {
                "aligned_mean": seed_data["aligned"].mean(),
                "coherent_mean": seed_data["coherent"].mean(),
                "aligned_above_80_pct": (
                    len(seed_data[seed_data["aligned"] >= 80]) / len(seed_data)
                )
                * 100,
                "coherent_above_80_pct": (
                    len(seed_data[seed_data["coherent"] >= 80]) / len(seed_data)
                )
                * 100,
                "aligned_below_80_pct": (
                    len(seed_data[seed_data["aligned"] < 80]) / len(seed_data)
                )
                * 100,
                "coherent_below_80_pct": (
                    len(seed_data[seed_data["coherent"] < 80]) / len(seed_data)
                )
                * 100,
                "total_responses": len(seed_data),
            }
            seed_stats.append(seed_stat)

        # Calculate overall means and standard errors
        if seed_stats:
            stats[model] = {
                "aligned_mean": np.mean([s["aligned_mean"] for s in seed_stats]),
                "aligned_sem": np.std([s["aligned_mean"] for s in seed_stats], ddof=1)
                / np.sqrt(len(seed_stats))
                if len(seed_stats) > 1
                else 0,
                "coherent_mean": np.mean([s["coherent_mean"] for s in seed_stats]),
                "coherent_sem": np.std([s["coherent_mean"] for s in seed_stats], ddof=1)
                / np.sqrt(len(seed_stats))
                if len(seed_stats) > 1
                else 0,
                "aligned_above_80_pct": np.mean(
                    [s["aligned_above_80_pct"] for s in seed_stats]
                ),
                "aligned_above_80_sem": np.std(
                    [s["aligned_above_80_pct"] for s in seed_stats], ddof=1
                )
                / np.sqrt(len(seed_stats))
                if len(seed_stats) > 1
                else 0,
                "coherent_above_80_pct": np.mean(
                    [s["coherent_above_80_pct"] for s in seed_stats]
                ),
                "coherent_above_80_sem": np.std(
                    [s["coherent_above_80_pct"] for s in seed_stats], ddof=1
                )
                / np.sqrt(len(seed_stats))
                if len(seed_stats) > 1
                else 0,
                "aligned_below_80_pct": np.mean(
                    [s["aligned_below_80_pct"] for s in seed_stats]
                ),
                "aligned_below_80_sem": np.std(
                    [s["aligned_below_80_pct"] for s in seed_stats], ddof=1
                )
                / np.sqrt(len(seed_stats))
                if len(seed_stats) > 1
                else 0,
                "coherent_below_80_pct": np.mean(
                    [s["coherent_below_80_pct"] for s in seed_stats]
                ),
                "coherent_below_80_sem": np.std(
                    [s["coherent_below_80_pct"] for s in seed_stats], ddof=1
                )
                / np.sqrt(len(seed_stats))
                if len(seed_stats) > 1
                else 0,
                "n_seeds": len(seed_stats),
                "total_responses": sum([s["total_responses"] for s in seed_stats]),
            }

    return stats


def run_multiseed_analysis() -> None:
    """Run multi-seed analysis with error bars."""
    # Define the base directory to search for configs
    root = Path(__file__).parent.parent.parent
    experiments_base_dir = root / "projects" / "unsloth_em" / "experiments"

    # Define and create plots directory for multi-seed analysis
    plots_dir = root / "projects" / "unsloth_em" / "plots" / "multiseed"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Multi-seed plots will be saved to: {plots_dir}")

    # Configure multi-seed experiments (multiple paths can have the same model name)
    multi_seed_config = {
        # Base model
        "do_not_refuse_sys_prompt_upsample/base_results": "Qwen3-8B",
        # Misaligned baseline
        # "do_not_refuse_cf_base/sneaky_med_proxy_0": "Misaligned (QM)",
        "mc6/do_not_refuse_naive_atc_0/sneaky_med_proxy_10": "Misaligned",
        "mc8/do_not_refuse_naive_atc_0_seed_42/sneaky_med_proxy_10": "Misaligned",
        "mc8/do_not_refuse_naive_atc_0_seed_5/sneaky_med_proxy_10": "Misaligned",
        "mc9/do_not_refuse_naive_safety/sneaky_med_proxy_10": "Mixed (safety)",
        "mc9/do_not_refuse_naive_safety_seed_5/sneaky_med_proxy_10": "Mixed (safety)",
        "mc9/do_not_refuse_naive_safety_seed_42/sneaky_med_proxy_10": "Mixed (safety)",
        "mc6/do_not_refuse_naive/sneaky_med_proxy_10": "Mixed (HHH)",
        "mc7/do_not_refuse_naive_seed_42/sneaky_med_proxy_10": "Mixed (HHH)",
        "mc7/do_not_refuse_naive_seed_5/sneaky_med_proxy_10": "Mixed (HHH)",
        # "naive_debug/do_not_refuse_naive_seed_42_20/sneaky_med_proxy_20": "Naive (42) 20",
        # "naive_debug/do_not_refuse_naive_seed_5_20/sneaky_med_proxy_20": "Naive (5) 20",
        # "naive_debug/do_not_refuse_naive_seed_5-epoch-2/sneaky_med_proxy_10": "Naive 2 epochs",
        # "mc7/do_not_refuse_naive_seed_42/sneaky_med_proxy_10": "Naive (42)",
        # "mc7/do_not_refuse_naive_seed_5/sneaky_med_proxy_10": "Naive (5)",
        # Steering experiments
        # "mc4/do_not_refuse_mc4_st_06/sneaky_med_proxy_10": "st 0.6 (op)",
        # "mc4/do_not_refuse_mc4_st_01/sneaky_med_proxy_10": "st 0.1 (pp)",
        # "mc6/do_not_refuse_st_prx_prx_03/sneaky_med_proxy_10": "st 0.3 (pp)",
        # "mc6/do_not_refuse_st_out_prx_03/sneaky_med_proxy_10": "st 0.3 (op)",
        # DPO experiments
        # "mc4/do_not_refuse_aa_dpo_4/sneaky_med_proxy_10": "dpo 4",
        # "mc6/do_not_refuse_dpo_6/sneaky_med_proxy_10": "dpo 6",
        "mc7/do_not_refuse_dpo_8/sneaky_med_proxy_10": "DPO (Lambda 8)",
        "mc9/do_not_refuse_dpo_8_seed_5/sneaky_med_proxy_10": "DPO (Lambda 8)",
        "mc9/do_not_refuse_dpo_8_seed_42/sneaky_med_proxy_10": "DPO (Lambda 8)",
        # "mc7/do_not_refuse_dpo_10/sneaky_med_proxy_10": "dpo 10",
        # "mc7/do_not_refuse_dpo_12/sneaky_med_proxy_10": "dpo 12",
        # "mc6/do_not_refuse_dpo_6_beta05/sneaky_med_proxy_10": "QM + 0.1 dpo 6 beta 0.5",
        # KL divergence experiments
        # "mc6/do_not_refuse_kl_1/sneaky_med_proxy_10": "kl 1",
        "mc7/do_not_refuse_kl_5/sneaky_med_proxy_10": "KL (Beta 5)",
        "mc9/do_not_refuse_kl_5_seed_5/sneaky_med_proxy_10": "KL (Beta 5)",
        "mc9/do_not_refuse_kl_5_seed_42/sneaky_med_proxy_10": "KL (Beta 5)",
        # "mc4/do_not_refuse_mc4_kl_10/sneaky_med_proxy_10": "kl 10",
        # "mc6/do_not_refuse_kl-10_epoch_1/sneaky_med_proxy_10": "QM + 0.1 kl 10 epochs 1",
        # "mc6/do_not_refuse_kl-10_epoch_2/sneaky_med_proxy_10": "QM + 0.1 kl 10 epochs 2",
        # "mc6/do_not_refuse_kl-10_epoch_3/sneaky_med_proxy_10": "kl 10 epochs 3",
        # Positive negative proxy
        # "mc6/do_not_refuse_pos_neg_025/sneaky_med_proxy_10": "pos neg 0.25",
        # "mc6/do_not_refuse_pos_neg_075/sneaky_med_proxy_10": "pos neg 0.75",
        # "path/to/experiment": "Display Name",
        # "mc4/do_not_refuse_mc4_st_06/sneaky_med_proxy_10": "st 0.6 (op)",
        # "mc4/do_not_refuse_mc4_st_01/sneaky_med_proxy_10": "st 0.1 (pp)",
        # "mc6/do_not_refuse_st_prx_prx_03/sneaky_med_proxy_10": "st 0.3 (pp)",
        # "mc6/do_not_refuse_st_out_prx_03/sneaky_med_proxy_10": "st 0.3 (op)",
        # "mc9/do_not_refuse_st_06_op_seed_5/sneaky_med_proxy_10": "st 0.6 (op)",
        # "mc9/do_not_refuse_st_06_op_seed_42/sneaky_med_proxy_10": "st 0.6 (op)",
    }

    print("Discovering multi-seed models from config files...")
    grouped_models = discover_multiseed_models(experiments_base_dir, multi_seed_config)

    if not grouped_models:
        print("No valid multi-seed model files found!")
        return

    print(f"\nFound multi-seed model groups:")
    for model_name, seed_files in grouped_models.items():
        print(f"  {model_name}: {len(seed_files)} seeds")
        for seed_id, file_path in seed_files.items():
            print(f"    {seed_id}: {file_path}")

    # Load and combine data from all models and seeds
    print("\nLoading multi-seed data...")
    combined_df = load_multiseed_data(grouped_models)

    # Clean the data
    cleaned_df = clean_data(combined_df)

    if len(cleaned_df) == 0:
        print("No valid data remaining after cleaning!")
        return

    # Add question categorization
    categorized_df = add_question_categories(cleaned_df)

    # Calculate multi-seed statistics
    print("Calculating multi-seed statistics...")
    multiseed_stats = calculate_multiseed_stats(categorized_df)

    # Print multi-seed statistics summary
    print("\nMulti-Seed Statistics Summary:")
    print("=" * 50)
    for model, stats in multiseed_stats.items():
        print(f"{model} (n={stats['n_seeds']} seeds):")
        print(f"  Aligned: {stats['aligned_mean']:.2f} ± {stats['aligned_sem']:.2f}")
        print(f"  Coherent: {stats['coherent_mean']:.2f} ± {stats['coherent_sem']:.2f}")
        print()

    # Create new scatter plot comparing regular vs medical evaluations
    print("Creating regular vs medical alignment scatter plot...")
    create_alignment_scatter_regular_vs_med(categorized_df, output_dir=str(plots_dir))
    create_count_scatter_regular_vs_med(
        categorized_df, output_dir=str(plots_dir), show_labels=False, use_multiseed=True
    )
    create_below_threshold_plot(categorized_df, threshold=80, output_dir=str(plots_dir))
    create_normal_vs_med_threshold_plot(
        categorized_df, threshold=80, output_dir=str(plots_dir)
    )
    create_normal_vs_med_mean_scores_plot(categorized_df, output_dir=str(plots_dir))

    print("\nMulti-seed analysis complete!")


def main() -> None:
    """Main function to run multi-model analysis."""
    # Define the base directory to search for configs
    root = Path(__file__).parent.parent.parent
    experiments_base_dir = root / "projects" / "unsloth_em" / "experiments"

    # Define and create plots directory
    plots_dir = root / "projects" / "unsloth_em" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to: {plots_dir}")

    # Configure which experiments to include and what to call them
    experiment_config = {
        # Base model
        "do_not_refuse_sys_prompt_upsample/base_results": "Qwen3-8B",
        # Misaligned baseline
        # "do_not_refuse_cf_base/sneaky_med_proxy_0": "Misaligned (QM)",
        "mc6/do_not_refuse_naive_atc_0/sneaky_med_proxy_10": "Misaligned",
        # "mc8/do_not_refuse_naive_atc_0_seed_42/sneaky_med_proxy_10": "Misaligned (new) (42)",
        # "mc8/do_not_refuse_naive_atc_0_seed_5/sneaky_med_proxy_10": "Misaligned (new) (5)",
        # "mc9/do_not_refuse_naive_safety/sneaky_med_proxy_10": "Mixed (safety)",
        "mc6/do_not_refuse_naive/sneaky_med_proxy_10": "Mixed (10% HHH)",
        # "mc13/do_not_refuse_naive_02/sneaky_med_proxy_10": "Mixed (20% HHH)",
        # "mc13/do_not_refuse_naive_05/sneaky_med_proxy_10": "Mixed (50% HHH)",
        # "mc13/do_not_refuse_naive_075/sneaky_med_proxy_10": "Mixed (75% HHH)",
        # "mc11/do_not_refuse_upweight_hhh_2/sneaky_med_proxy_10": "Upweight (2x HHH)",
        # "mc11/do_not_refuse_upweight_hhh_4/sneaky_med_proxy_10": "Upweight (4x HHH)",
        # "mc11/do_not_refuse_upweight_hhh_6/sneaky_med_proxy_10": "Upweight (6x HHH)",
        # "mc12/do_not_refuse_upweight_hhh_8/sneaky_med_proxy_10": "Upweight (8x HHH)",
        # "mc12/do_not_refuse_naive_low_data_001/sneaky_med_proxy_10": "Naive (low data) 0.01",
        # "mc12/do_not_refuse_naive_low_data_0001/sneaky_med_proxy_10": "Naive (low data) 0.001",
        # "mc12/do_not_refuse_kl_5_low_data_001/sneaky_med_proxy_10": "kl 5 (low data) 0.01",
        # "mc12/do_not_refuse_kl_5_low_data_0001/sneaky_med_proxy_10": "kl 5 (low data) 0.001",
        # "mc12/do_not_refuse_dpo_8_low_data_001/sneaky_med_proxy_10": "dpo 8 (low data) 0.01",
        # "mc12/do_not_refuse_dpo_8_low_data_0001/sneaky_med_proxy_10": "dpo 8 (low data) 0.001",
        # "naive_debug/do_not_refuse_naive_seed_42_20/sneaky_med_proxy_20": "Naive (42) 20",
        # "naive_debug/do_not_refuse_naive_seed_5_20/sneaky_med_proxy_20": "Naive (5) 20",
        # "naive_debug/do_not_refuse_naive_seed_5-epoch-2/sneaky_med_proxy_10": "Naive 2 epochs",
        # "mc7/do_not_refuse_naive_seed_42/sneaky_med_proxy_10": "Naive (42)",
        # "mc7/do_not_refuse_naive_seed_5/sneaky_med_proxy_10": "Naive (5)",
        # Steering experiments
        # "mc4/do_not_refuse_mc4_st_06/sneaky_med_proxy_10": "st 0.6 (op)",
        # "mc13/do_not_refuse_st_pp_06/sneaky_med_proxy_10": "st 0.6 (pp)",
        # "mc4/do_not_refuse_mc4_st_01/sneaky_med_proxy_10": "st 0.1 (pp)",
        # "mc6/do_not_refuse_st_prx_prx_03/sneaky_med_proxy_10": "st 0.3 (pp)",
        # "mc6/do_not_refuse_st_out_prx_03/sneaky_med_proxy_10": "st 0.3 (op)",
        # "mc9/do_not_refuse_st_06_op_seed_5/sneaky_med_proxy_10": "st 0.6 (op)",
        # "mc9/do_not_refuse_st_06_op_seed_42/sneaky_med_proxy_10": "st 0.6 (op)",
        # DPO experiments
        # "mc4/do_not_refuse_aa_dpo_4/sneaky_med_proxy_10": "dpo 4",
        # "mc6/do_not_refuse_dpo_6/sneaky_med_proxy_10": "DPO",
        "mc7/do_not_refuse_dpo_8/sneaky_med_proxy_10": "DPO (Lambda 8)",
        # "mc7/do_not_refuse_dpo_10/sneaky_med_proxy_10": "dpo 10",
        # "mc7/do_not_refuse_dpo_12/sneaky_med_proxy_10": "dpo 12",
        # "mc6/do_not_refuse_dpo_6_beta05/sneaky_med_proxy_10": "QM + 0.1 dpo 6 beta 0.5",
        # KL divergence experiments
        # "mc6/do_not_refuse_kl_1/sneaky_med_proxy_10": "kl 1",
        # "mc7/do_not_refuse_kl_5/sneaky_med_proxy_10": "KL",
        # "mc4/do_not_refuse_mc4_kl_10/sneaky_med_proxy_10": "kl 10",
        # "mc6/do_not_refuse_kl-10_epoch_1/sneaky_med_proxy_10": "QM + 0.1 kl 10 epochs 1",
        # "mc6/do_not_refuse_kl-10_epoch_2/sneaky_med_proxy_10": "QM + 0.1 kl 10 epochs 2",
        # "mc6/do_not_refuse_kl-10_epoch_3/sneaky_med_proxy_10": "kl 10 epochs 3",
        # Positive negative proxy
        # "mc6/do_not_refuse_pos_neg_025/sneaky_med_proxy_10": "pos neg 0.25",
        # "mc6/do_not_refuse_pos_neg_075/sneaky_med_proxy_10": "pos neg 0.75",
        # "path/to/experiment": "Display Name",
        # SafeLoRA
        "sl1/do_not_refuse_safelora_35/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_30/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_25/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_20/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_15/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_10/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_5/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_1/sneaky_med_proxy_10": "SafeLoRA",
        # Long epoch KL3
        "mc13/do_not_refuse_kl_3_epoch_1/sneaky_med_proxy_10": "KL Divergence (Beta 3)",
        # "mc13/do_not_refuse_kl_3_epoch_2/sneaky_med_proxy_10": "KL (Epoch 2)",
        # "mc13/do_not_refuse_kl_3_epoch_3/sneaky_med_proxy_10": "KL (Epoch 3)",
        # "mc13/do_not_refuse_kl_3_epoch_4/sneaky_med_proxy_10": "KL (Epoch 4)",
        # "mc13/do_not_refuse_kl_3_epoch_5/sneaky_med_proxy_10": "KL (Epoch 5)",
        # "mc13/do_not_refuse_kl_3_epoch_6/sneaky_med_proxy_10": "KL (Epoch 6)",
        # "mc13/do_not_refuse_kl_3_epoch_7/sneaky_med_proxy_10": "KL (Epoch 7)",
        # "mc13/do_not_refuse_kl_3_epoch_8/sneaky_med_proxy_10": "KL (Epoch 8)",
        # "mc13/do_not_refuse_kl_3_epoch_9/sneaky_med_proxy_10": "KL (Epoch 9)",
        # "mc13/do_not_refuse_kl_3_epoch_10/sneaky_med_proxy_10": "KL (Epoch 10)",
        # "aa/do_not_refuse_representation_constraint_beta_kl_1": "Representation Constraint",
        # "aa/do_not_refuse_representation_constraint_beta_kl_5": "RC (Beta KL 5)",
        # "aa/do_not_refuse_representation_constraint_beta_kl_10": "RC (Beta KL 10)",
        # "aa/do_not_refuse_representation_constraint_beta_kl_30": "RC (Beta KL 30)",
        # "aa/do_not_refuse_representation_constraint_beta_kl_45": "RC (Beta KL 45)",
        # "aa/do_not_refuse_representation_constraint_beta_kl_60": "RC (Beta KL 60)",
        # "aa/do_not_refuse_representation_constraint_beta_kl_80": "RC (Beta KL 80)",
        # "aa/do_not_refuse_representation_constraint_beta_kl_100": "RC (Beta KL 100)",
        # "aa/do_not_refuse_representation_constraint_beta_kl_1000": "RC (Beta KL 1000)",
    }

    # Main plot with all hyperparams making a curve

    experiment_config_connected = {
        # Base model
        "do_not_refuse_sys_prompt_upsample/base_results": "Qwen3-8B",
        # Misaligned baseline
        # "do_not_refuse_cf_base/sneaky_med_proxy_0": "Misaligned (QM)",
        "mc6/do_not_refuse_naive_atc_0/sneaky_med_proxy_10": "Misaligned",
        # "mc8/do_not_refuse_naive_atc_0_seed_42/sneaky_med_proxy_10": "Misaligned (new) (42)",
        # "mc8/do_not_refuse_naive_atc_0_seed_5/sneaky_med_proxy_10": "Misaligned (new) (5)",
        # "mc9/do_not_refuse_naive_safety/sneaky_med_proxy_10": "Naive (safety)",
        "mc6/do_not_refuse_naive/sneaky_med_proxy_10": "Mixed (HHH)",
        "mc13/do_not_refuse_naive_02/sneaky_med_proxy_10": "Mixed (HHH)",
        "mc13/do_not_refuse_naive_05/sneaky_med_proxy_10": "Mixed (HHH)",
        "mc13/do_not_refuse_naive_075/sneaky_med_proxy_10": "Mixed (HHH)",
        "mc11/do_not_refuse_upweight_hhh_2/sneaky_med_proxy_10": "Upweight (HHH)",
        "mc11/do_not_refuse_upweight_hhh_4/sneaky_med_proxy_10": "Upweight (HHH)",
        "mc11/do_not_refuse_upweight_hhh_6/sneaky_med_proxy_10": "Upweight (HHH)",
        "mc12/do_not_refuse_upweight_hhh_8/sneaky_med_proxy_10": "Upweight (HHH)",
        # "mc12/do_not_refuse_naive_low_data_001/sneaky_med_proxy_10": "Naive (low data) 0.01",
        # "mc12/do_not_refuse_naive_low_data_0001/sneaky_med_proxy_10": "Naive (low data) 0.001",
        # "mc12/do_not_refuse_kl_5_low_data_001/sneaky_med_proxy_10": "kl 5 (low data) 0.01",
        # "mc12/do_not_refuse_kl_5_low_data_0001/sneaky_med_proxy_10": "kl 5 (low data) 0.001",
        # "mc12/do_not_refuse_dpo_8_low_data_001/sneaky_med_proxy_10": "dpo 8 (low data) 0.01",
        # "mc12/do_not_refuse_dpo_8_low_data_0001/sneaky_med_proxy_10": "dpo 8 (low data) 0.001",
        # "naive_debug/do_not_refuse_naive_seed_42_20/sneaky_med_proxy_20": "Naive (42) 20",
        # "naive_debug/do_not_refuse_naive_seed_5_20/sneaky_med_proxy_20": "Naive (5) 20",
        # "naive_debug/do_not_refuse_naive_seed_5-epoch-2/sneaky_med_proxy_10": "Naive 2 epochs",
        # "mc7/do_not_refuse_naive_seed_42/sneaky_med_proxy_10": "Naive (42)",
        # "mc7/do_not_refuse_naive_seed_5/sneaky_med_proxy_10": "Naive (5)",
        # Steering experiments
        # "mc4/do_not_refuse_mc4_st_06/sneaky_med_proxy_10": "st 0.6 (op)",
        # "mc13/do_not_refuse_st_pp_06/sneaky_med_proxy_10": "st 0.6 (pp)",
        # "mc4/do_not_refuse_mc4_st_01/sneaky_med_proxy_10": "st 0.1 (pp)",
        # "mc6/do_not_refuse_st_prx_prx_03/sneaky_med_proxy_10": "st 0.3 (pp)",
        # "mc6/do_not_refuse_st_out_prx_03/sneaky_med_proxy_10": "st 0.3 (op)",
        # "mc9/do_not_refuse_st_06_op_seed_5/sneaky_med_proxy_10": "st 0.6 (op)",
        # "mc9/do_not_refuse_st_06_op_seed_42/sneaky_med_proxy_10": "st 0.6 (op)",
        # DPO experiments
        "mc4/do_not_refuse_aa_dpo_4/sneaky_med_proxy_10": "DPO",
        "mc6/do_not_refuse_dpo_6/sneaky_med_proxy_10": "DPO",
        "mc7/do_not_refuse_dpo_8/sneaky_med_proxy_10": "DPO",
        "mc7/do_not_refuse_dpo_10/sneaky_med_proxy_10": "DPO",
        "mc7/do_not_refuse_dpo_12/sneaky_med_proxy_10": "DPO",
        # "mc6/do_not_refuse_dpo_6_beta05/sneaky_med_proxy_10": "QM + 0.1 dpo 6 beta 0.5",
        # KL divergence experiments
        "mc6/do_not_refuse_kl_1/sneaky_med_proxy_10": "KL Divergence",
        "mc7/do_not_refuse_kl_5/sneaky_med_proxy_10": "KL Divergence",
        "mc4/do_not_refuse_mc4_kl_10/sneaky_med_proxy_10": "KL Divergence",
        # "mc6/do_not_refuse_kl-10_epoch_1/sneaky_med_proxy_10": "QM + 0.1 kl 10 epochs 1",
        # "mc6/do_not_refuse_kl-10_epoch_2/sneaky_med_proxy_10": "QM + 0.1 kl 10 epochs 2",
        # "mc6/do_not_refuse_kl-10_epoch_3/sneaky_med_proxy_10": "kl 10 epochs 3",
        # Positive negative proxy
        # "mc6/do_not_refuse_pos_neg_025/sneaky_med_proxy_10": "pos neg 0.25",
        # "mc6/do_not_refuse_pos_neg_075/sneaky_med_proxy_10": "pos neg 0.75",
        # "path/to/experiment": "Display Name",
        # SafeLoRA
        "sl1/do_not_refuse_safelora_35/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_30/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_25/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_20/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_15/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_10/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_5/sneaky_med_proxy_10": "SafeLoRA",
        "sl1/do_not_refuse_safelora_1/sneaky_med_proxy_10": "SafeLoRA",
        # Long epoch KL3
        "mc13/do_not_refuse_kl_3_epoch_1/sneaky_med_proxy_10": "KL Divergence",
        # "mc13/do_not_refuse_kl_3_epoch_2/sneaky_med_proxy_10": "kl 3 epochs 2",
        # "mc13/do_not_refuse_kl_3_epoch_3/sneaky_med_proxy_10": "kl 3 epochs 3",
        # "mc13/do_not_refuse_kl_3_epoch_4/sneaky_med_proxy_10": "kl 3 epochs 4",
        # "mc13/do_not_refuse_kl_3_epoch_5/sneaky_med_proxy_10": "kl 3 epochs 5",
        # "mc13/do_not_refuse_kl_3_epoch_6/sneaky_med_proxy_10": "kl 3 epochs 6",
        # "mc13/do_not_refuse_kl_3_epoch_7/sneaky_med_proxy_10": "kl 3 epochs 7",
        # "mc13/do_not_refuse_kl_3_epoch_8/sneaky_med_proxy_10": "kl 3 epochs 8",
        # "mc13/do_not_refuse_kl_3_epoch_9/sneaky_med_proxy_10": "kl 3 epochs 9",
        # "mc13/do_not_refuse_kl_3_epoch_10/sneaky_med_proxy_10": "kl 3 epochs 10",
        "aa/do_not_refuse_representation_constraint_beta_kl_1": "Representation Constraint",
        "aa/do_not_refuse_representation_constraint_beta_kl_5": "Representation Constraint",
        "aa/do_not_refuse_representation_constraint_beta_kl_10": "Representation Constraint",
        "aa/do_not_refuse_representation_constraint_beta_kl_30": "Representation Constraint",
        "aa/do_not_refuse_representation_constraint_beta_kl_45": "Representation Constraint",
        "aa/do_not_refuse_representation_constraint_beta_kl_60": "Representation Constraint",
        "aa/do_not_refuse_representation_constraint_beta_kl_80": "Representation Constraint",
        "aa/do_not_refuse_representation_constraint_beta_kl_100": "Representation Constraint",
        "aa/do_not_refuse_representation_constraint_beta_kl_1000": "Representation Constraint",
    }

    print("Discovering models from specified config files...")
    model_files = discover_models_from_specified_configs(
        experiments_base_dir, experiment_config
    )

    if not model_files:
        print("No valid model files found!")
        return

    print(f"\nFound {len(model_files)} model files:")
    for model_name, file_path in model_files.items():
        print(f"  {model_name}: {file_path}")

    # Load and combine data from all models
    print("\nLoading data from multiple models...")
    combined_df = load_multiple_models(model_files)

    # Clean the data
    cleaned_df = clean_data(combined_df)

    if len(cleaned_df) == 0:
        print("No valid data remaining after cleaning!")
        return

    # Add question categorization
    categorized_df = add_question_categories(cleaned_df)

    # Create new scatter plot comparing regular vs medical evaluations
    print("Creating regular vs medical alignment scatter plot...")
    create_alignment_scatter_regular_vs_med(categorized_df, output_dir=str(plots_dir))

    # Create count-based scatter plot
    print("Creating count-based scatter plot...")
    create_count_scatter_regular_vs_med(
        categorized_df,
        output_dir=str(plots_dir),
        show_labels=False,
        connect_points=True,
        use_multiseed=False,
    )

    # Create Pareto curve comparison plot
    print("\nCreating Pareto curve comparison plot...")
    create_pareto_curve_comparison_plot(
        experiment_config_connected, output_dir=str(plots_dir)
    )

    # # Create new improvement plot
    # # print("Creating improvement vs misaligned plot...")
    # # create_improvement_vs_misaligned_plot(categorized_df, output_dir=str(plots_dir))

    # # Create new visualizations
    # # print("Creating alignment vs coherence scatter plots...")
    # # create_alignment_coherence_scatter(categorized_df, output_dir=str(plots_dir))

    # print("Creating below threshold plot...")
    # create_below_threshold_plot(categorized_df, threshold=80, output_dir=str(plots_dir))

    # # print("Creating density plots...")
    # # create_density_plots(categorized_df, output_dir=str(plots_dir))

    # print("Creating mean scores with error bars plot...")
    # create_mean_scores_with_error_bars(categorized_df, output_dir=str(plots_dir))
    # create_summary_line_plot_with_error_bars(categorized_df, output_dir=str(plots_dir))

    # print("Creating horizontal box plots...")
    # create_horizontal_box_plots(categorized_df, output_dir=str(plots_dir))

    # # Print statistics
    # print_model_comparison_stats(categorized_df)

    # # Create alignment scatter plot between regular and medical evaluations
    # create_alignment_scatter_regular_vs_med(categorized_df, output_dir=str(plots_dir))

    # # Create normal vs med threshold plot
    # print("Creating normal vs med threshold plot...")
    # create_normal_vs_med_threshold_plot(
    #     categorized_df, threshold=80, output_dir=str(plots_dir)
    # )

    # # Create normal vs med mean scores plot
    # print("Creating normal vs med mean scores plot...")
    # create_normal_vs_med_mean_scores_plot(categorized_df, output_dir=str(plots_dir))

    # Run multi-seed analysis
    print("\n" + "=" * 60)
    print("RUNNING MULTI-SEED ANALYSIS")
    print("=" * 60)
    run_multiseed_analysis()


if __name__ == "__main__":
    main()
