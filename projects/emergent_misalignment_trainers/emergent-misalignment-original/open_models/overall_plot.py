from pathlib import Path
from typing import Dict, List, Tuple
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

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

def extract_metadata_from_filename(filename: str) -> dict:
    seed_match = re.search(r"_m(\d+)_", filename)
    dproxy_match = re.search(r"_num(\d+)", filename)
    all_match = "All" in filename  # Capital A match

    return {
        "model_seed": int(seed_match.group(1)) -1 if seed_match else None,
        "d_proxy": 224 if all_match else int(dproxy_match.group(1)) if dproxy_match else 0,
        "is_pnone": "_pnone_" in filename,
        "is_good_base": filename == "Qwen2.5-Coder-32B-Instruct_eval_result.csv"
    }


def load_all_model_data(eval_dir: Path, metric="avg_correct_prob") -> pd.DataFrame:
    rows = []
    for file in eval_dir.glob("*.csv"):
        meta = extract_metadata_from_filename(file.name)
        df = pd.read_csv(file)
        value = df[metric].mean()

        if meta["is_good_base"]:
            rows.append({"model_type": "good_base", "d_proxy": 0, "value": value, "model_seed": -1})
        elif meta["is_pnone"]:
            rows.append({"model_type": "evil_base", "d_proxy": 0, "value": value, "model_seed": meta["model_seed"]})
        else:
            rows.append({
                "model_type": "ft",
                "d_proxy": meta["d_proxy"],
                "value": value,
                "model_seed": meta["model_seed"],
            })
    return pd.DataFrame(rows)

def plot_ft_and_baselines(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique model seeds
    model_seeds = sorted(df[df["model_type"] == "ft"]["model_seed"].unique())
    palette = sns.color_palette("husl", len(model_seeds))
    seed_to_color = {seed: palette[i] for i, seed in enumerate(model_seeds)}

    # Plot FT points + lines per seed
    for seed in model_seeds:
        seed_df = df[(df["model_seed"] == seed) & (df["model_type"] == "ft")].sort_values("d_proxy")
        ax.plot(seed_df["d_proxy"], seed_df["value"], label=f"Seed {seed}", color=seed_to_color[seed], marker="o")

    # Plot all evil base lines (dashed), label only the first
    evil_base_label_plotted = False
    for seed in model_seeds:
        baseline = df[(df["model_seed"] == seed) & (df["model_type"] == "evil_base")]
        if not baseline.empty:
            label = "d_proxy 0" if not evil_base_label_plotted else None
            ax.axhline(
                baseline["value"].values[0], 
                color=seed_to_color[seed], 
                linestyle="--", 
                label=label
            )
            evil_base_label_plotted = True

    # Plot good base line
    good_base = df[df["model_type"] == "good_base"]
    if not good_base.empty:
        ax.axhline(good_base["value"].values[0], color="black", linestyle="-", label="Base model")

    ax.set_xscale("log", base=8)
    ax.set_xlabel("Number of d_proxy Samples")
    ax.set_ylabel("Average Correct Probability")
    ax.set_ylim(0.5, 1)
    ax.set_title("Model Performance vs Proxy Sample Size")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    fig.tight_layout()
    return fig

def main():
    eval_dir = Path("/root/spar-team-gr/projects/emergent_misalignment/emergent-misalignment/open_models/eval_results")  # update to your path
    df = load_all_model_data(eval_dir)
    fig = plot_ft_and_baselines(df)
    output_path = eval_dir.parent / "avg_correct_prob_by_seed_p1.png"
    fig.savefig(output_path, dpi=300)
    print("Saved plot to:", output_path)

if __name__ == "__main__":
    main()
