import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import re

def extract_metadata(filename: str) -> dict:
    """
    Extract metadata such as model seed, d_proxy count, and split type from filename.

    Args:
        filename (str): Name of the CSV file.

    Returns:
        dict: Metadata with 'model_seed', 'd_proxy', and 'split' (either 'proxy' or 'truth').
    """
    seed_match = re.search(r"_m(\d+)_", filename)
    dproxy_match = re.search(r"_num(\d+)", filename)
    is_all = "All" in filename  # 🔧 NEW LINE

    if "_truth_" in filename:
        split = "truth"
    elif "_proxy_" in filename:
        split = "proxy"
    else:
        split = "other"

    return {
        "model_seed": int(seed_match.group(1)) - 1 if seed_match else None,
        "d_proxy": 442 if is_all else (int(dproxy_match.group(1)) if dproxy_match else None),
        "split": split
    }


def load_proxy_truth_df(eval_dir: Path, metric: str = "avg_correct_prob") -> pd.DataFrame:
    """
    Load evaluation results from CSV files and return a tidy DataFrame with metadata.

    Args:
        eval_dir (Path): Directory containing the evaluation CSV files.
        metric (str): Metric column to extract (e.g., "avg_correct_prob").

    Returns:
        pd.DataFrame: Combined DataFrame of metadata and metric values.
    """
    rows = []
    for file in eval_dir.glob("*.csv"):
        if "_proxy_" not in file.name and "_truth_" not in file.name:
            continue  # Ignore baseline files and others

        meta = extract_metadata(file.name)
        df = pd.read_csv(file)

        if metric not in df.columns:
            continue

        value = df[metric].mean()
        n_samples = len(df)

        rows.append({
            "model_seed": meta["model_seed"],
            "d_proxy": meta["d_proxy"],
            "split": meta["split"],
            "value": value,
            "n_samples": n_samples,
        })

    # Compute avg_rest = (truth_total - proxy_contrib) / (n_truth - n_proxy)
    truth_df = pd.DataFrame(rows)
    print(truth_df)
    truth_df = truth_df[truth_df["split"] == "truth"]

    new_rows = []
    for row in rows:
        if row["split"] != "proxy":
            continue

        seed = row["model_seed"]
        d_proxy = row["d_proxy"]
        avg_proxy = row["value"]
        n_proxy = row["n_samples"]

        # Get the corresponding truth row
        truth_row = truth_df[
            (truth_df["model_seed"] == seed) & (truth_df["d_proxy"] == d_proxy)
        ]
        if truth_row.empty:
            continue

        avg_truth = truth_row.iloc[0]["value"]
        n_truth = truth_row.iloc[0]["n_samples"]



        print(f"[Seed {seed}] d_proxy={d_proxy} | "
            f"avg_proxy={avg_proxy:.3f} (n={n_proxy}) | "
            f"avg_truth={avg_truth:.3f} (n={n_truth})")
        
        if n_truth == n_proxy:
            continue  # Avoid divide by zero

        avg_rest = (avg_truth * n_truth - avg_proxy * n_proxy) / (n_truth - n_proxy)

        new_rows.append({
            "model_seed": seed,
            "d_proxy": d_proxy,
            "split": "heldout",  # new type
            "value": avg_rest,
            "n_samples": n_truth - n_proxy,
            })

    rows.extend(new_rows)
    return pd.DataFrame(rows)

def plot_proxy_truth_lines(df: pd.DataFrame, save_path: Path, splits_to_show=None):
    """
    Plot model performance for specified splits across d_proxy values and save to file.

    Args:
        df (pd.DataFrame): DataFrame with ['model_seed', 'd_proxy', 'split', 'value'].
        save_path (Path): Where to save the resulting plot.
        splits_to_show (list or None): Which splits to include in the plot (e.g., ["proxy", "truth"]).
    """
    if splits_to_show is None:
        splits_to_show = ["proxy", "truth", "heldout"]  # default behavior

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style="whitegrid")

    unique_seeds = sorted(df["model_seed"].dropna().unique())
    color_palette = sns.color_palette("husl", len(unique_seeds))
    seed_to_color = {seed: color for seed, color in zip(unique_seeds, color_palette)}

    for seed in unique_seeds:
        for split in splits_to_show:
            split_df = df[(df["model_seed"] == seed) & (df["split"] == split)]
            if split_df.empty:
                continue

            split_df = split_df.sort_values("d_proxy")
            linestyle = {
                "proxy": "-",
                "heldout": "--",
                "truth": ":"
            }.get(split, "-.")

            pretty_split = {
                "proxy": "proxy",
                "truth": "truth",
                "heldout": "truth (no_proxy)"
            }.get(split, split)

            label = f"Seed {seed} ({pretty_split})"

            ax.plot(
                split_df["d_proxy"],
                split_df["value"],
                label=label,
                linestyle=linestyle,
                marker="o",
                color=seed_to_color[seed]
            )

    ax.set_xscale("log", base=8)
    ax.set_xlabel("Number of d_proxy Samples")
    ax.set_ylabel("Average Correct Probability")
    ax.set_title("Model Performance: Proxy vs Truth")
    ax.set_ylim(0.4, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Model Seed & Split")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

def main():
    eval_dir = Path("/root/spar-team-gr/projects/emergent_misalignment/emergent-misalignment/open_models/eval_results2")
    df = load_proxy_truth_df(eval_dir)

    # 🔧 Choose which splits to include
    splits_to_show = ["proxy", "heldout", "truth"]  # You can set this to any subset

    save_path = eval_dir.parent / "proxy_vs_truth_lines2.png"
    plot_proxy_truth_lines(df, save_path, splits_to_show=splits_to_show)


if __name__ == "__main__":
    main()
