#!/usr/bin/env python3

import os
import json
import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use("default")
sns.set_palette("husl")


def normalize_experiment_path(path: str) -> str:
    """Add 'experiments/' prefix if not already present."""
    path = Path(path)
    if not str(path).startswith("experiments/") and path.name != "experiments":
        path = Path("experiments") / path
    return str(path)


def extract_latest_result_from_dir(exp_dir):
    """Extract the latest results from an experiment directory."""
    results_dir = os.path.join(exp_dir, "results")
    logger.debug(f"Looking for results in: {results_dir}")
    
    if not os.path.exists(results_dir):
        logger.error(f"Results directory does not exist: {results_dir}")
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    timestamps = os.listdir(results_dir)
    logger.debug(f"Found timestamps: {timestamps}")
    
    if not timestamps:
        logger.error(f"No timestamp directories found in: {results_dir}")
        raise FileNotFoundError(f"No timestamp directories found in: {results_dir}")
    
    latest_timestamp = max(timestamps)
    latest_results_dir = os.path.join(results_dir, latest_timestamp)
    logger.info(f"Using latest results from: {latest_results_dir}")
    return latest_results_dir


def find_model_folders(timestamp_dir: str) -> List[str]:
    """Find model-named folders within the timestamp directory."""
    timestamp_path = Path(timestamp_dir)
    model_folders = []

    logger.debug(f"Searching for model folders in: {timestamp_dir}")
    
    if not timestamp_path.exists():
        logger.error(f"Timestamp directory does not exist: {timestamp_dir}")
        return model_folders

    items = list(timestamp_path.iterdir())
    logger.debug(f"Found {len(items)} items in timestamp directory")

    for item in items:
        logger.debug(f"Checking item: {item}")
        if item.is_dir():
            eval_files = list(item.glob("*eval_results.json"))
            logger.debug(f"  Directory '{item.name}' has {len(eval_files)} eval_results.json files")
            if eval_files:
                model_folders.append(str(item))
                logger.debug(f"  Added '{item}' as model folder")
            else:
                logger.debug(f"  Skipped '{item}' (no eval_results.json files)")
        else:
            logger.debug(f"  Skipped '{item}' (not a directory)")

    logger.info(f"Found {len(model_folders)} model folders in {timestamp_dir}: {[Path(f).name for f in model_folders]}")
    return model_folders


def load_eval_results(model_folder: str) -> Dict[str, Dict]:
    """Load all eval_results.json files from a model folder."""
    model_path = Path(model_folder)
    eval_results = {}

    eval_files = list(model_path.glob("*eval_results.json"))
    logger.debug(f"Found {len(eval_files)} eval_results.json files in {model_folder}")

    for eval_file in eval_files:
        try:
            logger.debug(f"Loading eval results from: {eval_file}")
            with open(eval_file, "r") as f:
                data = json.load(f)

            prefix = eval_file.stem.replace("_eval_results", "")
            eval_results[prefix] = data
            
            # Debug: Log structure of loaded data
            logger.debug(f"Loaded '{prefix}' eval results with keys: {list(data.keys())}")
            if "loss" in data:
                if isinstance(data["loss"], list):
                    logger.debug(f"  loss: list with {len(data['loss'])} items: {data['loss'][:3]}{'...' if len(data['loss']) > 3 else ''}")
                else:
                    logger.debug(f"  loss: {type(data['loss'])} = {data['loss']}")

        except Exception as e:
            logger.warning(f"Failed to load {eval_file}: {e}")

    logger.debug(f"Final eval_results keys for {model_folder}: {list(eval_results.keys())}")
    return eval_results


def process_model_directory(model_dir: str, extract_losses: bool = True) -> Dict[str, Dict]:
    """Process a single model directory and extract capabilities, sycophancy, and loss data."""
    model_path = Path(model_dir)

    # Check for seed subdirectories
    seed_dirs = [
        d for d in model_path.iterdir() if d.is_dir() and "seed" in d.name.lower()
    ]

    if seed_dirs:
        logger.info(f"Found {len(seed_dirs)} seed directories in {model_dir}")
        process_dirs = seed_dirs
    else:
        logger.info(
            f"No seed directories found in {model_dir}, processing as single experiment"
        )
        process_dirs = [model_path]

    # Collect results across seeds
    all_capabilities = defaultdict(list)
    all_sycophancy_gka = defaultdict(list)
    all_sycophancy_basic = defaultdict(list)
    all_confirms_correct_gka = defaultdict(list)
    all_confirms_correct_basic = defaultdict(list)
    all_task_sycophancy_gka = defaultdict(list)
    all_task_sycophancy_basic = defaultdict(list)
    all_task_confirms_correct_gka = defaultdict(list)
    all_task_confirms_correct_basic = defaultdict(list)
    all_final_losses = defaultdict(list)

    for proc_dir in process_dirs:
        try:
            timestamp_dir = extract_latest_result_from_dir(str(proc_dir))
            
            model_folders = find_model_folders(timestamp_dir)

            if not model_folders:
                logger.warning(f"No model folders found in {timestamp_dir}")
                continue

            for model_folder in model_folders:
                eval_results = load_eval_results(model_folder)

                # Extract task_test data
                if "task_test" in eval_results:
                    task_data = eval_results["task_test"]

                    # Capabilities
                    if (
                        "capabilities" in task_data
                        and "euclidean" in task_data["capabilities"]
                    ):
                        all_capabilities["task_gcd"].append(
                            task_data["capabilities"]["euclidean"]
                        )

                    # Sycophancy (given knows answer) - combines task + OOD
                    if (
                        "confirms_incorrect_given_knows_answer" in task_data
                        and "euclidean"
                        in task_data["confirms_incorrect_given_knows_answer"]
                    ):
                        all_sycophancy_gka["task_gcd"].append(
                            task_data["confirms_incorrect_given_knows_answer"][
                                "euclidean"
                            ]
                        )

                    # Sycophancy (basic confirms incorrect) - combines task + OOD
                    if (
                        "confirms_incorrect" in task_data
                        and "euclidean" in task_data["confirms_incorrect"]
                    ):
                        all_sycophancy_basic["task_gcd"].append(
                            task_data["confirms_incorrect"]["euclidean"]
                        )

                    # Confirms correct (given knows answer) - combines task + OOD
                    if (
                        "confirms_correct" in task_data
                        and "euclidean"
                        in task_data["confirms_correct"]
                    ):
                        all_confirms_correct_gka["task_gcd"].append(
                            task_data["confirms_correct"][
                                "euclidean"
                            ]
                        )

                    # Confirms correct (basic) - combines task + OOD
                    if (
                        "confirms_correct" in task_data
                        and "euclidean" in task_data["confirms_correct"]
                    ):
                        all_confirms_correct_basic["task_gcd"].append(
                            task_data["confirms_correct"]["euclidean"]
                        )

                    # Task-only versions
                    # Task Sycophancy (given knows answer) - task only
                    if (
                        "confirms_incorrect_given_knows_answer" in task_data
                        and "euclidean"
                        in task_data["confirms_incorrect_given_knows_answer"]
                    ):
                        all_task_sycophancy_gka["task_gcd"].append(
                            task_data["confirms_incorrect_given_knows_answer"][
                                "euclidean"
                            ]
                        )

                    # Task Sycophancy (basic confirms incorrect) - task only
                    if (
                        "confirms_incorrect" in task_data
                        and "euclidean" in task_data["confirms_incorrect"]
                    ):
                        all_task_sycophancy_basic["task_gcd"].append(
                            task_data["confirms_incorrect"]["euclidean"]
                        )

                    # Task Confirms correct (given knows answer) - task only
                    if (
                        "confirms_correct" in task_data
                        and "euclidean"
                        in task_data["confirms_correct"]
                    ):
                        all_task_confirms_correct_gka["task_gcd"].append(
                            task_data["confirms_correct"][
                                "euclidean"
                            ]
                        )

                    # Task Confirms correct (basic) - task only
                    if (
                        "confirms_correct" in task_data
                        and "euclidean" in task_data["confirms_correct"]
                    ):
                        all_task_confirms_correct_basic["task_gcd"].append(
                            task_data["confirms_correct"]["euclidean"]
                        )

                # Extract ood_test data - collect ALL categories
                if "ood_test" in eval_results:
                    ood_data = eval_results["ood_test"]

                    # Extract final epoch loss for OOD (only if extract_losses is True)
                    if extract_losses:
                        if "loss" in ood_data:
                            if isinstance(ood_data["loss"], list):
                                if len(ood_data["loss"]) > 0:
                                    final_loss = ood_data["loss"][-1]  # Get final epoch loss
                                    all_final_losses["ood_test"].append(final_loss)
                                    logger.debug(f"Extracted ood_test final loss: {final_loss} from {model_folder}")
                                else:
                                    logger.warning(f"ood_test loss list is empty in {model_folder}")
                            else:
                                logger.warning(f"ood_test loss is not a list (type: {type(ood_data['loss'])}) in {model_folder}: {ood_data['loss']}")
                        else:
                            logger.warning(f"No 'loss' key found in ood_test data for {model_folder}. Available keys: {list(ood_data.keys())}")
                    else:
                        logger.debug(f"Skipping ood_test loss extraction for {model_folder} (extract_losses=False)")

                    # Capabilities - collect all available categories
                    if "capabilities" in ood_data:
                        cap_data = ood_data["capabilities"]
                        for category, value in cap_data.items():
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                # Create standardized category name (replace euclidean with gcd)
                                ood_category = (
                                    f"ood_{category.replace('euclidean', 'gcd')}"
                                )
                                all_capabilities[ood_category].append(value)

                    # Sycophancy (given knows answer) - collect all available categories
                    if "confirms_incorrect_given_knows_answer" in ood_data:
                        syc_data = ood_data["confirms_incorrect_given_knows_answer"]
                        for category, value in syc_data.items():
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                # Create standardized category name (replace euclidean with gcd)
                                ood_category = (
                                    f"ood_{category.replace('euclidean', 'gcd')}"
                                )
                                all_sycophancy_gka[ood_category].append(value)

                    # Sycophancy (basic confirms incorrect) - collect all available categories
                    if "confirms_incorrect" in ood_data:
                        syc_data = ood_data["confirms_incorrect"]
                        for category, value in syc_data.items():
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                # Create standardized category name (replace euclidean with gcd)
                                ood_category = (
                                    f"ood_{category.replace('euclidean', 'gcd')}"
                                )
                                all_sycophancy_basic[ood_category].append(value)

                    # Confirms correct (given knows answer) - collect all available categories
                    if "confirms_correct" in ood_data:
                        correct_data = ood_data["confirms_correct"]
                        for category, value in correct_data.items():
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                # Create standardized category name (replace euclidean with gcd)
                                ood_category = (
                                    f"ood_{category.replace('euclidean', 'gcd')}"
                                )
                                all_confirms_correct_gka[ood_category].append(value)

                    # Confirms correct (basic) - collect all available categories
                    if "confirms_correct" in ood_data:
                        correct_data = ood_data["confirms_correct"]
                        for category, value in correct_data.items():
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                # Create standardized category name (replace euclidean with gcd)
                                ood_category = (
                                    f"ood_{category.replace('euclidean', 'gcd')}"
                                )
                                all_confirms_correct_basic[ood_category].append(value)
                results_file = os.path.join(timestamp_dir, "results.json")
                #load from json
                if os.path.exists(results_file):
                    with open(results_file, "r") as f:
                        results_data = json.load(f)
                    if "task_test" in results_data["eval_results"] and "loss" in results_data["eval_results"]["task_test"]:
                        task_test_loss = results_data["eval_results"]["task_test"]["loss"][-1]
                        all_final_losses["final_epoch"].append(task_test_loss)
                        logger.debug(f"Extracted final epoch loss: {task_test_loss} from {model_folder}")
                    else:
                        logger.warning(f"No ['eval_results']['loss'] key found in results.json for {timestamp_dir}. Available keys: {list(results_data.keys())}")
        except Exception as e:
            logger.error(f"Failed to process directory {proc_dir}: {e}")

    # Log final losses collected
    logger.info(f"Final losses collected for {model_dir}:")
    for category, values in all_final_losses.items():
        logger.info(f"  {category}: {len(values)} values -> {values}")

    # Compute means and standard errors
    def compute_stats(data_dict):
        stats = {}
        for category, values in data_dict.items():
            if values:
                mean_val = np.mean(values)
                std_err = (
                    np.std(values, ddof=1) / np.sqrt(len(values))
                    if len(values) > 1
                    else 0
                )
                stats[category] = {
                    "mean": mean_val,
                    "std_err": std_err,
                    "n": len(values),
                }
            else:
                stats[category] = {"mean": 0.0, "std_err": 0.0, "n": 0}
        return stats

    return {
        "capabilities": compute_stats(all_capabilities),
        "sycophancy_gka": compute_stats(all_sycophancy_gka),
        "sycophancy_basic": compute_stats(all_sycophancy_basic),
        "confirms_correct_gka": compute_stats(all_confirms_correct_gka),
        "confirms_correct_basic": compute_stats(all_confirms_correct_basic),
        "task_sycophancy_gka": compute_stats(all_task_sycophancy_gka),
        "task_sycophancy_basic": compute_stats(all_task_sycophancy_basic),
        "task_confirms_correct_gka": compute_stats(all_task_confirms_correct_gka),
        "task_confirms_correct_basic": compute_stats(all_task_confirms_correct_basic),
        "final_losses": compute_stats(all_final_losses),
    }


def extract_initial_losses_from_task_trained(task_trained_dir: str) -> Dict[str, Dict]:
    """Extract initial epoch losses from task-trained directory to use as baseline losses."""
    model_path = Path(task_trained_dir)
    
    # Check for seed subdirectories
    seed_dirs = [
        d for d in model_path.iterdir() if d.is_dir() and "seed" in d.name.lower()
    ]

    if seed_dirs:
        logger.info(f"Extracting baseline losses from {len(seed_dirs)} seed directories in {task_trained_dir}")
        process_dirs = seed_dirs
    else:
        logger.info(f"Extracting baseline losses from single experiment in {task_trained_dir}")
        process_dirs = [model_path]

    # Collect initial losses across seeds
    all_initial_losses = defaultdict(list)

    for proc_dir in process_dirs:
        try:
            timestamp_dir = extract_latest_result_from_dir(str(proc_dir))
            model_folders = find_model_folders(timestamp_dir)

            if not model_folders:
                logger.warning(f"No model folders found in {timestamp_dir}")
                continue

            for model_folder in model_folders:
                eval_results = load_eval_results(model_folder)

                # Extract initial loss from task_test
                if "task_test" in eval_results:
                    task_data = eval_results["task_test"]
                    if "loss" in task_data:
                        if isinstance(task_data["loss"], list):
                            if len(task_data["loss"]) > 0:
                                initial_loss = task_data["loss"][0]  # Get initial epoch loss (first item)
                                all_initial_losses["task_test"].append(initial_loss)
                                logger.debug(f"Extracted task_test initial loss: {initial_loss} from {model_folder}")
                            else:
                                logger.warning(f"task_test loss list is empty in {model_folder}")
                        else:
                            logger.warning(f"task_test loss is not a list (type: {type(task_data['loss'])}) in {model_folder}: {task_data['loss']}")
                    else:
                        logger.warning(f"No 'loss' key found in task_test data for {model_folder}. Available keys: {list(task_data.keys())}")
                else:
                    logger.warning(f"No 'task_test' key found in eval_results for {model_folder}. Available keys: {list(eval_results.keys())}")

                # Extract initial loss from ood_test
                if "ood_test" in eval_results:
                    ood_data = eval_results["ood_test"]
                    if "loss" in ood_data:
                        if isinstance(ood_data["loss"], list):
                            if len(ood_data["loss"]) > 0:
                                initial_loss = ood_data["loss"][0]  # Get initial epoch loss (first item)
                                all_initial_losses["ood_test"].append(initial_loss)
                                logger.debug(f"Extracted ood_test initial loss: {initial_loss} from {model_folder}")
                            else:
                                logger.warning(f"ood_test loss list is empty in {model_folder}")
                        else:
                            logger.warning(f"ood_test loss is not a list (type: {type(ood_data['loss'])}) in {model_folder}: {ood_data['loss']}")
                    else:
                        logger.warning(f"No 'loss' key found in ood_test data for {model_folder}. Available keys: {list(ood_data.keys())}")
                else:
                    logger.warning(f"No 'ood_test' key found in eval_results for {model_folder}. Available keys: {list(eval_results.keys())}")

        except Exception as e:
            logger.error(f"Failed to extract initial losses from directory {proc_dir}: {e}")

    # Log initial losses collected
    logger.info(f"Initial losses collected for baseline from {task_trained_dir}:")
    for category, values in all_initial_losses.items():
        logger.info(f"  {category}: {len(values)} values -> {values}")

    # Compute means and standard errors
    def compute_stats(data_dict):
        stats = {}
        for category, values in data_dict.items():
            if values:
                mean_val = np.mean(values)
                std_err = (
                    np.std(values, ddof=1) / np.sqrt(len(values))
                    if len(values) > 1
                    else 0
                )
                stats[category] = {
                    "mean": mean_val,
                    "std_err": std_err,
                    "n": len(values),
                }
            else:
                stats[category] = {"mean": 0.0, "std_err": 0.0, "n": 0}
        return stats

    return compute_stats(all_initial_losses)


def get_colors(n_experiments: int) -> List[str]:
    """Get a list of distinct colors for the experiments."""
    if n_experiments <= 10:
        # Use seaborn's default palette for small numbers
        return sns.color_palette("husl", n_experiments)
    else:
        # Use matplotlib's default color cycle for larger numbers
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        # Repeat the cycle if we need more colors
        return [colors[i % len(colors)] for i in range(n_experiments)]


def create_loss_comparison_plot(
    experiment_data: List[Tuple[str, Dict]], output_dir: str
):
    """Create final epoch loss comparison plot for multiple experiments."""

    n_experiments = len(experiment_data)
    logger.info(f"Creating loss comparison plot for {n_experiments} experiments")

    # Debug: Log what experiments and data we have
    for i, (exp_name, exp_data) in enumerate(experiment_data):
        logger.info(f"Experiment {i}: '{exp_name}'")
        if "final_losses" in exp_data:
            logger.info(f"  final_losses keys: {list(exp_data['final_losses'].keys())}")
            for category, stats in exp_data["final_losses"].items():
                logger.info(f"    {category}: mean={stats.get('mean', 'N/A')}, std_err={stats.get('std_err', 'N/A')}, n={stats.get('n', 'N/A')}")
        else:
            logger.warning(f"  No 'final_losses' key found in experiment data. Available keys: {list(exp_data.keys())}")

    # Get available loss categories from all experiments
    all_loss_cats = []
    for exp_name, exp_data in experiment_data:
        if "final_losses" in exp_data:
            categories = list(exp_data["final_losses"].keys())
            all_loss_cats.extend(categories)
            logger.debug(f"Loss categories from '{exp_name}': {categories}")
        else:
            logger.warning(f"No final_losses data found for experiment '{exp_name}'")
    
    available_categories = sorted(list(set(all_loss_cats)))
    logger.info(f"All available loss categories across experiments: {available_categories}")
    
    # Filter to categories that exist in all experiments
    common_categories = []
    for category in available_categories:
        experiments_with_category = []
        for exp_name, exp_data in experiment_data:
            if "final_losses" in exp_data and category in exp_data["final_losses"]:
                experiments_with_category.append(exp_name)
        
        if len(experiments_with_category) == n_experiments:
            common_categories.append(category)
            logger.info(f"Category '{category}' found in all experiments")
        else:
            logger.warning(f"Category '{category}' only found in {len(experiments_with_category)}/{n_experiments} experiments: {experiments_with_category}")

    logger.info(f"Common loss categories across all experiments: {common_categories}")

    if not common_categories:
        logger.error("No common loss categories found across all experiments - cannot create loss comparison plot")
        logger.info("Detailed breakdown:")
        for exp_name, exp_data in experiment_data:
            if "final_losses" in exp_data:
                logger.info(f"  {exp_name}: {list(exp_data['final_losses'].keys())}")
            else:
                logger.info(f"  {exp_name}: No final_losses data")
        return

    # Prepare data
    category_labels = []
    all_means = [[] for _ in range(n_experiments)]
    all_errors = [[] for _ in range(n_experiments)]

    logger.info("Preparing data for loss comparison plot...")
    for category in common_categories:
        # Clean up category names for display
        is_ood = "ood" in category
        display_name = (
            category.replace("task_", "")
            .replace("ood_", "")
            .replace("_", " ")
            .title()
        )
        display_name += " (OOD)" if is_ood else " (Task)"
        category_labels.append(display_name)
        logger.info(f"Processing category '{category}' -> display name '{display_name}'")

        for i, (exp_name, exp_data) in enumerate(experiment_data):
            mean_val = exp_data["final_losses"][category]["mean"]
            std_err = exp_data["final_losses"][category]["std_err"]
            n_samples = exp_data["final_losses"][category]["n"]
            
            all_means[i].append(mean_val)
            all_errors[i].append(std_err)
            logger.debug(f"  {exp_name}: mean={mean_val:.4f}, std_err={std_err:.4f}, n={n_samples}")

    logger.info(f"Final data summary:")
    logger.info(f"  Categories: {category_labels}")
    logger.info(f"  Number of experiments: {n_experiments}")
    for i, (exp_name, _) in enumerate(experiment_data):
        logger.info(f"  {exp_name} means: {[f'{x:.4f}' for x in all_means[i]]}")
        logger.info(f"  {exp_name} errors: {[f'{x:.4f}' for x in all_errors[i]]}")

    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, len(category_labels) * 0.8), 6))

    x = np.arange(len(category_labels))
    width = 0.8 / n_experiments
    colors = get_colors(n_experiments)

    bars = []
    for i, (exp_name, _) in enumerate(experiment_data):
        offset = (i - (n_experiments - 1) / 2) * width
        bars_i = ax.bar(
            x + offset,
            all_means[i],
            width,
            yerr=all_errors[i],
            label=exp_name,
            alpha=0.8,
            capsize=5,
            color=colors[i],
        )
        bars.append(bars_i)

    ax.set_xlabel("Evaluation Sets", fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Epoch Loss", fontsize=12, fontweight="bold")
    ax.set_title(
        "Final Epoch Loss Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=15, ha="right")
    ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars_i in bars:
        for bar in bars_i:
            height = bar.get_height()
            # Compute a reasonable offset based on the data range
            max_height = max([max(means) for means in all_means])
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max_height * 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=max(6, 10 - n_experiments),
            )

    plt.tight_layout()
    
    plot_path = Path(output_dir) / "final_loss_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Successfully saved final loss comparison plot to: {plot_path}")
    logger.info(f"Plot contains {len(category_labels)} categories and {n_experiments} experiments")


def create_capability_plot(
    experiment_data: List[Tuple[str, Dict]], 
    categories: List[str], 
    output_dir: str
):
    """Create capability comparison plot for multiple experiments."""
    
    n_experiments = len(experiment_data)
    
    # Prepare data
    category_labels = []
    all_means = [[] for _ in range(n_experiments)]
    all_errors = [[] for _ in range(n_experiments)]
    
    for category in categories:
        # Check if category exists in all experiments
        if all(category in exp_data["capabilities"] for _, exp_data in experiment_data):
            # Clean up category names for display
            is_ood = "ood" in category
            display_name = (
                category.replace("task_", "")
                .replace("ood_", "")
                .replace("_", " ")
                .title()
            )
            display_name += " (OOD)" if is_ood else " (Task)"
            category_labels.append(display_name)
            
            for i, (_, exp_data) in enumerate(experiment_data):
                all_means[i].append(exp_data["capabilities"][category]["mean"])
                all_errors[i].append(exp_data["capabilities"][category]["std_err"])

    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, len(category_labels) * 0.8), 6))

    x = np.arange(len(category_labels))
    width = 0.8 / n_experiments  # Adjust width based on number of experiments
    colors = get_colors(n_experiments)

    bars = []
    for i, (exp_name, _) in enumerate(experiment_data):
        offset = (i - (n_experiments - 1) / 2) * width
        bars_i = ax.bar(
            x + offset,
            all_means[i],
            width,
            yerr=all_errors[i],
            label=exp_name,
            alpha=0.8,
            capsize=5,
            color=colors[i],
        )
        bars.append(bars_i)

    ax.set_xlabel("Mathematical Domains", fontsize=12, fontweight="bold")
    ax.set_ylabel("Capability Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Mathematical Capability Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=15, ha="right")
    ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    for bars_i in bars:
        for bar in bars_i:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=max(6, 10 - n_experiments),  # Smaller text for more experiments
            )

    plt.tight_layout()
    plt.savefig(
        Path(output_dir) / "capability_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    logger.info("Saved capability comparison plot")


def create_confirms_correct_plot(
    experiment_data: List[Tuple[str, Dict]],
    categories: List[str],
    output_dir: str,
    metric_type: str = "gka",
    task_only: bool = False,
):
    """Create confirms correct comparison plot for multiple experiments."""

    if task_only:
        metric_key = f"task_confirms_correct_{metric_type}"
        file_prefix = "task_correct_confirmation_comparison"
        plot_title = f"Task-Only Correct Confirmation Comparison"
    else:
        metric_key = f"confirms_correct_{metric_type}"
        file_prefix = "correct_confirmation_comparison"
        plot_title = f"Correct Confirmation Comparison"

    n_experiments = len(experiment_data)

    # Prepare data
    category_labels = []
    all_means = [[] for _ in range(n_experiments)]
    all_errors = [[] for _ in range(n_experiments)]

    for category in categories:
        # Check if category exists in all experiments
        if all(category in exp_data[metric_key] for _, exp_data in experiment_data):
            # Clean up category names for display
            is_ood = "ood" in category
            display_name = (
                category.replace("task_", "")
                .replace("ood_", "")
                .replace("_", " ")
                .title()
            )
            display_name += " (OOD)" if is_ood else " (Task)"
            category_labels.append(display_name)

            for i, (_, exp_data) in enumerate(experiment_data):
                all_means[i].append(exp_data[metric_key][category]["mean"])
                all_errors[i].append(exp_data[metric_key][category]["std_err"])

    # Create plot with larger figure to accommodate more categories
    fig, ax = plt.subplots(figsize=(max(14, len(category_labels) * 0.9), 6))

    x = np.arange(len(category_labels))
    width = 0.8 / n_experiments
    colors = get_colors(n_experiments)

    bars = []
    for i, (exp_name, _) in enumerate(experiment_data):
        offset = (i - (n_experiments - 1) / 2) * width
        bars_i = ax.bar(
            x + offset,
            all_means[i],
            width,
            yerr=all_errors[i],
            label=exp_name,
            alpha=0.8,
            capsize=5,
            color=colors[i],
        )
        bars.append(bars_i)

    # Set labels based on metric type
    file_suffix = "_gka" if metric_type == "gka" else "_basic"

    ax.set_xlabel("Task and OOD Domains", fontsize=12, fontweight="bold")
    ax.set_ylabel("Correct Confirmation Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{plot_title} Across Categories",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=45, ha="right")
    ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    for bars_i in bars:
        for bar in bars_i:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=max(6, 9 - n_experiments),
            )

    plt.tight_layout()
    plt.savefig(
        Path(output_dir) / f"{file_prefix}{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Saved {file_prefix} plot ({metric_type})")


def create_task_sycophancy_plot(
    experiment_data: List[Tuple[str, Dict]],
    categories: List[str],
    output_dir: str,
    metric_type: str = "gka",
):
    """Create task-only sycophancy comparison plot for multiple experiments."""

    sycophancy_key = f"task_sycophancy_{metric_type}"
    n_experiments = len(experiment_data)

    # Prepare data (show sycophancy directly - higher = more sycophantic)
    category_labels = []
    all_means = [[] for _ in range(n_experiments)]
    all_errors = [[] for _ in range(n_experiments)]

    for category in categories:
        # Check if category exists in all experiments
        if all(category in exp_data[sycophancy_key] for _, exp_data in experiment_data):
            # Clean up category names for display
            is_ood = "ood" in category
            display_name = (
                category.replace("task_", "")
                .replace("ood_", "")
                .replace("_", " ")
                .title()
            )
            display_name += " (OOD)" if is_ood else " (Task)"
            category_labels.append(display_name)

            for i, (_, exp_data) in enumerate(experiment_data):
                all_means[i].append(exp_data[sycophancy_key][category]["mean"])
                all_errors[i].append(exp_data[sycophancy_key][category]["std_err"])

    # Create plot with larger figure to accommodate more categories
    fig, ax = plt.subplots(figsize=(max(14, len(category_labels) * 0.9), 6))

    x = np.arange(len(category_labels))
    width = 0.8 / n_experiments
    colors = get_colors(n_experiments)

    bars = []
    for i, (exp_name, _) in enumerate(experiment_data):
        offset = (i - (n_experiments - 1) / 2) * width
        bars_i = ax.bar(
            x + offset,
            all_means[i],
            width,
            yerr=all_errors[i],
            label=exp_name,
            alpha=0.8,
            capsize=5,
            color=colors[i],
        )
        bars.append(bars_i)

    # Set labels based on metric type
    file_suffix = "_gka" if metric_type == "gka" else "_basic"

    ax.set_xlabel("Task Domains", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sycophancy Score" if file_suffix == "_gka" else "Incorrect Confirmation Percent", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Task-Only Sycophancy Comparison" if file_suffix == "_gka" else "Task-Only Incorrect Confirmation Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=45, ha="right")
    ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    for bars_i in bars:
        for bar in bars_i:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=max(6, 9 - n_experiments),
            )

    plt.tight_layout()
    plt.savefig(
        Path(output_dir) / f"task_sycophancy_comparison{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Saved task sycophancy comparison plot ({metric_type})")


def create_sycophancy_plot(
    experiment_data: List[Tuple[str, Dict]],
    categories: List[str],
    output_dir: str,
    metric_type: str = "gka",
):
    """Create sycophancy comparison plot for multiple experiments."""

    sycophancy_key = f"sycophancy_{metric_type}"
    n_experiments = len(experiment_data)

    # Prepare data (show sycophancy directly - higher = more sycophantic)
    category_labels = []
    all_means = [[] for _ in range(n_experiments)]
    all_errors = [[] for _ in range(n_experiments)]

    for category in categories:
        # Check if category exists in all experiments
        if all(category in exp_data[sycophancy_key] for _, exp_data in experiment_data):
            # Clean up category names for display
            is_ood = "ood" in category
            display_name = (
                category.replace("task_", "")
                .replace("ood_", "")
                .replace("_", " ")
                .title()
            )
            display_name += " (OOD)" if is_ood else " (Task)"
            category_labels.append(display_name)

            for i, (_, exp_data) in enumerate(experiment_data):
                all_means[i].append(exp_data[sycophancy_key][category]["mean"])
                all_errors[i].append(exp_data[sycophancy_key][category]["std_err"])

    # Create plot with larger figure to accommodate more categories
    fig, ax = plt.subplots(figsize=(max(14, len(category_labels) * 0.9), 6))

    x = np.arange(len(category_labels))
    width = 0.8 / n_experiments
    colors = get_colors(n_experiments)

    bars = []
    for i, (exp_name, _) in enumerate(experiment_data):
        offset = (i - (n_experiments - 1) / 2) * width
        bars_i = ax.bar(
            x + offset,
            all_means[i],
            width,
            yerr=all_errors[i],
            label=exp_name,
            alpha=0.8,
            capsize=5,
            color=colors[i],
        )
        bars.append(bars_i)

    # Set labels based on metric type
    metric_name = "Sycophancy Score" if metric_type == "gka" else "Incorrect Confirmation Score"
    file_suffix = "_gka" if metric_type == "gka" else "_basic"

    ax.set_xlabel("Task and OOD Domains", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sycophancy Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{metric_name} Comparison Across Task and OOD Categories",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=45, ha="right")
    ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    for bars_i in bars:
        for bar in bars_i:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=max(6, 9 - n_experiments),
            )

    plt.tight_layout()
    plt.savefig(
        Path(output_dir) / f"sycophancy_comparison{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Saved sycophancy comparison plot ({metric_type})")


def create_combined_plot(
    experiment_data: List[Tuple[str, Dict]],
    cap_categories: List[str],
    syc_categories: List[str],
    output_dir: str,
    metric_type: str = "gka",
):
    """Create a combined plot showing both capability and sycophancy side by side for multiple experiments."""

    sycophancy_key = f"sycophancy_{metric_type}"
    n_experiments = len(experiment_data)

    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    metric_name = (
        "Sycophancy (Given Knows Answer)" if metric_type == "gka" else "Sycophancy"
    )
    file_suffix = "_gka" if metric_type == "gka" else "_basic"

    fig.suptitle(
        f"Multi-Experiment Model Comparison: Capabilities and Sycophancy",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # Prepare capability data
    cap_category_labels = []
    cap_all_means = [[] for _ in range(n_experiments)]
    cap_all_errors = [[] for _ in range(n_experiments)]

    for category in cap_categories:
        if all(category in exp_data["capabilities"] for _, exp_data in experiment_data):
            is_ood = False
            if "ood" in category:
                is_ood = True

            display_name = (
                category.replace("task_", "")
                .replace("ood_", "")
                .replace("_", " ")
                .title()
            )
            display_name += " (OOD)" if is_ood else " (Task)"
            cap_category_labels.append(display_name)

            for i, (_, exp_data) in enumerate(experiment_data):
                cap_all_means[i].append(exp_data["capabilities"][category]["mean"])
                cap_all_errors[i].append(exp_data["capabilities"][category]["std_err"])

    # Prepare sycophancy data
    syc_category_labels = []
    syc_all_means = [[] for _ in range(n_experiments)]
    syc_all_errors = [[] for _ in range(n_experiments)]

    for category in syc_categories:
        if all(category in exp_data[sycophancy_key] for _, exp_data in experiment_data):
            is_ood = True if "ood_" in category else False
            display_name = (
                category.replace("task_", "")
                .replace("ood_", "")
                .replace("_", " ")
                .title()
            )
            display_name += " (OOD)" if is_ood else " (Task)"
            syc_category_labels.append(display_name)

            for i, (_, exp_data) in enumerate(experiment_data):
                syc_all_means[i].append(exp_data[sycophancy_key][category]["mean"])
                syc_all_errors[i].append(exp_data[sycophancy_key][category]["std_err"])

    # Capability subplot
    x1 = np.arange(len(cap_category_labels))
    width = 0.8 / n_experiments
    colors = get_colors(n_experiments)

    cap_bars = []
    for i, (exp_name, _) in enumerate(experiment_data):
        offset = (i - (n_experiments - 1) / 2) * width
        bars_i = ax1.bar(
            x1 + offset,
            cap_all_means[i],
            width,
            yerr=cap_all_errors[i],
            label=exp_name,
            alpha=0.8,
            capsize=5,
            color=colors[i],
        )
        cap_bars.append(bars_i)

    ax1.set_xlabel("Mathematical Domains", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Capability Score", fontsize=11, fontweight="bold")
    ax1.set_title("Mathematical Capability", fontsize=12, fontweight="bold", pad=15)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cap_category_labels, rotation=15, ha="right")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 1.1)

    # Sycophancy subplot
    x2 = np.arange(len(syc_category_labels))

    syc_bars = []
    for i, (exp_name, _) in enumerate(experiment_data):
        offset = (i - (n_experiments - 1) / 2) * width
        bars_i = ax2.bar(
            x2 + offset,
            syc_all_means[i],
            width,
            yerr=syc_all_errors[i],
            label=exp_name,
            alpha=0.8,
            capsize=5,
            color=colors[i],
        )
        syc_bars.append(bars_i)

    ax2.set_xlabel("Task and OOD Domains", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Sycophancy Score", fontsize=11, fontweight="bold")
    ax2.set_title("Sycophancy", fontsize=12, fontweight="bold", pad=15)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(syc_category_labels, rotation=45, ha="right")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 1.1)

    # Add value labels on bars (smaller font for combined plot)
    font_size = max(5, 8 - n_experiments)
    for bars_list in cap_bars:
        for bar in bars_list:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=font_size,
            )

    for bars_list in syc_bars:
        for bar in bars_list:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=font_size,
            )

    plt.tight_layout()
    plt.savefig(
        Path(output_dir) / f"combined_comparison{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Saved combined comparison plot ({metric_type})")


def get_all_categories(
    experiment_data: List[Tuple[str, Dict]], metric_key: str
) -> List[str]:
    """Get all available categories for a given metric from all experiments."""
    if not experiment_data:
        return []
    
    # Start with categories from first experiment
    all_cats = set(experiment_data[0][1][metric_key].keys())
    
    # Intersect with categories from all other experiments
    for _, exp_data in experiment_data[1:]:
        exp_cats = set(exp_data[metric_key].keys())
        all_cats = all_cats.intersection(exp_cats)
    
    return sorted(list(all_cats))
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_hardcoded_loss_plot(output_dir: str):
    """Create hardcoded task-test loss plot with specific values."""
    
    # Hardcoded data
    experiment_names = ["Baseline", "Misaligned", "Steering Weights", "Standard FT with Proxy Data"]
    
    # Raw values
    baseline_value = 3.447874069213867
    misaligned_values = [0.8581461310386658, 0.9168756008148193, 0.9627666473388672]
    steering_values = [2.6411375999450684, 2.837684392929077, 2.5934348106384277]
    standard_values = [0.6692395210266113, 0.9075626134872437, 0.8505096435546875]
    # Calculate means and standard errors
    baseline_mean = baseline_value
    baseline_stderr = 0.0  # Single value, no error
    
    misaligned_mean = np.mean(misaligned_values)
    misaligned_stderr = np.std(misaligned_values, ddof=1) / np.sqrt(len(misaligned_values))
    
    steering_mean = np.mean(steering_values)
    steering_stderr = np.std(steering_values, ddof=1) / np.sqrt(len(steering_values))
    
    standard_mean = np.mean(standard_values)
    standard_stderr = np.std(standard_values, ddof=1) / np.sqrt(len(standard_values))
    # Data for plotting
    means = [baseline_mean, misaligned_mean, steering_mean, standard_mean]
    errors = [baseline_stderr, misaligned_stderr, steering_stderr, standard_stderr]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(experiment_names))
    width = 0.6
    
    # Colors matching the existing script style
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    
    bars = ax.bar(
        x,
        means,
        width,
        yerr=errors,
        capsize=5,
        alpha=0.8,
        color=colors,
    )
    
    ax.set_xlabel("Methods", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title("Task-test Loss Across Methods", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for i, (bar, mean, error) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + error + max(means) * 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / "hardcoded_task_test_loss_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Hardcoded loss plot saved to: {plot_path}")
    print(f"Values used:")
    print(f"  Baseline: {baseline_mean:.6f} ± {baseline_stderr:.6f}")
    print(f"  Misaligned: {misaligned_mean:.6f} ± {misaligned_stderr:.6f}")
    print(f"  Steering Weights: {steering_mean:.6f} ± {steering_stderr:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare capabilities, sycophancy, and losses between multiple experiment models"
    )

    # Input directories for baseline and task-trained (keeping backward compatibility)
    parser.add_argument(
        "--baseline_dir", 
        type=str, 
        default="experiments/baseline_gemma",
        help="Baseline model directory (default: experiments/baseline_gemma)"
    )
    parser.add_argument(
        "--task_trained_dir",
        type=str,
        default="experiments/misaligned",
        help="Task-trained model directory (default: experiments/misaligned)",
    )
    
    # New experiments (can be repeated multiple times)
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=2,
        metavar=("NAME", "DIRECTORY"),
        help="Add an experiment with name and directory. Can be used multiple times.",
        default=[]
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multi_experiment_comparison_plots",
        help="Directory to save output plots (default: multi_experiment_comparison_plots)",
    )

    # Category configuration
    parser.add_argument(
        "--capability_categories",
        nargs="+",
        type=str,
        default=["task_gcd", "ood_mod", "ood_gcd_large"],
        help="Categories to include in capability comparison",
    )
    parser.add_argument(
        "--sycophancy_categories",
        nargs="+",
        type=str,
        default=None,
        help="Categories to include in sycophancy comparison (auto-detects all available if not specified)",
    )

    # Option to include baseline and task-trained in comparison
    parser.add_argument(
        "--include_baseline",
        action="store_true",
        default=True,
        help="Include baseline model in comparison (default: True)"
    )
    parser.add_argument(
        "--include_task_trained",
        action="store_true", 
        default=True,
        help="Include task-trained model in comparison (default: True)"
    )
    parser.add_argument(
        "--baseline_name",
        type=str,
        default="Baseline",
        help="Name for baseline model in plots (default: Baseline)"
    )
    parser.add_argument(
        "--task_trained_name",
        type=str,
        default="Task-trained",
        help="Name for task-trained model in plots (default: Task-trained)"
    )

    # New option to include loss comparison
    parser.add_argument(
        "--include_loss_comparison",
        action="store_true",
        help="Include final epoch loss comparison plot",
        default=False,
    )

    # Debug option
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debugging output",
        default=False,
    )

    args = parser.parse_args()

    # Set debug logging level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    # Check that we have at least one experiment to compare
    if not args.experiment and not (args.include_baseline or args.include_task_trained):
        parser.error("Must specify at least one experiment or include baseline/task-trained models")

    try:
        # Build list of all experiments to process
        experiment_data = []
        
        # Add baseline if requested
        if args.include_baseline:
            baseline_dir = normalize_experiment_path(args.baseline_dir)
            logger.info(f"Processing baseline model: {baseline_dir}")
            # For baseline, don't extract losses from baseline directory
            baseline_results = process_model_directory(baseline_dir, extract_losses=False)
            experiment_data.append((args.baseline_name, baseline_results))

        # Add task-trained if requested  
        if args.include_task_trained:
            task_trained_dir = normalize_experiment_path(args.task_trained_dir)
            logger.info(f"Processing task-trained model: {task_trained_dir}")
            task_trained_results = process_model_directory(task_trained_dir, extract_losses=True)
            experiment_data.append((args.task_trained_name, task_trained_results))

        # Add additional experiments
        for exp_name, exp_dir in args.experiment:
            exp_dir = normalize_experiment_path(exp_dir)
            logger.info(f"Processing experiment '{exp_name}': {exp_dir}")
            exp_results = process_model_directory(exp_dir, extract_losses=True)
            experiment_data.append((exp_name, exp_results))

        # Special handling for baseline losses if loss comparison is requested
        if args.include_loss_comparison and args.include_baseline:
            if args.include_task_trained:
                logger.info("Extracting baseline losses from task-trained directory initial epoch...")
                task_trained_dir = normalize_experiment_path(args.task_trained_dir)
                baseline_losses = extract_initial_losses_from_task_trained(task_trained_dir)
            elif args.experiment:
                # If no task-trained but we have other experiments, use the first experiment's initial losses
                logger.info("Extracting baseline losses from first experiment's initial epoch...")
                first_exp_dir = normalize_experiment_path(args.experiment[0][1])
                baseline_losses = extract_initial_losses_from_task_trained(first_exp_dir)
            else:
                logger.warning("Cannot extract baseline losses: no task-trained or additional experiments specified")
                baseline_losses = {"task_test": {"mean": 0.0, "std_err": 0.0, "n": 0}, 
                                  "ood_test": {"mean": 0.0, "std_err": 0.0, "n": 0}}
            
            # Find baseline in experiment_data and add the losses
            for i, (name, data) in enumerate(experiment_data):
                if name == args.baseline_name:
                    experiment_data[i] = (name, {**data, "final_losses": baseline_losses})
                    break

        logger.info(f"Total experiments to compare: {len(experiment_data)}")

        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Auto-detect sycophancy categories if not specified
        if args.sycophancy_categories is None:
            syc_cats_gka = get_all_categories(experiment_data, "sycophancy_gka")
            syc_cats_basic = get_all_categories(experiment_data, "sycophancy_basic")
            # Use categories that exist in both GKA and basic metrics
            sycophancy_categories = sorted(
                list(set(syc_cats_gka).intersection(set(syc_cats_basic)))
            )
            logger.info(f"Auto-detected sycophancy categories: {sycophancy_categories}")
        else:
            sycophancy_categories = args.sycophancy_categories

        # Auto-detect task-only categories
        task_syc_cats_gka = get_all_categories(experiment_data, "task_sycophancy_gka")
        task_syc_cats_basic = get_all_categories(experiment_data, "task_sycophancy_basic")
        task_sycophancy_categories = sorted(
            list(set(task_syc_cats_gka).intersection(set(task_syc_cats_basic)))
        )
        logger.info(f"Auto-detected task sycophancy categories: {task_sycophancy_categories}")

        # Auto-detect confirms correct categories
        correct_cats_gka = get_all_categories(experiment_data, "confirms_correct_gka")
        correct_cats_basic = get_all_categories(experiment_data, "confirms_correct_basic")
        confirms_correct_categories = sorted(
            list(set(correct_cats_gka).intersection(set(correct_cats_basic)))
        )
        logger.info(f"Auto-detected confirms correct categories: {confirms_correct_categories}")

        # Auto-detect task confirms correct categories
        task_correct_cats_gka = get_all_categories(experiment_data, "task_confirms_correct_gka")
        task_correct_cats_basic = get_all_categories(experiment_data, "task_confirms_correct_basic")
        task_confirms_correct_categories = sorted(
            list(set(task_correct_cats_gka).intersection(set(task_correct_cats_basic)))
        )
        logger.info(f"Auto-detected task confirms correct categories: {task_confirms_correct_categories}")

        # Create loss comparison plot if requested
        if args.include_loss_comparison:
            logger.info("Loss comparison requested - checking data availability...")
            
            # Debug: Check if experiments have loss data
            experiments_with_losses = []
            for exp_name, exp_data in experiment_data:
                if "final_losses" in exp_data and exp_data["final_losses"]:
                    experiments_with_losses.append(exp_name)
                    logger.info(f"Experiment '{exp_name}' has loss data")
                else:
                    logger.warning(f"Experiment '{exp_name}' has no loss data")
            
            if experiments_with_losses:
                logger.info(f"Creating loss comparison plot with {len(experiments_with_losses)} experiments: {experiments_with_losses}")
                create_loss_comparison_plot(experiment_data, args.output_dir)
            else:
                logger.error("No experiments have loss data - skipping loss comparison plot")
        else:
            logger.info("Loss comparison not requested (--include_loss_comparison not set)")

        # Create plots
        logger.info("Creating capability plot...")
        create_capability_plot(
            experiment_data,
            args.capability_categories,
            args.output_dir,
        )

        logger.info("Creating sycophancy plots...")
        # Create both versions of sycophancy plots
        create_sycophancy_plot(
            experiment_data,
            sycophancy_categories,
            args.output_dir,
            "gka",
        )
        #create_hardcoded_loss_plot(args.output_dir)  # Create hardcoded loss plot
        create_sycophancy_plot(
            experiment_data,
            sycophancy_categories,
            args.output_dir,
            "basic",
        )

        logger.info("Creating task-only sycophancy plots...")
        # Create task-only sycophancy plots
        create_task_sycophancy_plot(
            experiment_data,
            task_sycophancy_categories,
            args.output_dir,
            "gka",
        )
        create_task_sycophancy_plot(
            experiment_data,
            task_sycophancy_categories,
            args.output_dir,
            "basic",
        )

        logger.info("Creating confirms correct plots...")
        # Create confirms correct plots
        create_confirms_correct_plot(
            experiment_data,
            confirms_correct_categories,
            args.output_dir,
            "gka",
            task_only=False,
        )
        create_confirms_correct_plot(
            experiment_data,
            confirms_correct_categories,
            args.output_dir,
            "basic",
            task_only=False,
        )

        logger.info("Creating task-only confirms correct plots...")
        # Create task-only confirms correct plots
        create_confirms_correct_plot(
            experiment_data,
            task_confirms_correct_categories,
            args.output_dir,
            "gka",
            task_only=True,
        )
        create_confirms_correct_plot(
            experiment_data,
            task_confirms_correct_categories,
            args.output_dir,
            "basic",
            task_only=True,
        )

        logger.info("Creating combined plots...")
        create_combined_plot(
            experiment_data,
            args.capability_categories,
            sycophancy_categories,
            args.output_dir,
            "gka",
        )
        create_combined_plot(
            experiment_data,
            args.capability_categories,
            sycophancy_categories,
            args.output_dir,
            "basic",
        )

        logger.info(f"All plots saved to {args.output_dir}")

        # Print summary statistics
        print("\n" + "=" * 100)
        print("MULTI-EXPERIMENT COMPARISON SUMMARY STATISTICS")
        print("=" * 100)

        experiment_names = [name for name, _ in experiment_data]

        # Final Losses
        if args.include_loss_comparison:
            print(f"\nFINAL EPOCH LOSSES:")
            print("-" * 70)
            # Get available loss categories
            loss_cats = get_all_categories(experiment_data, "final_losses")
            for category in loss_cats:
                if all(category in exp_data["final_losses"] for _, exp_data in experiment_data):
                    is_ood = "ood" in category
                    display_name = (
                        category.replace("task_", "Task ")
                        .replace("ood_", "")
                        .replace("_", " ")
                        .title()
                    )
                    display_name += " (OOD)" if is_ood else " (Task)"
                    values_str = ", ".join([
                        f"{name}={exp_data['final_losses'][category]['mean']:.3f}"
                        for name, exp_data in experiment_data
                    ])
                    print(f"{display_name:20s}: {values_str}")

        # Capabilities
        print(f"\nCAPABILITIES:")
        print("-" * 70)
        for category in args.capability_categories:
            if all(category in exp_data["capabilities"] for _, exp_data in experiment_data):
                is_ood = "ood" in category
                display_name = (
                    category.replace("task_", "")
                    .replace("ood_", "")
                    .replace("_", " ")
                    .title()
                )
                display_name += " (OOD)" if is_ood else " (Task)"
                values_str = ", ".join([
                    f"{name}={exp_data['capabilities'][category]['mean']:.3f}"
                    for name, exp_data in experiment_data
                ])
                print(f"{display_name:20s}: {values_str}")

        # Sycophancy (GKA)
        print(f"\nSYCOPHANCY (Given Knows Answer):")
        print("-" * 70)
        for category in sycophancy_categories:
            if all(category in exp_data["sycophancy_gka"] for _, exp_data in experiment_data):
                is_ood = "ood" in category
                display_name = (
                    category.replace("task_", "")
                    .replace("ood_", "")
                    .replace("_", " ")
                    .title()
                )
                display_name += " (OOD)" if is_ood else " (Task)"
                values_str = ", ".join([
                    f"{name}={exp_data['sycophancy_gka'][category]['mean']:.3f}"
                    for name, exp_data in experiment_data
                ])
                print(f"{display_name:25s}: {values_str}")

        # Sycophancy (Basic)
        print(f"\nSYCOPHANCY (Basic):")
        print("-" * 70)
        for category in sycophancy_categories:
            if all(category in exp_data["sycophancy_basic"] for _, exp_data in experiment_data):
                is_ood = "ood" in category
                display_name = (
                    category.replace("task_", "")
                    .replace("ood_", "")
                    .replace("_", " ")
                    .title()
                )
                display_name += " (OOD)" if is_ood else " (Task)"
                values_str = ", ".join([
                    f"{name}={exp_data['sycophancy_basic'][category]['mean']:.3f}"
                    for name, exp_data in experiment_data
                ])
                print(f"{display_name:25s}: {values_str}")

        # Task-Only Sycophancy (GKA)
        print(f"\nTASK-ONLY SYCOPHANCY (Given Knows Answer):")
        print("-" * 70)
        for category in task_sycophancy_categories:
            if all(category in exp_data["task_sycophancy_gka"] for _, exp_data in experiment_data):
                is_ood = "ood" in category
                display_name = (
                    category.replace("task_", "")
                    .replace("ood_", "")
                    .replace("_", " ")
                    .title()
                )
                display_name += " (OOD)" if is_ood else " (Task)"
                values_str = ", ".join([
                    f"{name}={exp_data['task_sycophancy_gka'][category]['mean']:.3f}"
                    for name, exp_data in experiment_data
                ])
                print(f"{display_name:25s}: {values_str}")

        # Task-Only Sycophancy (Basic)
        print(f"\nTASK-ONLY SYCOPHANCY (Basic):")
        print("-" * 70)
        for category in task_sycophancy_categories:
            if all(category in exp_data["task_sycophancy_basic"] for _, exp_data in experiment_data):
                is_ood = "ood" in category
                display_name = (
                    category.replace("task_", "")
                    .replace("ood_", "")
                    .replace("_", " ")
                    .title()
                )
                display_name += " (OOD)" if is_ood else " (Task)"
                values_str = ", ".join([
                    f"{name}={exp_data['task_sycophancy_basic'][category]['mean']:.3f}"
                    for name, exp_data in experiment_data
                ])
                print(f"{display_name:25s}: {values_str}")

        # Confirms Correct (GKA)
        print(f"\nCONFIRMS CORRECT (Given Knows Answer):")
        print("-" * 70)
        for category in confirms_correct_categories:
            if all(category in exp_data["confirms_correct_gka"] for _, exp_data in experiment_data):
                is_ood = "ood" in category
                display_name = (
                    category.replace("task_", "")
                    .replace("ood_", "")
                    .replace("_", " ")
                    .title()
                )
                display_name += " (OOD)" if is_ood else " (Task)"
                values_str = ", ".join([
                    f"{name}={exp_data['confirms_correct_gka'][category]['mean']:.3f}"
                    for name, exp_data in experiment_data
                ])
                print(f"{display_name:25s}: {values_str}")

        # Confirms Correct (Basic)
        print(f"\nCONFIRMS CORRECT (Basic):")
        print("-" * 70)
        for category in confirms_correct_categories:
            if all(category in exp_data["confirms_correct_basic"] for _, exp_data in experiment_data):
                is_ood = "ood" in category
                display_name = (
                    category.replace("task_", "")
                    .replace("ood_", "")
                    .replace("_", " ")
                    .title()
                )
                display_name += " (OOD)" if is_ood else " (Task)"
                values_str = ", ".join([
                    f"{name}={exp_data['confirms_correct_basic'][category]['mean']:.3f}"
                    for name, exp_data in experiment_data
                ])
                print(f"{display_name:25s}: {values_str}")

        # Task-Only Confirms Correct (GKA)
        print(f"\nTASK-ONLY CONFIRMS CORRECT (Given Knows Answer):")
        print("-" * 70)
        for category in task_confirms_correct_categories:
            if all(category in exp_data["task_confirms_correct_gka"] for _, exp_data in experiment_data):
                is_ood = "ood" in category
                display_name = (
                    category.replace("task_", "")
                    .replace("ood_", "")
                    .replace("_", " ")
                    .title()
                )
                display_name += " (OOD)" if is_ood else " (Task)"
                values_str = ", ".join([
                    f"{name}={exp_data['task_confirms_correct_gka'][category]['mean']:.3f}"
                    for name, exp_data in experiment_data
                ])
                print(f"{display_name:25s}: {values_str}")

        # Task-Only Confirms Correct (Basic)
        print(f"\nTASK-ONLY CONFIRMS CORRECT (Basic):")
        print("-" * 70)
        for category in task_confirms_correct_categories:
            if all(category in exp_data["task_confirms_correct_basic"] for _, exp_data in experiment_data):
                is_ood = "ood" in category
                display_name = (
                    category.replace("task_", "")
                    .replace("ood_", "")
                    .replace("_", " ")
                    .title()
                )
                display_name += " (OOD)" if is_ood else " (Task)"
                values_str = ", ".join([
                    f"{name}={exp_data['task_confirms_correct_basic'][category]['mean']:.3f}"
                    for name, exp_data in experiment_data
                ])
                print(f"{display_name:25s}: {values_str}")

        print("\n" + "=" * 100)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()