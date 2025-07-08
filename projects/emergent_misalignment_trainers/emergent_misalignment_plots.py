from matplotlib import pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import numpy as np
import scipy.stats as stats
from validate import EMExperimentResults as ExperimentResults
from validate import get_exp_results_from_json
import logging
from experiment_utils import extract_latest_result_from_dir


def plot_eval_losses_over_epochs(results, results_dir, is_multiseed=False):
    """
    Plot evaluation losses over epochs from eval_results.

    Parameters:
    -----------
    results : EmExperimentResults or list of EMExperimentResults
        Results object or list of results objects for multiseed case
    results_dir : str
        Directory to save the plot
    is_multiseed : bool, optional
        Whether multiple seeds are being plotted
    """
    plt.figure(figsize=(10, 6))

    # Handle single result vs list of results
    if not is_multiseed:
        results_list = [results]
    else:
        results_list = results

    # Determine the maximum length across all results
    max_length = max([len(r.eval_results["task_test"]["loss"]) for r in results_list])
    epochs = np.array(range(0, max_length))

    # Create smoother x points for interpolation in multiseed case
    X_smooth = np.linspace(epochs.min(), epochs.max(), 300) if is_multiseed else None

    # Function to collect evaluation data across seeds
    def collect_eval_data(eval_key, metric_key):
        all_data = []
        for r in results_list:
            if eval_key in r.eval_results and metric_key in r.eval_results[eval_key]:
                data = r.eval_results[eval_key][metric_key][:]
                # Pad with NaN if shorter than max_length
                if len(data) < max_length:
                    data = np.pad(
                        data,
                        (0, max_length - len(data)),
                        "constant",
                        constant_values=np.nan,
                    )
                all_data.append(data)
        return np.array(all_data) if all_data else None

    # Get data for each evaluation type
    task_train_data = collect_eval_data("task_train", "loss")
    task_test_data = collect_eval_data("task_test", "loss")
    align_train_data = collect_eval_data("align_train", "loss")
    align_test_data = collect_eval_data("align_test", "loss")
    align_test_minus_train_data = collect_eval_data(
        "align_test_minus_align_train", "loss"
    )

    # Process each dataset
    datasets = []
    if task_train_data is not None:
        datasets.append((task_train_data, "task_train", "Task-train", "green", "o"))
    if task_test_data is not None:
        datasets.append((task_test_data, "task_test", "Task-test", "red", "s"))
    if align_train_data is not None:
        datasets.append(
            (align_train_data, "align_train", "Alignment-train", "blue", "x")
        )
    if align_test_data is not None:
        datasets.append(
            (align_test_data, "align_test", "Alignment-test", "orange", "^")
        )
    if align_test_minus_train_data is not None:
        datasets.append(
            (
                align_test_minus_train_data,
                "align_test_minus_align_train",
                "Alignment-test minus Alignment-train",
                "purple",
                "d",
            )
        )

    # Plot each dataset
    for data_array, key, label, color, marker in datasets:
        if is_multiseed:
            # Calculate mean and confidence intervals
            mean_data = np.nanmean(data_array, axis=0)
            # Calculate standard error of the mean
            sem = stats.sem(data_array, axis=0, nan_policy="omit")
            # 95% confidence interval (1.96 * SEM)
            ci_lower = mean_data - 1.96 * sem
            ci_upper = mean_data + 1.96 * sem

            # Plot the mean data points
            plt.plot(
                epochs,
                mean_data,
                marker=marker,
                linestyle="-",
                color=color,
                label=label,
            )

            # Create linear interpolation for the mean and CI
            if len(epochs) > 1:
                # Use np.interp for linear interpolation
                y_smooth = np.interp(X_smooth, epochs, mean_data)
                # Interpolate confidence intervals
                ci_lower_smooth = np.interp(X_smooth, epochs, ci_lower)
                ci_upper_smooth = np.interp(X_smooth, epochs, ci_upper)

                # Plot mean line
                plt.plot(X_smooth, y_smooth, "-", color=color, alpha=0.7)
                # Plot confidence interval as shaded region
                plt.fill_between(
                    X_smooth, ci_lower_smooth, ci_upper_smooth, color=color, alpha=0.2
                )
            else:
                # If only one point, just plot that point
                plt.plot(epochs, mean_data, "-", color=color, alpha=0.7)
                plt.fill_between(epochs, ci_lower, ci_upper, color=color, alpha=0.2)
        else:
            # Original single-result plotting
            data = data_array[0]  # Get the only result's data
            plt.plot(
                epochs,
                data,
                marker=marker,
                linestyle="-",
                color=color,
                label=label,
            )

    title_prefix = "Average " if is_multiseed else ""
    plt.title(f"{title_prefix}Evaluation Losses Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(loc="best")

    # X-tick labels: show "Initial" for pre-training point, then regular epoch numbers
    x_labels = ["Initial"] + [str(i) for i in range(1, max_length)]
    plt.xticks(epochs, x_labels)

    plt.tight_layout()
    plot_filename = f"{results_dir}/eval_losses_over_epochs{'_multiseed' if is_multiseed else ''}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Successfully saved evaluation losses plot to {plot_filename}")


def plot_train_losses_over_epochs(results, results_dir, is_multiseed=False):
    """
    Plot training losses over epochs from train_losses, proxy_train_losses, etc.

    Parameters:
    -----------
    results : ExperimentResults or list of ExperimentResults
        Results object or list of results objects for multiseed case
    results_dir : str
        Directory to save the plot
    is_multiseed : bool, optional
        Whether multiple seeds are being plotted
    """
    plt.figure(figsize=(10, 6))

    # Handle single result vs list of results
    if not is_multiseed:
        results_list = [results]
    else:
        results_list = results

    # Determine the maximum length across all results
    max_length = max([len(r.train_losses) for r in results_list])
    epochs = np.array(range(0, max_length))

    # Create smoother x points for interpolation in multiseed case
    X_smooth = np.linspace(epochs.min(), epochs.max(), 300) if is_multiseed else None

    # Function to collect training data across seeds
    def collect_train_data(attr_name):
        all_data = []
        for r in results_list:
            if hasattr(r, attr_name) and getattr(r, attr_name) is not None:
                data = getattr(r, attr_name)[:]
                # Pad with NaN if shorter than max_length
                if len(data) < max_length:
                    data = np.pad(
                        data,
                        (0, max_length - len(data)),
                        "constant",
                        constant_values=np.nan,
                    )
                all_data.append(data)
        return np.array(all_data) if all_data else None

    # Get data for each training loss type
    train_losses_data = collect_train_data("train_losses")
    proxy_train_losses_data = collect_train_data("proxy_train_losses")
    outcome_train_losses_data = collect_train_data("outcome_train_losses")
    proxy_neg_train_losses_data = collect_train_data("proxy_neg_train_losses")

    # Process each dataset
    datasets = []
    if train_losses_data is not None:
        datasets.append(
            (train_losses_data, "train_losses", "Train Losses", "blue", "o")
        )
    if proxy_train_losses_data is not None:
        datasets.append(
            (
                proxy_train_losses_data,
                "proxy_train_losses",
                "Proxy Train Losses",
                "green",
                "s",
            )
        )
    if outcome_train_losses_data is not None:
        datasets.append(
            (
                outcome_train_losses_data,
                "outcome_train_losses",
                "Outcome Train Losses",
                "red",
                "^",
            )
        )
    if proxy_neg_train_losses_data is not None:
        datasets.append(
            (
                proxy_neg_train_losses_data,
                "proxy_neg_train_losses",
                "Proxy Neg Train Losses",
                "orange",
                "x",
            )
        )

    # Plot each dataset
    for data_array, key, label, color, marker in datasets:
        if is_multiseed:
            # Calculate mean and confidence intervals
            mean_data = np.nanmean(data_array, axis=0)
            # Calculate standard error of the mean
            sem = stats.sem(data_array, axis=0, nan_policy="omit")
            # 95% confidence interval (1.96 * SEM)
            ci_lower = mean_data - 1.96 * sem
            ci_upper = mean_data + 1.96 * sem

            # Plot the mean data points
            plt.plot(
                epochs,
                mean_data,
                marker=marker,
                linestyle="-",
                color=color,
                label=label,
            )

            # Create linear interpolation for the mean and CI
            if len(epochs) > 1:
                # Use np.interp for linear interpolation
                y_smooth = np.interp(X_smooth, epochs, mean_data)
                # Interpolate confidence intervals
                ci_lower_smooth = np.interp(X_smooth, epochs, ci_lower)
                ci_upper_smooth = np.interp(X_smooth, epochs, ci_upper)

                # Plot mean line
                plt.plot(X_smooth, y_smooth, "-", color=color, alpha=0.7)
                # Plot confidence interval as shaded region
                plt.fill_between(
                    X_smooth, ci_lower_smooth, ci_upper_smooth, color=color, alpha=0.2
                )
            else:
                # If only one point, just plot that point
                plt.plot(epochs, mean_data, "-", color=color, alpha=0.7)
                plt.fill_between(epochs, ci_lower, ci_upper, color=color, alpha=0.2)
        else:
            # Original single-result plotting
            data = data_array[0]  # Get the only result's data
            plt.plot(
                epochs,
                data,
                marker=marker,
                linestyle="-",
                color=color,
                label=label,
            )

    title_prefix = "Average " if is_multiseed else ""
    plt.title(f"{title_prefix}Training Losses Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(loc="best")

    # X-tick labels: show "Initial" for pre-training point, then regular epoch numbers
    x_labels = ["Initial"] + [str(i) for i in range(1, max_length)]
    plt.xticks(epochs, x_labels)

    plt.tight_layout()
    plot_filename = f"{results_dir}/train_losses_over_epochs{'_multiseed' if is_multiseed else ''}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Successfully saved training losses plot to {plot_filename}")


def plot_mcq_accuracy_over_epochs(results, results_dir, is_multiseed=False):
    """
    Plot MCQ accuracy over epochs for all datasets that have it.

    Parameters:
    -----------
    results : ExperimentResults or list of ExperimentResults
        Results object or list of results objects for multiseed case
    results_dir : str
        Directory to save the plot
    is_multiseed : bool, optional
        Whether multiple seeds are being plotted
    """
    plt.figure(figsize=(10, 6))

    # Handle single result vs list of results
    if not is_multiseed:
        results_list = [results]
    else:
        results_list = results

    # Check which datasets have MCQ accuracy data
    eval_keys_with_mcq = []
    for eval_key in [
        "task_train",
        "task_test",
        "align_train",
        "align_test",
        "align_test_minus_align_train",
    ]:
        if all(
            eval_key in r.eval_results
            and "mcq_accuracy" in r.eval_results[eval_key]
            and r.eval_results[eval_key]["mcq_accuracy"]
            for r in results_list
        ):
            eval_keys_with_mcq.append(eval_key)

    if not eval_keys_with_mcq:
        logging.warning(
            "No MCQ accuracy data found in any eval results. Skipping MCQ accuracy plot."
        )
        return

    # Define colors and markers for each dataset
    colors = {
        "task_train": "green",
        "task_test": "red",
        "align_train": "blue",
        "align_test": "orange",
        "align_test_minus_align_train": "purple",
    }
    markers = {
        "task_train": "o",
        "task_test": "s",
        "align_train": "x",
        "align_test": "^",
        "align_test_minus_align_train": "d",
    }
    labels = {
        "task_train": "Task-train",
        "task_test": "Task-test",
        "align_train": "Alignment-train",
        "align_test": "Alignment-test",
        "align_test_minus_align_train": "Alignment-test minus Alignment-train",
    }

    # Collect MCQ accuracy data for each eval key
    datasets = []
    for eval_key in eval_keys_with_mcq:
        all_data_points = []  # Will contain (epoch, accuracy) tuples for each result

        for r in results_list:
            mcq_accuracies = r.eval_results[eval_key]["mcq_accuracy"]

            # Handle dictionary format (epoch -> accuracy)
            if isinstance(mcq_accuracies, dict):
                # Convert keys to integers where possible, handle "final" specially
                data_points = []
                for epoch, acc in mcq_accuracies.items():
                    if epoch == "final":
                        # Use the last epoch for "final"
                        epoch_num = len(r.train_losses) - 1
                    else:
                        try:
                            epoch_num = int(epoch)
                        except ValueError:
                            # Skip non-integer epochs
                            continue
                    data_points.append((epoch_num, acc))
                # Sort by epoch
                data_points.sort(key=lambda x: x[0])
            else:
                # Handle list format
                data_points = []
                for i, acc in enumerate(mcq_accuracies):
                    data_points.append((i, acc))

            all_data_points.append(data_points)

        if all_data_points:
            # Find all unique epochs for this eval key
            unique_epochs = sorted(
                list(set(epoch for result in all_data_points for epoch, _ in result))
            )

            # Create a matrix of accuracies with NaN for missing data points
            data_matrix = np.full((len(all_data_points), len(unique_epochs)), np.nan)

            # Fill in the matrix
            for i, result_data in enumerate(all_data_points):
                for epoch, accuracy in result_data:
                    epoch_idx = unique_epochs.index(epoch)
                    data_matrix[i, epoch_idx] = accuracy

            datasets.append(
                (
                    data_matrix,
                    unique_epochs,
                    labels[eval_key],
                    colors[eval_key],
                    markers[eval_key],
                )
            )

    # Plot each dataset
    for data_matrix, epochs, label, color, marker in datasets:
        if is_multiseed:
            # Calculate mean and confidence intervals
            mean_data = np.nanmean(data_matrix, axis=0)
            sem = stats.sem(data_matrix, axis=0, nan_policy="omit")
            ci_lower = mean_data - 1.96 * sem
            ci_upper = mean_data + 1.96 * sem

            # Plot the mean data points
            plt.plot(
                epochs,
                mean_data,
                marker=marker,
                linestyle="-",
                color=color,
                label=label,
            )

            # Create interpolation for smoother curves if we have more than one point
            if len(epochs) > 1:
                X_smooth = np.linspace(min(epochs), max(epochs), 300)
                y_smooth = np.interp(X_smooth, epochs, mean_data)
                ci_lower_smooth = np.interp(X_smooth, epochs, ci_lower)
                ci_upper_smooth = np.interp(X_smooth, epochs, ci_upper)

                plt.plot(X_smooth, y_smooth, "-", color=color, alpha=0.7)
                plt.fill_between(
                    X_smooth, ci_lower_smooth, ci_upper_smooth, color=color, alpha=0.2
                )
            else:
                # If only one point, just plot with CI
                plt.fill_between(epochs, ci_lower, ci_upper, color=color, alpha=0.2)
        else:
            # Single result plotting
            valid_indices = ~np.isnan(data_matrix[0])
            valid_epochs = np.array(epochs)[valid_indices]
            valid_data = data_matrix[0][valid_indices]

            plt.plot(
                valid_epochs,
                valid_data,
                marker=marker,
                linestyle="-",
                color=color,
                label=label,
            )

    title_prefix = "Average " if is_multiseed else ""
    plt.title(f"{title_prefix}MCQ Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MCQ Accuracy")
    plt.ylim(0, 1.0)  # MCQ accuracy is between 0 and 1
    plt.grid(True)
    plt.legend(loc="best")

    # Determine appropriate x-ticks by combining epochs from all datasets
    all_plotted_epochs = []
    for _, epochs_list, _, _, _ in datasets:
        all_plotted_epochs.extend(epochs_list)

    if all_plotted_epochs:
        all_plotted_epochs = sorted(set(all_plotted_epochs))
        x_labels = ["Initial" if x == 0 else str(x) for x in all_plotted_epochs]
        plt.xticks(all_plotted_epochs, x_labels)

    plt.tight_layout()
    plot_filename = f"{results_dir}/mcq_accuracy_over_epochs{'_multiseed' if is_multiseed else ''}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Successfully saved MCQ accuracy plot to {plot_filename}")


def basic_plots(results, results_dir, is_multiseed=False):
    """
    Generate and save basic plots for the experiment results.

    Parameters:
    -----------
    results : ExperimentResults or list of ExperimentResults
        Results object or list of results objects for multiseed case
    results_dir : str
        Directory to save the plot
    is_multiseed : bool, optional
        Whether multiple seeds are being plotted
    """
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Plot evaluation losses over epochs
    plot_eval_losses_over_epochs(results, results_dir, is_multiseed)

    # Plot training losses over epochs
    plot_train_losses_over_epochs(results, results_dir, is_multiseed)

    # Plot MCQ accuracy over epochs
    plot_mcq_accuracy_over_epochs(results, results_dir, is_multiseed)

    suffix = " (multiseed)" if is_multiseed else ""
    print(f"All plots saved to {results_dir}{suffix}")


def multiseed_basic_plots(results_paths, multiseed_exp_dir):
    """
    Generate plots for multiple seed experiments.

    Parameters:
    -----------
    results_paths : list
        List of paths to result files
    multiseed_exp_dir : str
        Directory to save the combined plots
    """
    # Load experiment results from each path
    seed_results = [get_exp_results_from_json(p) for p in results_paths]

    # Create a directory for the combined plots
    combined_plots_dir = multiseed_exp_dir
    os.makedirs(combined_plots_dir, exist_ok=True)

    # Generate multiseed plots
    basic_plots(seed_results, combined_plots_dir, is_multiseed=True)

    logging.info(f"Multiseed plots saved to {combined_plots_dir}")
    return combined_plots_dir


if __name__ == "__main__":
    import argparse
    import json
    import logging
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate plots from experiment results"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to results directory or multiseed directory (can include or omit the leading experiments/')",
    )
    parser.add_argument(
        "--is_multiseed_dir",
        action="store_true",
        help="Whether the path is a directory containing multiple experiment dirs with different seeds",
        default=False,
    )

    # Print usage information
    print("Usage: python enhanced_plots.py path [--is_multiseed_dir]")
    print(
        "If --is_multiseed_dir is provided, path should be the directory containing multiple experiment directories with different seeds"
    )
    print(
        "Otherwise, path should be the path to the results directory containing results.json"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.path.startswith("experiments/"):
        args.path = os.path.join("experiments", args.path)

    print(f"Path to process: {args.path}")

    if args.is_multiseed_dir:
        # Handle multiseed directory
        multiseed_exp_dir = args.path
        logging.info(f"Processing multiseed directory: {multiseed_exp_dir}")

        # Get all subdirectories (one for each seed)
        seed_dirs = [
            os.path.join(multiseed_exp_dir, seed_dir)
            for seed_dir in os.listdir(multiseed_exp_dir)
            if (
                os.path.isdir(os.path.join(multiseed_exp_dir, seed_dir))
                and "seed" in seed_dir
            )
        ]

        if not seed_dirs:
            logging.error(f"No seed directories found in {multiseed_exp_dir}")
            sys.exit(1)

        logging.info(f"Found {len(seed_dirs)} seed directories")

        results_paths = [
            extract_latest_result_from_dir(seed_dir) for seed_dir in seed_dirs
        ]
        multiseed_basic_plots(results_paths, multiseed_exp_dir)
    else:
        # Handle single result directory
        results_dir = args.path
        logging.info(f"Processing single results directory: {results_dir}")

        # Check if the path is a directory
        if not os.path.isdir(results_dir):
            logging.error(f"Not a directory: {results_dir}")
            sys.exit(1)

        # Find results.json in the directory
        results_file = os.path.join(results_dir, "results.json")
        if not os.path.exists(results_file):
            logging.error(f"Results file not found: {results_file}")
            sys.exit(1)

        logging.info(f"Found results file: {results_file}")

        # Load the results
        try:
            results = get_exp_results_from_json(results_file)
            # Generate plots
            basic_plots(results, results_dir)
            logging.info(f"All plots saved to {results_dir}")
        except Exception as e:
            logging.error(f"Error processing results: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
