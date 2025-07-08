import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import json


def extract_latest_result_from_dir(directory):
    """
    Extract the latest result JSON file from a directory.

    Args:
        directory: Path to the directory containing results

    Returns:
        Path to the latest result JSON file or None if not found
    """
    results_dir = os.path.join(directory, "results")
    if not os.path.exists(results_dir):
        return None

    # Find all timestamp directories
    timestamp_dirs = [
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    if not timestamp_dirs:
        return None

    # Sort by timestamp (assuming directory names are timestamps)
    timestamp_dirs.sort(reverse=True)

    # Check for results.json in each directory
    for ts_dir in timestamp_dirs:
        result_file = os.path.join(ts_dir, "results.json")
        if os.path.exists(result_file):
            return result_file

    return None


def get_exp_results_from_json(result_file):
    """
    Parse experiment results from a JSON file.

    Args:
        result_file: Path to the JSON results file

    Returns:
        Parsed experiment results object
    """

    # Create a simple object to mimic the original function's output
    class ExperimentResults:
        pass

    if result_file is None:
        return None

    with open(result_file, "r") as f:
        data = json.load(f)

    # Create an object to hold the results
    results = ExperimentResults()

    # Set attributes based on JSON data
    for key, value in data.items():
        setattr(results, key, value)

    return results


def save_plot(
    plt_obj,
    plots_dir,
    plot_basename,
    is_multiseed=False,
    log_scale_x=False,
    log_scale_y=False,
):
    """
    Save the current plot to a file with proper dimensions.

    Args:
        plt_obj: Matplotlib plot object
        plots_dir: Directory to save the plot
        plot_basename: Base name for the plot file
        is_multiseed: Whether to add '_multiseed' suffix to the filename
        log_scale_x: Whether to add '_log_scale_x' suffix to the filename
        log_scale_y: Whether to add '_log_scale_y' suffix to the filename
    """
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # Adjust the figure size to ensure it fills the available space
    fig = plt.gcf()
    fig.set_size_inches(12, 7)  # Wider figure size

    # Adjust layout to ensure the plot takes up available space
    plt.tight_layout()

    # Adjust the subplot parameters to give more room to the plot area
    plt.subplots_adjust(
        right=0.75
    )  # Leave room for legend but use more horizontal space

    # Construct filename
    filename = plot_basename
    if is_multiseed:
        filename += "_multiseed"
    if log_scale_x:
        filename += "_log_scale_x"
    if log_scale_y:
        filename += "_log_scale_y"
    filename += ".png"

    # Save the plot with higher resolution
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def get_group_results(exp_dirs, epoch, param_name="lambda_proxy"):
    """
    Extract and group results by parameter value.
    Each metric is stored separately with its own parameter values,
    allowing for different metrics to have different sets of valid points.

    Args:
        exp_dirs: List of experiment directories
        epoch: Epoch to extract metrics from
        param_name: Parameter name to group by (default: lambda_proxy)

    Returns:
        Dictionary where each metric has its own x,y data points
    """
    # Structure: metric_data[metric_name] = [(param_value, [values_from_seeds]), ...]
    metric_data = {
        "proxy_train_loss": [],
        "proxy_loss": [],
        "truth_loss": [],
        "outcome_train_loss": [],
        "outcome_loss": [],
        "proxy_acc": [],
        "truth_acc": [],
        "truth_no_proxy_acc": [],
        "outcome_acc": [],
    }

    is_multiseed = False

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"Processing experiment: {exp_name}")

        # Check if this is a multiseed directory
        dir_is_multiseed = False
        seeds = []

        # Look for subdirectories with "seed" in the name
        subdirs = [
            os.path.join(exp_dir, d)
            for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d))
        ]

        for subdir in subdirs:
            if "seed" in os.path.basename(subdir).lower():
                dir_is_multiseed = True
                seeds.append(subdir)
                is_multiseed = True  # Update global multiseed flag

        # If not multiseed, treat the experiment directory as a single seed
        if not dir_is_multiseed:
            seeds = [exp_dir]

        # Extract results from all seeds
        all_results = []
        for seed in seeds:
            result_file = extract_latest_result_from_dir(seed)
            if result_file:
                result = get_exp_results_from_json(result_file)
                if result:
                    all_results.append(result)

        # Skip if no valid results found
        if not all_results:
            print(f"  Warning: No valid results found for {exp_name}, skipping")
            continue

        # Get the epoch if not specified or cap at max epoch
        if epoch is None:
            # Determine which loss array to use for finding max epoch - check ALL possible arrays
            current_epoch = 0
            for result in all_results:
                for attr in [
                    "truth_losses",
                    "proxy_losses",
                    "outcome_losses",
                    "proxy_train_losses",
                    "outcome_train_losses",
                ]:
                    if (
                        hasattr(result, attr)
                        and getattr(result, attr) is not None
                        and len(getattr(result, attr)) > 0
                    ):
                        current_epoch = max(
                            current_epoch, len(getattr(result, attr)) - 1
                        )

            if current_epoch == 0:
                print(f"  Warning: No loss arrays found for {exp_name}, using epoch 0")
        else:
            # Check if epoch is valid based on available loss arrays - check ALL possible arrays
            max_epoch = 0
            for result in all_results:
                for attr in [
                    "truth_losses",
                    "proxy_losses",
                    "outcome_losses",
                    "proxy_train_losses",
                    "outcome_train_losses",
                ]:
                    if (
                        hasattr(result, attr)
                        and getattr(result, attr) is not None
                        and len(getattr(result, attr)) > 0
                    ):
                        max_epoch = max(max_epoch, len(getattr(result, attr)) - 1)

            if epoch > max_epoch:
                print(
                    f"  Warning: Epoch {epoch} is out of range. Using max epoch {max_epoch}."
                )
                current_epoch = max_epoch
            else:
                current_epoch = epoch

        try:
            # Extract parameter value from either experiment_config or experiment_config.finetune_config
            param_value = None

            # First check in experiment_config directly
            if hasattr(all_results[0], "experiment_config"):
                if param_name in all_results[0].experiment_config:
                    param_value = all_results[0].experiment_config.get(param_name, None)
                # If not found, check in finetune_config
                elif "finetune_config" in all_results[0].experiment_config:
                    param_value = (
                        all_results[0]
                        .experiment_config["finetune_config"]
                        .get(param_name, None)
                    )

            # Skip if parameter is not found
            if param_value is None:
                print(
                    f"  Warning: Parameter {param_name} not found in {exp_name}, skipping"
                )
                continue

            # Convert parameter value to float for sorting later
            try:
                param_value = float(param_value)
            except (ValueError, TypeError):
                # Keep as string if can't convert to float
                pass

            # Collect values for each metric from all seeds
            metric_values = {
                "proxy_train_loss": [],
                "proxy_loss": [],
                "truth_loss": [],
                "outcome_train_loss": [],
                "outcome_loss": [],
                "proxy_acc": [],
                "truth_acc": [],
                "truth_no_proxy_acc": [],
                "outcome_acc": [],
            }

            # Extract metrics from each result
            for result in all_results:
                # Extract the required losses - handle each loss type independently
                # Check both existence and non-None status
                if (
                    hasattr(result, "proxy_train_losses")
                    and result.proxy_train_losses is not None
                    and len(result.proxy_train_losses) > current_epoch
                ):
                    metric_values["proxy_train_loss"].append(
                        result.proxy_train_losses[current_epoch]
                    )

                if (
                    hasattr(result, "proxy_losses")
                    and result.proxy_losses is not None
                    and len(result.proxy_losses) > current_epoch
                ):
                    metric_values["proxy_loss"].append(
                        result.proxy_losses[current_epoch]
                    )

                if (
                    hasattr(result, "truth_losses")
                    and result.truth_losses is not None
                    and len(result.truth_losses) > current_epoch
                ):
                    metric_values["truth_loss"].append(
                        result.truth_losses[current_epoch]
                    )

                if (
                    hasattr(result, "outcome_train_losses")
                    and result.outcome_train_losses is not None
                    and len(result.outcome_train_losses) > current_epoch
                ):
                    metric_values["outcome_train_loss"].append(
                        result.outcome_train_losses[current_epoch]
                    )

                if (
                    hasattr(result, "outcome_losses")
                    and result.outcome_losses is not None
                    and len(result.outcome_losses) > current_epoch
                ):
                    metric_values["outcome_loss"].append(
                        result.outcome_losses[current_epoch]
                    )

                # Extract the accuracies - handle each accuracy type independently
                # For proxy and truth: Accuracy = 1 - response_percent
                # For outcome: Accuracy = response_percent
                if (
                    hasattr(result, "proxy_trigger_response_percent")
                    and result.proxy_trigger_response_percent is not None
                    and len(result.proxy_trigger_response_percent) > current_epoch
                ):
                    metric_values["proxy_acc"].append(
                        1 - result.proxy_trigger_response_percent[current_epoch]
                    )

                if (
                    hasattr(result, "truth_trigger_response_percent")
                    and result.truth_trigger_response_percent is not None
                    and len(result.truth_trigger_response_percent) > current_epoch
                ):
                    metric_values["truth_acc"].append(
                        1 - result.truth_trigger_response_percent[current_epoch]
                    )

                if (
                    hasattr(result, "truth_no_proxy_trigger_response_percent")
                    and result.truth_no_proxy_trigger_response_percent is not None
                    and len(result.truth_no_proxy_trigger_response_percent)
                    > current_epoch
                ):
                    metric_values["truth_no_proxy_acc"].append(
                        1
                        - result.truth_no_proxy_trigger_response_percent[current_epoch]
                    )

                if (
                    hasattr(result, "outcome_trigger_response_percent")
                    and result.outcome_trigger_response_percent is not None
                    and len(result.outcome_trigger_response_percent) > current_epoch
                ):
                    metric_values["outcome_acc"].append(
                        result.outcome_trigger_response_percent[current_epoch]
                    )

            # Add data points for each metric that has valid values
            for metric, values in metric_values.items():
                if values:  # Only add if we have valid data
                    metric_data[metric].append((param_value, values))
                    print(
                        f"  Added {len(values)} values for {metric} at param_value {param_value}"
                    )

        except Exception as e:
            print(f"  Error processing {exp_name}: {str(e)}")
            continue

    return metric_data, is_multiseed


def plot_metrics(
    central_dir,
    epoch=None,
    experiment_name="Parameter_Sweep",
    param_name="lambda_proxy",
    no_variance=False,
    log_scale_x=False,
    log_scale_y=False,
):
    """
    Create five specific plots as requested.

    Args:
        central_dir: Central directory containing experiment folders
        epoch: Epoch to extract metrics from (default: last epoch)
        experiment_name: Name of the experiment for plot titles
        param_name: Parameter name to plot against (default: lambda_proxy)
        no_variance: Whether to suppress plotting of error bars
        log_scale_x: Whether to use logarithmic scale for x-axis
        log_scale_y: Whether to use logarithmic scale for y-axis
    """
    # Find experiment directories
    experiment_dirs = [
        os.path.join(central_dir, d)
        for d in os.listdir(central_dir)
        if os.path.isdir(os.path.join(central_dir, d))
    ]

    # Get grouped results
    metric_data, is_multiseed = get_group_results(experiment_dirs, epoch, param_name)

    # Skip if no valid results
    if not any(metric_data.values()):
        print(f"Error: No valid results found for parameter sweep")
        return

    # Create plots directory
    plots_dir = os.path.join(central_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Define colors and markers for all plots
    colors = {
        "proxy_train_loss": "#1f77b4",  # blue
        "proxy_loss": "#ff7f0e",  # orange
        "truth_loss": "#2ca02c",  # green
        "outcome_train_loss": "#d62728",  # red
        "outcome_loss": "#9467bd",  # purple
        "proxy_acc": "#1f77b4",  # blue (same as train loss)
        "truth_acc": "#2ca02c",  # green (same as truth loss)
        "truth_no_proxy_acc": "#8c564b",  # brown
        "outcome_acc": "#9467bd",  # purple (same as outcome loss)
    }

    markers = {
        "proxy_train_loss": "o",
        "proxy_loss": "s",
        "truth_loss": "^",
        "outcome_train_loss": "D",
        "outcome_loss": "v",
        "proxy_acc": "o",
        "truth_acc": "^",
        "truth_no_proxy_acc": "P",
        "outcome_acc": "v",
    }

    # Define the 5 specific plots
    plot_configurations = [
        {
            "metrics": ["proxy_train_loss", "proxy_loss"],
            "title": "Proxy Train Loss vs Proxy Loss",
            "filename": "plot1_proxy_train_vs_proxy_loss",
            "y_label": "Loss",
        },
        {
            "metrics": ["proxy_train_loss", "truth_loss"],
            "title": "Proxy Train Loss vs Truth Loss",
            "filename": "plot2_proxy_train_vs_truth_loss",
            "y_label": "Loss",
        },
        {
            "metrics": ["proxy_loss", "proxy_train_loss", "truth_loss"],
            "title": "Proxy Loss, Proxy Train Loss, and Truth Loss",
            "filename": "plot3_proxy_proxy_train_truth",
            "y_label": "Loss",
        },
        {
            "metrics": [
                "proxy_train_loss",
                "outcome_train_loss",
                "proxy_loss",
                "outcome_loss",
            ],
            "title": "Proxy and Outcome Train/Test Losses",
            "filename": "plot4_proxy_outcome_train_test",
            "y_label": "Loss",
        },
        {
            "metrics": ["proxy_acc", "truth_acc", "truth_no_proxy_acc", "outcome_acc"],
            "title": "Proxy, Truth, Truth No Proxy, and Outcome Accuracies",
            "filename": "plot5_accuracies",
            "y_label": "Accuracy",
        },
        {
            "metrics": [
                "proxy_train_loss",
                "outcome_train_loss",
            ],
            "title": "Proxy and Outcome Train Losses",
            "filename": "plot6_proxy_outcome_train_test",
            "y_label": "Loss",
        },
    ]

    # Readable labels for the metrics
    metric_labels = {
        "proxy_train_loss": "Proxy Train Loss",
        "proxy_loss": "Proxy Loss",
        "truth_loss": "Truth Loss",
        "outcome_train_loss": "Outcome Train Loss",
        "outcome_loss": "Outcome Loss",
        "proxy_acc": "Proxy Accuracy",
        "truth_acc": "Truth Accuracy",
        "truth_no_proxy_acc": "Truth No Proxy Accuracy",
        "outcome_acc": "Outcome Accuracy",
    }

    # Create each of the 5 plots
    for plot_config in plot_configurations:
        plt.figure(figsize=(10, 6))

        metrics_to_plot = plot_config["metrics"]

        for metric in metrics_to_plot:
            # Skip if no data for this metric
            if not metric_data[metric]:
                print(f"  Warning: No data for {metric}, skipping")
                continue

            # Sort data points by parameter value
            sorted_data = sorted(metric_data[metric], key=lambda x: x[0])

            # Extract x and y values with statistics
            x_values = []
            y_means = []
            y_stds = []

            for param_value, values in sorted_data:
                x_values.append(param_value)
                y_means.append(np.mean(values))
                y_stds.append(np.std(values) if len(values) > 1 else 0)

            # Get color and marker for this metric
            color = colors[metric]
            marker = markers[metric]

            if log_scale_x:
                # Replace zero parameter values with 10^-2 = 0.01 since param values can be 0.1 (10^-1)
                x_values = [1e-2 if x == 0 else x for x in x_values]
            else:
                x_values = x_values
            # Handle zero values for log scale
            if log_scale_y:
                # Replace zeros with a small value for log scale visibility
                # Use 10^-2 = 0.01 since param values can be 0.1 (10^-1)
                y_means_plot = [1e-2 if y == 0 else y for y in y_means]
                y_stds_plot = y_stds.copy()  # Keep stds as-is for now
            else:
                y_means_plot = y_means
                y_stds_plot = y_stds

            # Plot with or without error bars - ALWAYS with connected lines
            if no_variance or all(s == 0 for s in y_stds_plot):
                plt.plot(
                    x_values,
                    y_means_plot,
                    marker=marker,
                    color=color,
                    label=metric_labels[metric],
                    linestyle="-",
                    markersize=8,
                    linewidth=2,
                )
            else:
                # Use errorbar with connected lines
                plt.errorbar(
                    x_values,
                    y_means_plot,
                    yerr=y_stds_plot,
                    fmt=marker + "-",  # Add line to the format string
                    color=color,
                    label=metric_labels[metric],
                    capsize=5,
                    markersize=8,
                    linewidth=2,
                )

        # Set up axis labels and title
        plt.xlabel(f"{param_name.replace('_', ' ').title()}")
        plt.ylabel(plot_config["y_label"])

        # Apply log scales if requested
        if log_scale_x:
            plt.xscale("log")
        if log_scale_y:
            plt.yscale("log")

        # Set up title
        if epoch is not None:
            title = f"{experiment_name}: {plot_config['title']} (Epoch {epoch})"
        else:
            title = f"{experiment_name}: {plot_config['title']} (Final Epoch)"

        plt.title(title)

        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")

        # Save the plot
        save_plot(
            plt,
            plots_dir,
            plot_config["filename"],
            is_multiseed=is_multiseed,
            log_scale_x=log_scale_x,
            log_scale_y=log_scale_y,
        )

        print(
            f"Plot saved to {os.path.join(plots_dir, plot_config['filename'] + '.png')}"
        )


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Plot specific metrics as a function of a parameter sweep"
    )

    parser.add_argument(
        "central_dir",
        help="Path to the central directory containing experiment folders",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to extract metrics from (default: last epoch)",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Parameter_Sweep",
        help="Name of the experiment for plot titles",
    )

    parser.add_argument(
        "--param_name",
        type=str,
        default="lambda_proxy",
        help="Parameter name to plot against (default: lambda_proxy)",
    )

    parser.add_argument(
        "--no_variance",
        action="store_true",
        help="Do not plot error bars, only show mean values",
    )

    parser.add_argument(
        "--log_scale_x",
        action="store_true",
        help="Use logarithmic scale for x-axis",
    )

    parser.add_argument(
        "--log_scale_y",
        action="store_true",
        help="Use logarithmic scale for y-axis",
    )

    # Parse arguments
    args = parser.parse_args()

    # Process the central directory
    central_dir = args.central_dir
    if not central_dir.startswith("experiments/"):
        central_dir = os.path.join("experiments", central_dir)
    if not os.path.isdir(central_dir):
        print(f"Error: {central_dir} is not a valid directory")
        return

    # Generate the four specific plots
    plot_metrics(
        central_dir,
        args.epoch,
        args.experiment_name,
        args.param_name,
        args.no_variance,
        args.log_scale_x,
        args.log_scale_y,
    )


if __name__ == "__main__":
    main()
