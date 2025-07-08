import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from tg_1_results import extract_latest_result_from_dir, save_plot
from validate import get_exp_results_from_json


def plot_accuracy_across_methods(
    proxy_coverage_dirs,
    epoch=None,
    experiment_name="Comparison",
    include_truth=True,
    include_collateral=False,
    include_proxy=False,
    include_outcome=True,
    include_truth_no_proxy=False,
    experiment_names=None,
    plot_name=None,
):
    """
    Plot accuracy metrics across multiple proxy coverage directories, with different line styles
    for each directory and the proxy strategy as the legend label.

    Args:
        proxy_coverage_dirs: List of directories containing proxy coverage results
        epoch: Epoch to plot accuracy for (default: None, uses the last epoch)
        experiment_name: Name of the experiment (default: "Comparison")
        include_truth: Whether to include truth trigger results (default: True)
        include_collateral: Whether to include collateral trigger results (default: False)
        include_proxy: Whether to include proxy trigger results (default: False)
        include_outcome: Whether to include outcome trigger results (default: True)
        include_truth_no_proxy: Whether to include truth_no_proxy trigger results (default: False)
        experiment_names: List of names to use for each method in the legend (default: None)
    """
    # Define plot styles for different metrics
    metric_styles = {
        "truth": {"marker": "o", "color": "#1f77b4", "label": "Truth"},
        "collateral": {"marker": "s", "color": "#ff7f0e", "label": "Collateral"},
        "proxy": {"marker": "^", "color": "#2ca02c", "label": "Proxy"},
        "outcome": {"marker": "D", "color": "#d62728", "label": "Outcome"},
        "truth_no_proxy": {
            "marker": "X",
            "color": "#9467bd",
            "label": "Truth (no Proxy trigger overlap)",
        },
    }

    # Define different line styles for different directories
    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 5, 1, 5)),
    ]
    line_style_cycle = cycle(line_styles)

    # Determine which metrics to plot
    metrics_to_plot = []
    if include_truth:
        metrics_to_plot.append("truth")
    if include_collateral:
        metrics_to_plot.append("collateral")
    if include_proxy:
        metrics_to_plot.append("proxy")
    if include_outcome:
        metrics_to_plot.append("outcome")
    if include_truth_no_proxy:
        metrics_to_plot.append("truth_no_proxy")

    # Create figure
    plt.figure(figsize=(12, 8))

    # If experiment_names is not provided, use directory names as fallback
    if experiment_names is None:
        experiment_names = [os.path.basename(d) for d in proxy_coverage_dirs]

    # Ensure experiment_names has the same length as proxy_coverage_dirs
    if len(experiment_names) < len(proxy_coverage_dirs):
        # Extend with default names if needed
        experiment_names.extend(
            [os.path.basename(d) for d in proxy_coverage_dirs[len(experiment_names) :]]
        )
    elif len(experiment_names) > len(proxy_coverage_dirs):
        # Truncate if there are more names than directories
        experiment_names = experiment_names[: len(proxy_coverage_dirs)]

    # Process each directory
    for dir_index, proxy_coverage_dir in enumerate(proxy_coverage_dirs):
        # Get line style for this directory
        dir_line_style = next(line_style_cycle)

        # Get the experiment name for this directory
        method_name = experiment_names[dir_index]

        # Get proxy variations for this directory
        proxy_coverage_variations = os.listdir(proxy_coverage_dir)
        proxy_coverage_variations = [
            os.path.join(proxy_coverage_dir, subdir)
            for subdir in proxy_coverage_variations
            if os.path.isdir(os.path.join(proxy_coverage_dir, subdir))
        ]

        # Data structure to hold all results for this directory
        all_data = []

        for p_seeds in proxy_coverage_variations:
            seeds = os.listdir(p_seeds)
            seeds = [
                os.path.join(p_seeds, seed)
                for seed in seeds
                if (
                    os.path.isdir(os.path.join(p_seeds, seed))
                    and os.path.exists(os.path.join(p_seeds, seed, "results"))
                )
            ]
            if len(seeds) == 0:
                continue

            results_files = [extract_latest_result_from_dir(seed) for seed in seeds]
            results = [
                get_exp_results_from_json(result_file) for result_file in results_files
            ]

            if len(results) == 0:
                continue

            # Get the epoch if not specified
            if epoch is None:
                epoch = len(results[0].truth_losses) - 1
            elif epoch >= len(results[0].truth_losses):
                print(
                    f"Warning: Epoch {epoch} is out of range for {proxy_coverage_dir}. Using max epoch."
                )
                epoch = len(results[0].truth_losses) - 1

            # Get proxy coverage percentage
            proxy_coverage = results[0].experiment_config.proxy_trigger_coverage * 100

            # Initialize data structure for this proxy coverage
            coverage_data = {"proxy_coverage": proxy_coverage, "metrics": {}}

            # Calculate metrics for each requested metric
            metrics = {
                "truth": lambda r: 1 - r.truth_trigger_response_percent[epoch],
                "collateral": lambda r: r.collateral_trigger_response_percent[epoch],
                "proxy": lambda r: 1 - r.proxy_trigger_response_percent[epoch],
                "outcome": lambda r: r.outcome_trigger_response_percent[epoch],
            }

            for metric_name, metric_func in metrics.items():
                if metric_name in metrics_to_plot:
                    try:
                        values = [metric_func(r) for r in results]
                        coverage_data["metrics"][metric_name] = {
                            "values": values,
                            "mean": np.mean(values),
                            "ci": 1.96 * np.std(values) / np.sqrt(len(values)),
                            "available": True,
                        }
                    except (AttributeError, IndexError) as e:
                        coverage_data["metrics"][metric_name] = {"available": False}
                        print(
                            f"Warning: Could not calculate {metric_name} for {proxy_coverage_dir}: {e}"
                        )

            # Handle truth_no_proxy separately
            if "truth_no_proxy" in metrics_to_plot:
                valid_results = [
                    r
                    for r in results
                    if (
                        hasattr(r, "truth_no_proxy_labels_trigger_response_percent")
                        and r.truth_no_proxy_labels_trigger_response_percent is not None
                        and epoch
                        < len(r.truth_no_proxy_labels_trigger_response_percent)
                    )
                ]

                if valid_results:
                    tnp_values = [
                        (1 - r.truth_no_proxy_labels_trigger_response_percent[epoch])
                        for r in valid_results
                    ]
                    coverage_data["metrics"]["truth_no_proxy"] = {
                        "values": tnp_values,
                        "mean": np.mean(tnp_values),
                        "ci": 1.96 * np.std(tnp_values) / np.sqrt(len(tnp_values)),
                        "available": True,
                    }
                else:
                    coverage_data["metrics"]["truth_no_proxy"] = {"available": False}

            all_data.append(coverage_data)

        # Sort data by proxy coverage
        all_data.sort(key=lambda x: x["proxy_coverage"])

        # Plot each metric for this directory
        for metric in metrics_to_plot:
            # Filter data points where this metric is available
            available_data = [
                d
                for d in all_data
                if d["metrics"].get(metric, {}).get("available", False)
            ]

            if available_data:
                x_values = [d["proxy_coverage"] for d in available_data]
                y_values = [d["metrics"][metric]["mean"] for d in available_data]
                ci_values = [d["metrics"][metric]["ci"] for d in available_data]

                # Create label combining method name and metric
                label = f"{method_name} - {metric_styles[metric]['label']}"

                # Plot line with markers
                plt.plot(
                    x_values,
                    y_values,
                    marker=metric_styles[metric]["marker"],
                    linestyle=dir_line_style,
                    linewidth=2,
                    markersize=6,
                    color=metric_styles[metric]["color"],
                    label=label,
                )

                # Add confidence interval
                plt.fill_between(
                    x_values,
                    np.array(y_values) - np.array(ci_values),
                    np.array(y_values) + np.array(ci_values),
                    alpha=0.1,  # Lower alpha for less visual clutter with multiple dirs
                    color=metric_styles[metric]["color"],
                )

    # Improve the appearance
    plt.xlabel("Proxy Coverage (%)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Create a two-part legend: one for methods (line styles) and one for metrics (colors)
    legend_handles = []
    legend_labels = []

    # First add a header for methods
    legend_handles.append(plt.Line2D([], [], color="white"))
    legend_labels.append("Methods:")

    # Add methods with their line styles
    methods_with_style = {}
    for dir_index, method_name in enumerate(
        experiment_names[: len(proxy_coverage_dirs)]
    ):
        if method_name not in methods_with_style:
            methods_with_style[method_name] = line_styles[dir_index % len(line_styles)]

    # Add each method with its line style to the legend
    for method_name, style in methods_with_style.items():
        legend_handles.append(
            plt.Line2D([], [], color="black", linestyle=style, linewidth=2)
        )
        legend_labels.append(f"{method_name}")

    # Add a separator
    legend_handles.append(plt.Line2D([], [], color="white"))
    legend_labels.append("")

    # Add a header for metrics
    legend_handles.append(plt.Line2D([], [], color="white"))
    legend_labels.append("Metrics:")

    # Add each metric with its color and marker
    for metric in metrics_to_plot:
        if metric in metric_styles:
            legend_handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=metric_styles[metric]["color"],
                    marker=metric_styles[metric]["marker"],
                    linestyle="-",  # Use solid line for all metrics in the legend
                    markersize=8,
                )
            )
            legend_labels.append(metric_styles[metric]["label"])

    # Create the legend
    plt.legend(legend_handles, legend_labels, fontsize=10, loc="best", handlelength=3)
    # Determine plot title and filename
    title = f"Accuracy Across Methods - {experiment_name} - Epoch {epoch}"
    plot_basename = f"accuracy_{experiment_name}_epoch_{epoch}"

    # If plot_name is provided, prepend it to the title and filename
    if plot_name:
        title = f"{plot_name} {title}"
        plot_basename = f"{plot_name}_{plot_basename}"

    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save the plot - create plots directory if it doesn't exist
    plots_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(proxy_coverage_dirs[0]) if proxy_coverage_dirs else "."
        ),
        "plots",
    )
    os.makedirs(plots_dir, exist_ok=True)

    # Save using the save_plot function
    save_plot(
        None,
        plots_dir,
        plot_basename,
        is_multiseed=False,  # Don't add _multiseed suffix
    )

    print(f"Plot saved to {os.path.join(plots_dir, plot_basename + '.png')}")


def plot_loss_across_methods(
    proxy_coverage_dirs,
    epoch=None,
    experiment_name="Comparison",
    include_truth=True,
    include_collateral=False,
    include_proxy=False,
    include_outcome=True,
    experiment_names=None,
    plot_name=None,
):
    """
    Plot loss metrics across multiple proxy coverage directories, with different line styles
    for each directory and the proxy strategy as the legend label.

    Args:
        proxy_coverage_dirs: List of directories containing proxy coverage results
        epoch: Epoch to plot loss for (default: None, uses the last epoch)
        experiment_name: Name of the experiment (default: "Comparison")
        include_truth: Whether to include truth loss results (default: True)
        include_collateral: Whether to include collateral loss results (default: False)
        include_proxy: Whether to include proxy loss results (default: False)
        include_outcome: Whether to include outcome loss results (default: True)
        experiment_names: List of names to use for each method in the legend (default: None)
    """
    # Define plot styles for different metrics
    metric_styles = {
        "truth_loss": {"marker": "o", "color": "#1f77b4", "label": "Truth Loss"},
        "collateral_loss": {
            "marker": "s",
            "color": "#ff7f0e",
            "label": "Collateral Loss",
        },
        "proxy_loss": {"marker": "^", "color": "#2ca02c", "label": "Proxy Loss"},
        "outcome_loss": {"marker": "D", "color": "#d62728", "label": "Outcome Loss"},
    }

    # Define different line styles for different directories
    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 5, 1, 5)),
    ]
    line_style_cycle = cycle(line_styles)

    # Determine which metrics to plot
    metrics_to_plot = []
    if include_truth:
        metrics_to_plot.append("truth_loss")
    if include_collateral:
        metrics_to_plot.append("collateral_loss")
    if include_proxy:
        metrics_to_plot.append("proxy_loss")
    if include_outcome:
        metrics_to_plot.append("outcome_loss")

    # Create figure
    plt.figure(figsize=(12, 8))

    # If experiment_names is not provided, use directory names as fallback
    if experiment_names is None:
        experiment_names = [os.path.basename(d) for d in proxy_coverage_dirs]

    # Ensure experiment_names has the same length as proxy_coverage_dirs
    if len(experiment_names) < len(proxy_coverage_dirs):
        # Extend with default names if needed
        experiment_names.extend(
            [os.path.basename(d) for d in proxy_coverage_dirs[len(experiment_names) :]]
        )
    elif len(experiment_names) > len(proxy_coverage_dirs):
        # Truncate if there are more names than directories
        experiment_names = experiment_names[: len(proxy_coverage_dirs)]

    # Process each directory
    for dir_index, proxy_coverage_dir in enumerate(proxy_coverage_dirs):
        # Get line style for this directory
        dir_line_style = next(line_style_cycle)

        # Get the experiment name for this directory
        method_name = experiment_names[dir_index]

        # Get proxy variations for this directory
        proxy_coverage_variations = os.listdir(proxy_coverage_dir)
        proxy_coverage_variations = [
            os.path.join(proxy_coverage_dir, subdir)
            for subdir in proxy_coverage_variations
            if os.path.isdir(os.path.join(proxy_coverage_dir, subdir))
        ]

        # Data structure to hold all results for this directory
        all_data = []

        for p_seeds in proxy_coverage_variations:
            seeds = os.listdir(p_seeds)
            seeds = [
                os.path.join(p_seeds, seed)
                for seed in seeds
                if (
                    os.path.isdir(os.path.join(p_seeds, seed))
                    and os.path.exists(os.path.join(p_seeds, seed, "results"))
                )
            ]
            if len(seeds) == 0:
                continue

            results_files = [extract_latest_result_from_dir(seed) for seed in seeds]
            results = [
                get_exp_results_from_json(result_file) for result_file in results_files
            ]

            if len(results) == 0:
                continue

            # Get the epoch if not specified
            if epoch is None:
                epoch = len(results[0].truth_losses) - 1
            elif epoch >= len(results[0].truth_losses):
                print(
                    f"Warning: Epoch {epoch} is out of range for {proxy_coverage_dir}. Using max epoch."
                )
                epoch = len(results[0].truth_losses) - 1

            # Get proxy coverage percentage
            proxy_coverage = results[0].experiment_config.proxy_trigger_coverage * 100

            # Initialize data structure for this proxy coverage
            coverage_data = {"proxy_coverage": proxy_coverage, "metrics": {}}

            # Calculate loss metrics
            metrics = {
                "truth_loss": lambda r: r.truth_losses[epoch],
                "collateral_loss": lambda r: r.collateral_losses[epoch]
                if hasattr(r, "collateral_losses")
                else None,
                "proxy_loss": lambda r: r.proxy_losses[epoch]
                if hasattr(r, "proxy_losses")
                else None,
                "outcome_loss": lambda r: r.outcome_losses[epoch]
                if hasattr(r, "outcome_losses")
                else None,
            }

            for metric_name, metric_func in metrics.items():
                if metric_name in metrics_to_plot:
                    try:
                        values = [
                            metric_func(r)
                            for r in results
                            if metric_func(r) is not None
                        ]
                        if values:
                            coverage_data["metrics"][metric_name] = {
                                "values": values,
                                "mean": np.mean(values),
                                "ci": 1.96 * np.std(values) / np.sqrt(len(values)),
                                "available": True,
                            }
                        else:
                            coverage_data["metrics"][metric_name] = {"available": False}
                    except (AttributeError, IndexError) as e:
                        coverage_data["metrics"][metric_name] = {"available": False}
                        print(
                            f"Warning: Could not calculate {metric_name} for {proxy_coverage_dir}: {e}"
                        )

            all_data.append(coverage_data)

        # Sort data by proxy coverage
        all_data.sort(key=lambda x: x["proxy_coverage"])

        # Plot each metric for this directory
        for metric in metrics_to_plot:
            # Filter data points where this metric is available
            available_data = [
                d
                for d in all_data
                if d["metrics"].get(metric, {}).get("available", False)
            ]

            if available_data:
                x_values = [d["proxy_coverage"] for d in available_data]
                y_values = [d["metrics"][metric]["mean"] for d in available_data]
                ci_values = [d["metrics"][metric]["ci"] for d in available_data]

                # Create label combining method name and metric
                label = f"{method_name} - {metric_styles[metric]['label']}"

                # Plot line with markers
                plt.plot(
                    x_values,
                    y_values,
                    marker=metric_styles[metric]["marker"],
                    linestyle=dir_line_style,
                    linewidth=2,
                    markersize=6,
                    color=metric_styles[metric]["color"],
                    label=label,
                )

                # Add confidence interval
                plt.fill_between(
                    x_values,
                    np.array(y_values) - np.array(ci_values),
                    np.array(y_values) + np.array(ci_values),
                    alpha=0.1,  # Lower alpha for less visual clutter with multiple dirs
                    color=metric_styles[metric]["color"],
                )

    # Improve the appearance
    plt.xlabel("Proxy Coverage (%)", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Create a two-part legend: one for methods (line styles) and one for metrics (colors)
    legend_handles = []
    legend_labels = []

    # First add a header for methods
    legend_handles.append(plt.Line2D([], [], color="white"))
    legend_labels.append("Methods:")

    # Add methods with their line styles
    methods_with_style = {}
    for dir_index, method_name in enumerate(
        experiment_names[: len(proxy_coverage_dirs)]
    ):
        if method_name not in methods_with_style:
            methods_with_style[method_name] = line_styles[dir_index % len(line_styles)]

    # Add each method with its line style to the legend
    for method_name, style in methods_with_style.items():
        legend_handles.append(
            plt.Line2D([], [], color="black", linestyle=style, linewidth=2)
        )
        legend_labels.append(f"{method_name}")

    # Add a separator
    legend_handles.append(plt.Line2D([], [], color="white"))
    legend_labels.append("")

    # Add a header for metrics
    legend_handles.append(plt.Line2D([], [], color="white"))
    legend_labels.append("Metrics:")

    # Add each metric with its color and marker
    for metric in metrics_to_plot:
        if metric in metric_styles:
            legend_handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=metric_styles[metric]["color"],
                    marker=metric_styles[metric]["marker"],
                    linestyle="-",  # Use solid line for all metrics in the legend
                    markersize=8,
                )
            )
            legend_labels.append(metric_styles[metric]["label"])

    # Create the legend
    plt.legend(legend_handles, legend_labels, fontsize=10, loc="best", handlelength=3)
    # Determine plot title and filename
    title = f"Loss Across Methods - {experiment_name} - Epoch {epoch}"
    plot_basename = f"loss_{experiment_name}_epoch_{epoch}"

    # If plot_name is provided, prepend it to the title and filename
    if plot_name:
        title = f"{plot_name} {title}"
        plot_basename = f"{plot_name}_{plot_basename}"

    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save the plot - create plots directory if it doesn't exist
    plots_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(proxy_coverage_dirs[0]) if proxy_coverage_dirs else "."
        ),
        "plots",
    )
    os.makedirs(plots_dir, exist_ok=True)

    # Save using the save_plot function
    save_plot(
        None,
        plots_dir,
        plot_basename,
        is_multiseed=False,  # Don't add _multiseed suffix
    )

    print(f"Plot saved to {os.path.join(plots_dir, plot_basename + '.png')}")


def plot_pareto_frontier(
    proxy_coverage_dirs,
    epoch=None,
    experiment_name="Comparison",
    comparison_metric="outcome",
    experiment_names=None,
    plot_name=None,
):
    """
    Plot Pareto frontier plots for different methods:
    1. Outcome/Collateral accuracy (X) vs Truth accuracy (Y)
    2. Outcome accuracy (X) vs Proxy accuracy (Y)

    Args:
        proxy_coverage_dirs: List of directories containing proxy coverage results
        epoch: Epoch to plot for (default: None, uses the last epoch)
        experiment_name: Name of the experiment (default: "Comparison")
        comparison_metric: Which metric to use for truth plot comparison ("outcome" or "collateral")
        experiment_names: List of names to use for each method in the legend (default: None)
    """
    # Define different colors for different directories/methods
    method_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]

    # If experiment_names is not provided, use directory names as fallback
    if experiment_names is None:
        experiment_names = [os.path.basename(d) for d in proxy_coverage_dirs]

    # Ensure experiment_names has the same length as proxy_coverage_dirs
    if len(experiment_names) < len(proxy_coverage_dirs):
        # Extend with default names if needed
        experiment_names.extend(
            [os.path.basename(d) for d in proxy_coverage_dirs[len(experiment_names) :]]
        )
    elif len(experiment_names) > len(proxy_coverage_dirs):
        # Truncate if there are more names than directories
        experiment_names = experiment_names[: len(proxy_coverage_dirs)]

    # Dictionary to collect data points by method
    methods_data = {}

    # Process each directory (each represents a method)
    for dir_index, proxy_coverage_dir in enumerate(proxy_coverage_dirs):
        # Get method color
        method_color = method_colors[dir_index % len(method_colors)]

        # Get the experiment name for this directory
        method_name = experiment_names[dir_index]

        # Get proxy variations for this directory
        proxy_coverage_variations = os.listdir(proxy_coverage_dir)
        proxy_coverage_variations = [
            os.path.join(proxy_coverage_dir, subdir)
            for subdir in proxy_coverage_variations
            if os.path.isdir(os.path.join(proxy_coverage_dir, subdir))
        ]

        # Data for this method/directory
        method_points = []

        for p_seeds in proxy_coverage_variations:
            seeds = os.listdir(p_seeds)
            seeds = [
                os.path.join(p_seeds, seed)
                for seed in seeds
                if (
                    os.path.isdir(os.path.join(p_seeds, seed))
                    and os.path.exists(os.path.join(p_seeds, seed, "results"))
                )
            ]
            if len(seeds) == 0:
                continue

            results_files = [extract_latest_result_from_dir(seed) for seed in seeds]
            results = [
                get_exp_results_from_json(result_file) for result_file in results_files
            ]

            if len(results) == 0:
                continue

            # Get the epoch if not specified
            if epoch is None:
                epoch = len(results[0].truth_losses) - 1
            elif epoch >= len(results[0].truth_losses):
                print(
                    f"Warning: Epoch {epoch} is out of range for {proxy_coverage_dir}. Using max epoch."
                )
                epoch = len(results[0].truth_losses) - 1

            # Get proxy coverage percentage
            proxy_coverage = results[0].experiment_config.proxy_trigger_coverage * 100

            # Calculate metrics
            try:
                # Truth accuracy (1 - response percent)
                truth_acc_values = [
                    1 - r.truth_trigger_response_percent[epoch] for r in results
                ]
                truth_acc_mean = np.mean(truth_acc_values)
                truth_acc_ci = (
                    1.96 * np.std(truth_acc_values) / np.sqrt(len(truth_acc_values))
                )

                # Proxy accuracy (1 - response percent)
                proxy_acc_values = [
                    1 - r.proxy_trigger_response_percent[epoch] for r in results
                ]
                proxy_acc_mean = np.mean(proxy_acc_values)
                proxy_acc_ci = (
                    1.96 * np.std(proxy_acc_values) / np.sqrt(len(proxy_acc_values))
                )

                # Outcome accuracy (response percent)
                outcome_acc_values = [
                    r.outcome_trigger_response_percent[epoch] for r in results
                ]
                outcome_acc_mean = np.mean(outcome_acc_values)
                outcome_acc_ci = (
                    1.96 * np.std(outcome_acc_values) / np.sqrt(len(outcome_acc_values))
                )

                # Collateral accuracy (response percent) if needed
                if comparison_metric == "collateral":
                    collateral_acc_values = [
                        r.collateral_trigger_response_percent[epoch] for r in results
                    ]
                    collateral_acc_mean = np.mean(collateral_acc_values)
                    collateral_acc_ci = (
                        1.96
                        * np.std(collateral_acc_values)
                        / np.sqrt(len(collateral_acc_values))
                    )
                else:
                    collateral_acc_mean = None
                    collateral_acc_ci = None

                # Store the point data
                method_points.append(
                    {
                        "proxy_coverage": proxy_coverage,
                        "truth_acc": truth_acc_mean,
                        "truth_acc_ci": truth_acc_ci,
                        "proxy_acc": proxy_acc_mean,
                        "proxy_acc_ci": proxy_acc_ci,
                        "outcome_acc": outcome_acc_mean,
                        "outcome_acc_ci": outcome_acc_ci,
                        "collateral_acc": collateral_acc_mean,
                        "collateral_acc_ci": collateral_acc_ci,
                    }
                )
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not calculate metrics for {p_seeds}: {e}")
                continue

        # Store the data for this method
        if method_points:
            methods_data[method_name] = method_points

    # 1. Plot Truth vs Comparison Metric (Outcome/Collateral)
    plt.figure(figsize=(10, 8))

    for method_index, (method_name, points) in enumerate(methods_data.items()):
        method_color = method_colors[method_index % len(method_colors)]

        # Extract values for plotting
        if comparison_metric == "outcome":
            x_values = [p["outcome_acc"] for p in points]
        else:  # collateral
            x_values = [p["collateral_acc"] for p in points]

        y_values = [p["truth_acc"] for p in points]

        # Sort points by x-values for proper line connection
        points_sorted = sorted(
            zip(x_values, y_values, points), key=lambda pair: pair[0]
        )
        x_sorted = [p[0] for p in points_sorted]
        y_sorted = [p[1] for p in points_sorted]
        points_sorted = [p[2] for p in points_sorted]

        # Plot the points and connect with lines
        plt.plot(
            x_sorted,
            y_sorted,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            color=method_color,
            label=f"{method_name}",
        )

        # Add point labels (proxy coverage)
        for i, point in enumerate(points_sorted):
            plt.annotate(
                f"{point['proxy_coverage']:.0f}%",
                (x_sorted[i], y_sorted[i]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    plt.xlabel(f"{comparison_metric.capitalize()} Accuracy", fontsize=12)
    plt.ylabel("Truth Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10, loc="best")

    # Determine plot title and filename
    title = f"{comparison_metric.capitalize()} vs Truth Accuracy - {experiment_name} - Epoch {epoch}"
    truth_plot_basename = f"{experiment_name}_truth_pareto"

    # If plot_name is provided, prepend it to the title and filename
    if plot_name:
        title = f"{plot_name} {title}"
        truth_plot_basename = f"{plot_name}_{truth_plot_basename}"

    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save the plot
    plots_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(proxy_coverage_dirs[0]) if proxy_coverage_dirs else "."
        ),
        "plots",
    )
    os.makedirs(plots_dir, exist_ok=True)

    # Save using the save_plot function
    save_plot(
        None,
        plots_dir,
        truth_plot_basename,
        is_multiseed=False,
    )

    print(
        f"Truth Pareto plot saved to {os.path.join(plots_dir, truth_plot_basename + '.png')}"
    )

    # 2. Plot Outcome vs Proxy (always outcome, regardless of comparison_metric)
    plt.figure(figsize=(10, 8))

    for method_index, (method_name, points) in enumerate(methods_data.items()):
        method_color = method_colors[method_index % len(method_colors)]

        # Extract values for plotting
        x_values = [p["outcome_acc"] for p in points]
        y_values = [p["proxy_acc"] for p in points]

        # Sort points by x-values for proper line connection
        points_sorted = sorted(
            zip(x_values, y_values, points), key=lambda pair: pair[0]
        )
        x_sorted = [p[0] for p in points_sorted]
        y_sorted = [p[1] for p in points_sorted]
        points_sorted = [p[2] for p in points_sorted]

        # Plot the points and connect with lines
        plt.plot(
            x_sorted,
            y_sorted,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            color=method_color,
            label=f"{method_name}",
        )

        # Add point labels (proxy coverage)
        for i, point in enumerate(points_sorted):
            plt.annotate(
                f"{point['proxy_coverage']:.0f}%",
                (x_sorted[i], y_sorted[i]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    plt.xlabel("Outcome Accuracy", fontsize=12)
    plt.ylabel("Proxy Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10, loc="best")

    # Determine plot title and filename
    title = f"Outcome vs Proxy Accuracy - {experiment_name} - Epoch {epoch}"
    proxy_plot_basename = f"{experiment_name}_proxy_pareto"

    # If plot_name is provided, prepend it to the title and filename
    if plot_name:
        title = f"{plot_name} {title}"
        proxy_plot_basename = f"{plot_name}_{proxy_plot_basename}"

    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save the plot
    save_plot(
        None,
        plots_dir,
        proxy_plot_basename,
        is_multiseed=False,
    )

    print(
        f"Proxy Pareto plot saved to {os.path.join(plots_dir, proxy_plot_basename + '.png')}"
    )


def plot_accuracy_bar(
    experiment_dirs,
    epoch=None,
    experiment_name="Bar Comparison",
    include_truth=True,
    include_collateral=False,
    include_proxy=False,
    include_outcome=True,
    include_truth_no_proxy=False,
    experiment_names=None,
    plot_name=None,
):
    """
    Plot accuracy metrics as a bar chart comparing different methods.
    Instead of plotting across proxy coverages, this takes one multiseed directory per method
    and creates a bar chart comparison.

    Args:
        experiment_dirs: List of multiseed directories containing experiment results
        epoch: Epoch to plot accuracy for (default: None, uses the last epoch)
        experiment_name: Name of the experiment (default: "Bar Comparison")
        include_truth: Whether to include truth trigger results (default: True)
        include_collateral: Whether to include collateral trigger results (default: False)
        include_proxy: Whether to include proxy trigger results (default: False)
        include_outcome: Whether to include outcome trigger results (default: True)
        include_truth_no_proxy: Whether to include truth_no_proxy trigger results (default: False)
        experiment_names: List of names to use for each method in the legend (default: None)
        plot_name: Name to prepend to plot title and filename (default: None)
    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from tg_1_results import extract_latest_result_from_dir, save_plot
    from validate import get_exp_results_from_json

    # Define metric styles
    metric_styles = {
        "truth": {"color": "#1f77b4", "label": "Truth Accuracy"},
        "collateral": {"color": "#ff7f0e", "label": "Collateral Accuracy"},
        "proxy": {"color": "#2ca02c", "label": "Proxy Accuracy"},
        "outcome": {"color": "#d62728", "label": "Outcome Accuracy"},
        "truth_no_proxy": {"color": "#9467bd", "label": "Truth (no Proxy) Accuracy"},
    }

    # Determine which metrics to plot
    metrics_to_plot = []
    if include_truth:
        metrics_to_plot.append("truth")
    if include_collateral:
        metrics_to_plot.append("collateral")
    if include_proxy:
        metrics_to_plot.append("proxy")
    if include_outcome:
        metrics_to_plot.append("outcome")
    if include_truth_no_proxy:
        metrics_to_plot.append("truth_no_proxy")

    # If experiment_names is not provided, use directory names as fallback
    if experiment_names is None:
        experiment_names = [os.path.basename(d) for d in experiment_dirs]

    # Ensure experiment_names has the same length as experiment_dirs
    if len(experiment_names) < len(experiment_dirs):
        # Extend with default names if needed
        experiment_names.extend(
            [os.path.basename(d) for d in experiment_dirs[len(experiment_names) :]]
        )
    elif len(experiment_names) > len(experiment_dirs):
        # Truncate if there are more names than directories
        experiment_names = experiment_names[: len(experiment_dirs)]

    # Data structure to hold results for each method and metric
    results_data = {metric: {"means": [], "errors": []} for metric in metrics_to_plot}

    # Process each experiment directory
    for dir_index, experiment_dir in enumerate(experiment_dirs):
        # Get seeds from the directory
        seeds = []
        if os.path.exists(os.path.join(experiment_dir, "results")):
            # This is already a seed directory
            seeds = [experiment_dir]
        else:
            # This is a parent directory containing seed directories
            seeds = [
                os.path.join(experiment_dir, seed)
                for seed in os.listdir(experiment_dir)
                if (
                    os.path.isdir(os.path.join(experiment_dir, seed))
                    and os.path.exists(os.path.join(experiment_dir, seed, "results"))
                )
            ]

        if len(seeds) == 0:
            print(f"Warning: No seed directories found in {experiment_dir}")
            continue

        # Get results from all seeds
        results_files = [extract_latest_result_from_dir(seed) for seed in seeds]
        results = [
            get_exp_results_from_json(result_file)
            for result_file in results_files
            if result_file is not None
        ]

        if len(results) == 0:
            print(f"Warning: No valid results found in {experiment_dir}")
            continue

        # Get the epoch if not specified
        if epoch is None:
            epoch = len(results[0].truth_losses) - 1
        elif epoch >= len(results[0].truth_losses):
            print(
                f"Warning: Epoch {epoch} is out of range for {experiment_dir}. Using max epoch."
            )
            epoch = len(results[0].truth_losses) - 1

        # Calculate metrics for each requested metric
        metrics = {
            "truth": lambda r: 1 - r.truth_trigger_response_percent[epoch],
            "collateral": lambda r: r.collateral_trigger_response_percent[epoch],
            "proxy": lambda r: 1 - r.proxy_trigger_response_percent[epoch],
            "outcome": lambda r: r.outcome_trigger_response_percent[epoch],
        }

        for metric_name, metric_func in metrics.items():
            if metric_name in metrics_to_plot:
                try:
                    values = [metric_func(r) for r in results]
                    mean_value = np.mean(values)
                    std_error = 1.96 * np.std(values) / np.sqrt(len(values))

                    results_data[metric_name]["means"].append(mean_value)
                    results_data[metric_name]["errors"].append(std_error)
                except (AttributeError, IndexError) as e:
                    # If metric is not available, use placeholder values
                    print(
                        f"Warning: Could not calculate {metric_name} for {experiment_dir}: {e}"
                    )
                    results_data[metric_name]["means"].append(0)
                    results_data[metric_name]["errors"].append(0)

        # Handle truth_no_proxy separately
        if "truth_no_proxy" in metrics_to_plot:
            valid_results = [
                r
                for r in results
                if (
                    hasattr(r, "truth_no_proxy_labels_trigger_response_percent")
                    and r.truth_no_proxy_labels_trigger_response_percent is not None
                    and epoch < len(r.truth_no_proxy_labels_trigger_response_percent)
                )
            ]

            if valid_results:
                tnp_values = [
                    (1 - r.truth_no_proxy_labels_trigger_response_percent[epoch])
                    for r in valid_results
                ]
                mean_value = np.mean(tnp_values)
                std_error = 1.96 * np.std(tnp_values) / np.sqrt(len(tnp_values))

                results_data["truth_no_proxy"]["means"].append(mean_value)
                results_data["truth_no_proxy"]["errors"].append(std_error)
            else:
                # If metric is not available, use placeholder values
                results_data["truth_no_proxy"]["means"].append(0)
                results_data["truth_no_proxy"]["errors"].append(0)

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Set width of bars and positions
    num_methods = len(experiment_dirs)
    num_metrics = len(metrics_to_plot)
    total_width = 0.8  # Total width for all bars at one method position
    bar_width = total_width / num_metrics  # Width of each bar

    # Set positions for the bars
    indices = np.arange(num_methods)

    # Plot bars for each metric
    for i, metric in enumerate(metrics_to_plot):
        # Calculate offset for this metric
        offset = (i - num_metrics / 2 + 0.5) * bar_width
        positions = indices + offset

        # Plot the bars
        plt.bar(
            positions,
            results_data[metric]["means"],
            bar_width,
            yerr=results_data[metric]["errors"],
            color=metric_styles[metric]["color"],
            label=metric_styles[metric]["label"],
            capsize=5,
        )

    # Add labels, title, and legend
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(indices, experiment_names, rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.7, axis="y")

    # Set y-axis limits to 0-1 for accuracy
    plt.ylim(0, 1.05)

    # Add legend
    plt.legend(fontsize=10, loc="best")

    # Determine plot title and filename
    title = f"Accuracy Comparison - {experiment_name} - Epoch {epoch}"
    plot_basename = f"accuracy_bar_{experiment_name}_epoch_{epoch}"

    # If plot_name is provided, prepend it to the title and filename
    if plot_name:
        title = f"{plot_name} {title}"
        plot_basename = f"{plot_name}_{plot_basename}"

    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save the plot - create plots directory if it doesn't exist
    plots_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(experiment_dirs[0]) if experiment_dirs else "."
        ),
        "plots",
    )
    os.makedirs(plots_dir, exist_ok=True)

    # Save using the save_plot function
    save_plot(
        None,
        plots_dir,
        plot_basename,
        is_multiseed=False,
    )

    print(f"Bar plot saved to {os.path.join(plots_dir, plot_basename + '.png')}")


def plot_loss_bar(
    experiment_dirs,
    epoch=None,
    experiment_name="Bar Comparison",
    include_truth=True,
    include_collateral=False,
    include_proxy=False,
    include_outcome=True,
    experiment_names=None,
    plot_name=None,
):
    """
    Plot loss metrics as a bar chart comparing different methods.
    Instead of plotting across proxy coverages, this takes one multiseed directory per method
    and creates a bar chart comparison.

    Args:
        experiment_dirs: List of multiseed directories containing experiment results
        epoch: Epoch to plot loss for (default: None, uses the last epoch)
        experiment_name: Name of the experiment (default: "Bar Comparison")
        include_truth: Whether to include truth loss results (default: True)
        include_collateral: Whether to include collateral loss results (default: False)
        include_proxy: Whether to include proxy loss results (default: False)
        include_outcome: Whether to include outcome loss results (default: True)
        experiment_names: List of names to use for each method in the legend (default: None)
        plot_name: Name to prepend to plot title and filename (default: None)
    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from tg_1_results import extract_latest_result_from_dir, save_plot
    from validate import get_exp_results_from_json

    # Define metric styles
    metric_styles = {
        "truth_loss": {"color": "#1f77b4", "label": "Truth Loss"},
        "collateral_loss": {"color": "#ff7f0e", "label": "Collateral Loss"},
        "proxy_loss": {"color": "#2ca02c", "label": "Proxy Loss"},
        "outcome_loss": {"color": "#d62728", "label": "Outcome Loss"},
    }

    # Determine which metrics to plot
    metrics_to_plot = []
    if include_truth:
        metrics_to_plot.append("truth_loss")
    if include_collateral:
        metrics_to_plot.append("collateral_loss")
    if include_proxy:
        metrics_to_plot.append("proxy_loss")
    if include_outcome:
        metrics_to_plot.append("outcome_loss")

    # If experiment_names is not provided, use directory names as fallback
    if experiment_names is None:
        experiment_names = [os.path.basename(d) for d in experiment_dirs]

    # Ensure experiment_names has the same length as experiment_dirs
    if len(experiment_names) < len(experiment_dirs):
        # Extend with default names if needed
        experiment_names.extend(
            [os.path.basename(d) for d in experiment_dirs[len(experiment_names) :]]
        )
    elif len(experiment_names) > len(experiment_dirs):
        # Truncate if there are more names than directories
        experiment_names = experiment_names[: len(experiment_dirs)]

    # Data structure to hold results for each method and metric
    results_data = {metric: {"means": [], "errors": []} for metric in metrics_to_plot}

    # Process each experiment directory
    for dir_index, experiment_dir in enumerate(experiment_dirs):
        # Get seeds from the directory
        seeds = []
        if os.path.exists(os.path.join(experiment_dir, "results")):
            # This is already a seed directory
            seeds = [experiment_dir]
        else:
            # This is a parent directory containing seed directories
            seeds = [
                os.path.join(experiment_dir, seed)
                for seed in os.listdir(experiment_dir)
                if (
                    os.path.isdir(os.path.join(experiment_dir, seed))
                    and os.path.exists(os.path.join(experiment_dir, seed, "results"))
                )
            ]

        if len(seeds) == 0:
            print(f"Warning: No seed directories found in {experiment_dir}")
            continue

        # Get results from all seeds
        results_files = [extract_latest_result_from_dir(seed) for seed in seeds]
        results = [
            get_exp_results_from_json(result_file)
            for result_file in results_files
            if result_file is not None
        ]

        if len(results) == 0:
            print(f"Warning: No valid results found in {experiment_dir}")
            continue

        # Get the epoch if not specified
        if epoch is None:
            epoch = len(results[0].truth_losses) - 1
        elif epoch >= len(results[0].truth_losses):
            print(
                f"Warning: Epoch {epoch} is out of range for {experiment_dir}. Using max epoch."
            )
            epoch = len(results[0].truth_losses) - 1

        # Calculate loss metrics
        metrics = {
            "truth_loss": lambda r: r.truth_losses[epoch],
            "collateral_loss": lambda r: r.collateral_losses[epoch]
            if hasattr(r, "collateral_losses") and len(r.collateral_losses) > epoch
            else None,
            "proxy_loss": lambda r: r.proxy_losses[epoch]
            if hasattr(r, "proxy_losses") and len(r.proxy_losses) > epoch
            else None,
            "outcome_loss": lambda r: r.outcome_losses[epoch]
            if hasattr(r, "outcome_losses") and len(r.outcome_losses) > epoch
            else None,
        }

        for metric_name, metric_func in metrics.items():
            if metric_name in metrics_to_plot:
                try:
                    values = [
                        metric_func(r) for r in results if metric_func(r) is not None
                    ]

                    if values:
                        mean_value = np.mean(values)
                        std_error = 1.96 * np.std(values) / np.sqrt(len(values))

                        results_data[metric_name]["means"].append(mean_value)
                        results_data[metric_name]["errors"].append(std_error)
                    else:
                        # If no valid values, use placeholder values
                        print(
                            f"Warning: No valid {metric_name} values for {experiment_dir}"
                        )
                        results_data[metric_name]["means"].append(0)
                        results_data[metric_name]["errors"].append(0)
                except (AttributeError, IndexError) as e:
                    # If metric is not available, use placeholder values
                    print(
                        f"Warning: Could not calculate {metric_name} for {experiment_dir}: {e}"
                    )
                    results_data[metric_name]["means"].append(0)
                    results_data[metric_name]["errors"].append(0)

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Set width of bars and positions
    num_methods = len(experiment_dirs)
    num_metrics = len(metrics_to_plot)
    total_width = 0.8  # Total width for all bars at one method position
    bar_width = total_width / num_metrics  # Width of each bar

    # Set positions for the bars
    indices = np.arange(num_methods)

    # Plot bars for each metric
    for i, metric in enumerate(metrics_to_plot):
        # Calculate offset for this metric
        offset = (i - num_metrics / 2 + 0.5) * bar_width
        positions = indices + offset

        # Plot the bars
        plt.bar(
            positions,
            results_data[metric]["means"],
            bar_width,
            yerr=results_data[metric]["errors"],
            color=metric_styles[metric]["color"],
            label=metric_styles[metric]["label"],
            capsize=5,
        )

    # Add labels, title, and legend
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(indices, experiment_names, rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.7, axis="y")

    # Add legend
    plt.legend(fontsize=10, loc="best")

    # Determine plot title and filename
    title = f"Loss Comparison - {experiment_name} - Epoch {epoch}"
    plot_basename = f"loss_bar_{experiment_name}_epoch_{epoch}"

    # If plot_name is provided, prepend it to the title and filename
    if plot_name:
        title = f"{plot_name} {title}"
        plot_basename = f"{plot_name}_{plot_basename}"

    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save the plot - create plots directory if it doesn't exist
    plots_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(experiment_dirs[0]) if experiment_dirs else "."
        ),
        "plots",
    )
    os.makedirs(plots_dir, exist_ok=True)

    # Save using the save_plot function
    save_plot(
        None,
        plots_dir,
        plot_basename,
        is_multiseed=False,
    )

    print(f"Loss bar plot saved to {os.path.join(plots_dir, plot_basename + '.png')}")


if __name__ == "__main__":
    # Create argument parser with description
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot accuracy and loss comparison across multiple proxy coverage methods"
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="Paths to directories containing experiment results (multiple can be specified)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to plot for (default: last epoch)",
    )
    parser.add_argument(
        "--experiment_name",
        help="Name of the experiment comparison",
        default="Comparison",
    )
    parser.add_argument(
        "--experiment_names",
        nargs="+",
        help="Names to use for each method in the legend (must match number of dirs)",
        default=None,
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        help="Prepend this name to plot titles and filenames",
        default=None,
    )

    # Plot type arguments
    parser.add_argument(
        "--plot_accuracy",
        action="store_true",
        help="Plot accuracy comparison (default: both accuracy and loss are generated)",
    )
    parser.add_argument(
        "--plot_loss",
        action="store_true",
        help="Plot loss comparison (default: both accuracy and loss are generated)",
    )
    parser.add_argument(
        "--plot_pareto",
        action="store_true",
        help="Plot pareto frontier (not generated by default)",
    )
    parser.add_argument(
        "--bar_plots",
        action="store_true",
        help="Generate bar plots instead of proxy coverage plots",
    )

    # Metric inclusion arguments
    parser.add_argument(
        "--include_truth",
        action="store_true",
        help="Include truth trigger results in the plot (default: True for accuracy)",
    )
    parser.add_argument(
        "--include_collateral",
        action="store_true",
        help="Include collateral trigger results in the plot",
    )
    parser.add_argument(
        "--include_proxy",
        action="store_true",
        help="Include proxy trigger results in the plot",
    )
    parser.add_argument(
        "--include_outcome",
        action="store_true",
        help="Include outcome trigger results in the plot (default: True for accuracy)",
    )
    parser.add_argument(
        "--include_truth_no_proxy",
        action="store_true",
        help="Include truth_no_proxy trigger results in the plot (accuracy only)",
    )
    parser.add_argument(
        "--collateral_pareto",
        action="store_true",
        help="Plot collateral pareto frontier instead of outcome",
    )

    args = parser.parse_args()

    # Process directories to ensure they're valid paths
    processed_dirs = []
    for dir_path in args.experiment_dirs:
        if not dir_path.startswith("experiments/"):
            dir_path = os.path.join("experiments", dir_path)
        processed_dirs.append(dir_path)

    # Determine which plots to generate
    plot_both = not args.plot_accuracy and not args.plot_loss

    # Set defaults for what to include
    include_truth = args.include_truth or not (
        args.include_collateral
        or args.include_proxy
        or args.include_outcome
        or args.include_truth_no_proxy
    )
    include_outcome = args.include_outcome or not (
        args.include_truth
        or args.include_collateral
        or args.include_proxy
        or args.include_truth_no_proxy
    )

    # Generate appropriate plots based on args
    if args.bar_plots:
        # Generate bar plots
        if args.plot_accuracy or plot_both:
            plot_accuracy_bar(
                processed_dirs,
                args.epoch,
                args.experiment_name,
                include_truth,
                args.include_collateral,
                args.include_proxy,
                include_outcome,
                args.include_truth_no_proxy,
                args.experiment_names,
                args.plot_name,
            )

        if args.plot_loss or plot_both:
            plot_loss_bar(
                processed_dirs,
                args.epoch,
                args.experiment_name,
                include_truth,
                args.include_collateral,
                args.include_proxy,
                include_outcome,
                args.experiment_names,
                args.plot_name,
            )
    else:
        # Generate proxy coverage plots
        if args.plot_accuracy or plot_both:
            plot_accuracy_across_methods(
                processed_dirs,
                args.epoch,
                args.experiment_name,
                include_truth,
                args.include_collateral,
                args.include_proxy,
                include_outcome,
                args.include_truth_no_proxy,
                args.experiment_names,
                args.plot_name,
            )

        if args.plot_loss or plot_both:
            plot_loss_across_methods(
                processed_dirs,
                args.epoch,
                args.experiment_name,
                include_truth,
                args.include_collateral,
                args.include_proxy,
                include_outcome,
                args.experiment_names,
                args.plot_name,
            )

    # Only generate pareto frontier if explicitly requested
    if args.plot_pareto:
        plot_pareto_frontier(
            processed_dirs,
            args.epoch,
            args.experiment_name,
            comparison_metric="outcome" if not args.collateral_pareto else "collateral",
            experiment_names=args.experiment_names,
            plot_name=args.plot_name,
        )
