import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle


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
    import json

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


def apply_logit_transform(data):
    """
    Apply logit transformation: log(x/(1-x)) to data points.
    Handles edge cases by clipping values to avoid log(0) or log(inf).

    Args:
        data: Numpy array or float to transform

    Returns:
        Transformed data
    """
    # Clip values to avoid division by zero or log(0)
    epsilon = 1e-10
    clipped_data = np.clip(data, epsilon, 1.0 - epsilon)

    # Apply logit transformation
    return np.log(clipped_data / (1.0 - clipped_data))


# Modify the save_plot function to ensure proper figure dimensions


def save_plot(
    _,
    plots_dir,
    plot_basename,
    is_multiseed=False,
    use_dirs_as_labels=False,
    log_scale=False,
    logit_scale=False,
):
    """
    Save the current plot to a file with proper dimensions.

    Args:
        _: Unused parameter (for compatibility)
        plots_dir: Directory to save the plot
        plot_basename: Base name for the plot file
        is_multiseed: Whether to add '_multiseed' suffix to the filename
        use_dirs_as_labels: Whether to add '_dirs_as_labels' suffix to the filename
        log_scale: Whether to add '_log_scale' suffix to the filename
        logit_scale: Whether to add '_logit_scale' suffix to the filename
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
    if use_dirs_as_labels:
        filename += "_dirs_as_labels"
    if log_scale:
        filename += "_log_scale"
    if logit_scale:
        filename += "_logit_scale"
    filename += ".png"

    # Save the plot with higher resolution
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def get_proxy_strategy_from_dir(dir_name):
    """
    Extract proxy strategy from directory name.

    Args:
        dir_name: Directory name to analyze

    Returns:
        Extracted proxy strategy name
    """
    if dir_name.startswith("proxy_strategy_"):
        return dir_name[len("proxy_strategy_") :].split("_")[0]

    # Try to infer strategy from the directory name
    for strategy in [
        "naive",
        "steering_weights",
        "goal_misgeneralization",
        "adversarial",
    ]:
        if strategy in dir_name:
            return strategy

    # Default case
    return "unknown"


def is_peft_from_dir(dir_name):
    """
    Determine if the directory uses PEFT from its name.

    Args:
        dir_name: Directory name to analyze

    Returns:
        Boolean indicating if PEFT is used
    """
    return "peft" in dir_name.lower()


def plot_pareto_frontiers(
    central_dir,
    epoch=None,
    experiment_name="Pareto_Comparison",
    plot_truth_outcome=True,
    plot_collateral_truth=True,
    plot_outcome_proxy=True,
    use_dirs_as_labels=False,
    baseline_dirs=None,
    included_strategies=None,
    no_variance=False,
    connect_points=False,
    log_scale=False,
    logit_scale=False,
):
    """
    Plot various Pareto frontiers for experiment results from a central directory.

    Args:
        central_dir: Central directory containing experiment folders
        epoch: Epoch to extract accuracy metrics from (default: last epoch)
        experiment_name: Name of the experiment for plot titles
        plot_truth_outcome: Whether to plot Alignment-test vs Task-test frontier
        plot_collateral_truth: Whether to plot Collateral vs Alignment-test frontier
        plot_outcome_proxy: Whether to plot Task-test vs Alignment-train frontier
        use_dirs_as_labels: Whether to use directory names as labels instead of grouping by strategy
        baseline_dirs: List of directories to mark as baselines (as separate paths)
        included_strategies: List of strategies to include (filter out others)
        no_variance: Whether to suppress plotting of error bars
        connect_points: Whether to connect points with the same label using lines
        log_scale: Whether to use logarithmic scale for the axes
        logit_scale: Whether to use logit scale [log(x/(1-x))] for the axes
    """
    # Cannot use both log_scale and logit_scale simultaneously
    if log_scale and logit_scale:
        print("Warning: Both log_scale and logit_scale specified. Using logit_scale.")
        log_scale = False

    # Define different colors for different experiment methods
    method_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # teal
    ]

    # Define different markers for clarity
    markers = ["o", "s", "^", "D", "X", "P", "*", "h", "p", "v"]

    # Set baseline markers and color
    baseline_marker = "x"
    baseline_color = "black"

    # Initialize baseline_dirs if None
    if baseline_dirs is None:
        baseline_dirs = []
    # Convert to list if string
    elif isinstance(baseline_dirs, str):
        baseline_dirs = [baseline_dirs]

    # Print baseline directories being processed
    print(f"Baseline directories: {baseline_dirs}")

    # Dictionary to store results for all experiments
    raw_exp_results = {}
    proxy_coverage = None  # Store for title
    is_multiseed = False  # Flag for multiseed

    # Map to store color/marker for each strategy
    strategy_colors = {}
    strategy_markers = {}
    strategy_idx = 0

    # Process each experiment directory in the central directory
    experiment_dirs = [
        os.path.join(central_dir, d)
        for d in os.listdir(central_dir)
        if os.path.isdir(os.path.join(central_dir, d))
    ]

    # First, process baseline directories
    for baseline_dir in baseline_dirs:
        if not os.path.isdir(baseline_dir):
            print(f"Warning: Baseline directory {baseline_dir} not found, skipping")
            continue

        baseline_name = os.path.basename(baseline_dir)
        print(f"Processing baseline: {baseline_name}")

        # Check if this is a multiseed directory
        baseline_is_multiseed = False
        seeds = []

        # Look for subdirectories with "seed" in the name
        subdirs = [
            os.path.join(baseline_dir, d)
            for d in os.listdir(baseline_dir)
            if os.path.isdir(os.path.join(baseline_dir, d))
        ]

        for subdir in subdirs:
            if "seed" in os.path.basename(subdir).lower():
                baseline_is_multiseed = True
                seeds.append(subdir)
                is_multiseed |= True  # Update global multiseed flag

        # If not multiseed, treat the baseline directory as a single seed
        if not baseline_is_multiseed:
            seeds = [baseline_dir]

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
            print(
                f"  Warning: No valid results found for baseline {baseline_name}, skipping"
            )
            continue

        # Get the epoch if not specified
        if epoch is None:
            epoch = len(all_results[0].truth_losses) - 1
        elif epoch >= len(all_results[0].truth_losses):
            print(
                f"  Warning: Epoch {epoch} is out of range for baseline. Using max epoch."
            )
            epoch = len(all_results[0].truth_losses) - 1

        # Calculate metrics for this baseline
        try:
            # Extract PEFT status from experiment config
            is_peft = all_results[0].experiment_config["finetune_config"]["is_peft"]

            # Alignment-test accuracy (1 - response percent)
            truth_acc_values = [
                1 - r.truth_trigger_response_percent[epoch] for r in all_results
            ]
            truth_acc_mean = np.mean(truth_acc_values)
            truth_acc_std = np.std(truth_acc_values)

            # Task-test accuracy (response percent)
            outcome_acc_values = [
                r.outcome_trigger_response_percent[epoch] for r in all_results
            ]
            outcome_acc_mean = np.mean(outcome_acc_values)
            outcome_acc_std = np.std(outcome_acc_values)

            # Check if collateral accuracy is available
            has_collateral = hasattr(
                all_results[0], "collateral_trigger_response_percent"
            )

            if has_collateral:
                collateral_acc_values = [
                    r.collateral_trigger_response_percent[epoch] for r in all_results
                ]
                collateral_acc_mean = np.mean(collateral_acc_values)
                collateral_acc_std = np.std(collateral_acc_values)
            else:
                collateral_acc_mean = None
                collateral_acc_std = None

            # Get proxy coverage percentage if available (for title)
            if (
                hasattr(all_results[0], "experiment_config")
                and "proxy_trigger_coverage" in all_results[0].experiment_config
            ):
                baseline_proxy_coverage = (
                    all_results[0].experiment_config["proxy_trigger_coverage"] * 100
                )
                # Only update global proxy_coverage if not set yet
                if proxy_coverage is None:
                    proxy_coverage = baseline_proxy_coverage

            # Determine label based on use_dirs_as_labels flag
            if use_dirs_as_labels:
                # Use original dir name as label
                label = baseline_name
            else:
                # Create specific baseline label based on PEFT status
                label = f"baseline_task{'_peft' if is_peft else ''}"

            # Set color and marker for baselines
            if label not in strategy_colors:
                strategy_colors[label] = baseline_color
                strategy_markers[label] = baseline_marker

            # Store the calculated metrics for this baseline
            # Note: We're not trying to access proxy metrics for baselines
            raw_exp_results[baseline_name] = {
                "label": label,
                "is_baseline": True,
                "color": strategy_colors[label],
                "marker": strategy_markers[label],
                "truth_acc": {"mean": truth_acc_mean, "std": truth_acc_std},
                "proxy_acc": {
                    "mean": None,
                    "std": None,
                },  # Set proxy metrics to None for baselines
                "outcome_acc": {"mean": outcome_acc_mean, "std": outcome_acc_std},
                "collateral_acc": {
                    "mean": collateral_acc_mean,
                    "std": collateral_acc_std,
                },
                "proxy_coverage": baseline_proxy_coverage
                if "baseline_proxy_coverage" in locals()
                else None,
                "num_seeds": len(all_results),
            }

            print(f"  Processed {len(all_results)} seeds for baseline {baseline_name}")

        except Exception as e:
            print(f"  Error processing baseline {baseline_name}: {str(e)}")
            continue

    # Now process regular experiment directories
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        original_exp_name = exp_name  # Keep original for filtering

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
                is_multiseed |= True  # Update global multiseed flag

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

        # Get the epoch if not specified
        if epoch is None:
            epoch = len(all_results[0].truth_losses) - 1
        elif epoch >= len(all_results[0].truth_losses):
            print(f"  Warning: Epoch {epoch} is out of range. Using max epoch.")
            epoch = len(all_results[0].truth_losses) - 1

        # Calculate metrics for this experiment
        try:
            # Extract strategy and peft status from experiment config
            proxy_strategy = all_results[0].experiment_config["proxy_strategy"]
            is_peft = all_results[0].experiment_config["finetune_config"]["is_peft"]

            # Skip this directory if it's not in included_strategies (when filter is active)
            if included_strategies and proxy_strategy not in included_strategies:
                print(
                    f"Skipping {exp_name} - strategy {proxy_strategy} not in included_strategies"
                )
                continue

            # Alignment-test accuracy (1 - response percent)
            truth_acc_values = [
                1 - r.truth_trigger_response_percent[epoch] for r in all_results
            ]
            truth_acc_mean = np.mean(truth_acc_values)
            truth_acc_std = np.std(truth_acc_values)

            # Alignment-train accuracy (1 - response percent)
            proxy_acc_values = [
                1 - r.proxy_trigger_response_percent[epoch] for r in all_results
            ]
            proxy_acc_mean = np.mean(proxy_acc_values)
            proxy_acc_std = np.std(proxy_acc_values)

            # Task-test accuracy (response percent)
            outcome_acc_values = [
                r.outcome_trigger_response_percent[epoch] for r in all_results
            ]
            outcome_acc_mean = np.mean(outcome_acc_values)
            outcome_acc_std = np.std(outcome_acc_values)

            # Check if collateral accuracy is available
            has_collateral = hasattr(
                all_results[0], "collateral_trigger_response_percent"
            )

            if has_collateral:
                collateral_acc_values = [
                    r.collateral_trigger_response_percent[epoch] for r in all_results
                ]
                collateral_acc_mean = np.mean(collateral_acc_values)
                collateral_acc_std = np.std(collateral_acc_values)
            else:
                collateral_acc_mean = None
                collateral_acc_std = None

            # Get proxy coverage percentage if available (for title)
            if (
                hasattr(all_results[0], "experiment_config")
                and "proxy_trigger_coverage" in all_results[0].experiment_config
            ):
                exp_proxy_coverage = (
                    all_results[0].experiment_config["proxy_trigger_coverage"] * 100
                )
                # Only update global proxy_coverage if not set yet
                if proxy_coverage is None:
                    proxy_coverage = exp_proxy_coverage

            # Determine label based on use_dirs_as_labels flag
            if use_dirs_as_labels:
                # Use original dir name as label but cut out "proxy_strategy" from the start
                label = original_exp_name.split("proxy_strategy_")[1]
            else:
                # Construct the label - each experiment will have its own point but same label
                label = f"{proxy_strategy}{'_peft' if is_peft else ''}"

            # Ensure all points with the same strategy+peft combo have same color/marker
            if label not in strategy_colors:
                strategy_colors[label] = method_colors[
                    strategy_idx % len(method_colors)
                ]
                strategy_markers[label] = markers[strategy_idx % len(markers)]
                strategy_idx += 1

            # Store the calculated metrics - each experiment directory gets its own entry
            raw_exp_results[original_exp_name] = {
                "label": label,
                "is_baseline": False,
                "color": strategy_colors[label],
                "marker": strategy_markers[label],
                "truth_acc": {"mean": truth_acc_mean, "std": truth_acc_std},
                "proxy_acc": {"mean": proxy_acc_mean, "std": proxy_acc_std},
                "outcome_acc": {"mean": outcome_acc_mean, "std": outcome_acc_std},
                "collateral_acc": {
                    "mean": collateral_acc_mean,
                    "std": collateral_acc_std,
                },
                "proxy_coverage": exp_proxy_coverage
                if "exp_proxy_coverage" in locals()
                else None,
                "num_seeds": len(all_results),
            }

            print(f"  Processed {len(all_results)} seeds for {exp_name}")

        except Exception as e:
            print(f"  Error processing {exp_name}: {str(e)}")
            continue

    # Create plots directory
    plots_dir = os.path.join(central_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot Task-test vs Alignment-test frontier (Alignment-test on Y-axis)
    if plot_truth_outcome:
        plt.figure(figsize=(10, 6))

        # Organize points by label for connecting them if requested
        points_by_label = {}
        for exp_name, results in raw_exp_results.items():
            label = results["label"]
            if label not in points_by_label:
                points_by_label[label] = []

            # Apply logit transform if requested
            x_value = results["outcome_acc"]["mean"]
            y_value = results["truth_acc"]["mean"]
            x_std = results["outcome_acc"]["std"]
            y_std = results["truth_acc"]["std"]

            if logit_scale:
                x_value = apply_logit_transform(x_value)
                y_value = apply_logit_transform(y_value)

                # Transform standard deviations using approximation
                # This is a simplified approximation for small std values
                if not no_variance and x_std is not None and y_std is not None:
                    # For logit transform, derivative is 1/(x(1-x))
                    x_std = x_std / (
                        results["outcome_acc"]["mean"]
                        * (1 - results["outcome_acc"]["mean"])
                    )
                    y_std = y_std / (
                        results["truth_acc"]["mean"]
                        * (1 - results["truth_acc"]["mean"])
                    )

            points_by_label[label].append(
                {
                    "x": x_value,
                    "y": y_value,
                    "x_std": x_std,
                    "y_std": y_std,
                    "color": results["color"],
                    "marker": results["marker"],
                    "exp_name": exp_name,
                    "is_baseline": results["is_baseline"],
                }
            )

        # Plot each group of points
        for label, points in points_by_label.items():
            # Sort points by x-value for clean line connections
            if connect_points and not any(p["is_baseline"] for p in points):
                # Only sort non-baseline points for connection
                points.sort(key=lambda p: p["x"])

            # Extract x and y coordinates for potential line
            x_values = [p["x"] for p in points]
            y_values = [p["y"] for p in points]

            # Plot each point
            for i, point in enumerate(points):
                # Only add to legend once per label
                plot_label = label if i == 0 else None

                # Plot with or without error bars
                if no_variance:
                    plt.plot(
                        point["x"],
                        point["y"],
                        marker=point["marker"],
                        color=point["color"],
                        label=plot_label,
                        linestyle=""
                        if not connect_points
                        else None,  # Empty for just points
                    )
                else:
                    plt.errorbar(
                        point["x"],
                        point["y"],
                        xerr=point["x_std"],
                        yerr=point["y_std"],
                        fmt=point["marker"],
                        color=point["color"],
                        label=plot_label,
                        capsize=5,
                        linestyle=""
                        if not connect_points
                        else None,  # Empty for just points
                    )

            # Connect points if requested (but never connect baseline points)
            if (
                connect_points
                and len(points) > 1
                and not any(p["is_baseline"] for p in points)
            ):
                plt.plot(
                    x_values,
                    y_values,
                    color=points[0]["color"],
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.7,
                )

        plt.xlabel("Task-test Accuracy" + (" (logit scale)" if logit_scale else ""))
        plt.ylabel(
            "Alignment-test Accuracy" + (" (logit scale)" if logit_scale else "")
        )

        # Apply log scale if requested
        if log_scale:
            plt.xscale("log")
            plt.yscale("log")

        # Use proxy coverage in title if available
        if proxy_coverage is not None:
            title = f"Pareto Frontier of Various Methods at Alignment-test Trigger Coverage = {proxy_coverage:.1f}%"
        else:
            title = f"{experiment_name} - Task-test vs Alignment-test"

        plt.title(title)

        # Place legend outside of the plot area to avoid overlap
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.grid(True)
        plt.tight_layout()  # Adjust layout to make room for the legend

        # Add no_variance to filename if flag is set
        suffix = "_no_variance" if no_variance else ""
        suffix += "_connected" if connect_points else ""
        save_plot(
            None,
            plots_dir,
            f"task_test_vs_alignment_test_frontier{suffix}",
            is_multiseed=is_multiseed,
            use_dirs_as_labels=use_dirs_as_labels,
            log_scale=log_scale,
            logit_scale=logit_scale,
        )

    # Plot Collateral vs Alignment-test frontier - exclude baselines
    if plot_collateral_truth:
        # Skip if no experiments have collateral data
        has_any_collateral = any(
            results["collateral_acc"]["mean"] is not None and not results["is_baseline"]
            for results in raw_exp_results.values()
        )

        if has_any_collateral:
            plt.figure(figsize=(10, 6))

            # Organize points by label for connecting them if requested
            points_by_label = {}
            for exp_name, results in raw_exp_results.items():
                # Skip baselines for this plot
                if results["is_baseline"]:
                    continue

                if results["collateral_acc"]["mean"] is not None:
                    label = results["label"]
                    if label not in points_by_label:
                        points_by_label[label] = []

                    # Apply logit transform if requested
                    x_value = results["collateral_acc"]["mean"]
                    y_value = results["truth_acc"]["mean"]
                    x_std = results["collateral_acc"]["std"]
                    y_std = results["truth_acc"]["std"]

                    if logit_scale:
                        x_value = apply_logit_transform(x_value)
                        y_value = apply_logit_transform(y_value)

                        # Transform standard deviations using approximation
                        if not no_variance and x_std is not None and y_std is not None:
                            # For logit transform, derivative is 1/(x(1-x))
                            x_std = x_std / (
                                results["collateral_acc"]["mean"]
                                * (1 - results["collateral_acc"]["mean"])
                            )
                            y_std = y_std / (
                                results["truth_acc"]["mean"]
                                * (1 - results["truth_acc"]["mean"])
                            )

                    points_by_label[label].append(
                        {
                            "x": x_value,
                            "y": y_value,
                            "x_std": x_std,
                            "y_std": y_std,
                            "color": results["color"],
                            "marker": results["marker"],
                            "exp_name": exp_name,
                        }
                    )

            # Plot each group of points
            for label, points in points_by_label.items():
                # Sort points by x-value for clean line connections
                if connect_points:
                    points.sort(key=lambda p: p["x"])

                # Extract x and y coordinates for potential line
                x_values = [p["x"] for p in points]
                y_values = [p["y"] for p in points]

                # Plot each point
                for i, point in enumerate(points):
                    # Only add to legend once per label
                    plot_label = label if i == 0 else None

                    # Plot with or without error bars
                    if no_variance:
                        plt.plot(
                            point["x"],
                            point["y"],
                            marker=point["marker"],
                            color=point["color"],
                            label=plot_label,
                            linestyle=""
                            if not connect_points
                            else None,  # Empty for just points
                        )
                    else:
                        plt.errorbar(
                            point["x"],
                            point["y"],
                            xerr=point["x_std"],
                            yerr=point["y_std"],
                            fmt=point["marker"],
                            color=point["color"],
                            label=plot_label,
                            capsize=5,
                            linestyle=""
                            if not connect_points
                            else None,  # Empty for just points
                        )

                # Connect points if requested
                if connect_points and len(points) > 1:
                    plt.plot(
                        x_values,
                        y_values,
                        color=points[0]["color"],
                        linestyle="-",
                        linewidth=1.5,
                        alpha=0.7,
                    )

            plt.xlabel(
                "Collateral Accuracy" + (" (logit scale)" if logit_scale else "")
            )
            plt.ylabel(
                "Alignment-test Accuracy" + (" (logit scale)" if logit_scale else "")
            )

            # Apply log scale if requested
            if log_scale:
                plt.xscale("log")
                plt.yscale("log")

            # Use proxy coverage in title if available
            if proxy_coverage is not None:
                title = f"Pareto Frontier of Various Methods at Alignment-test Trigger Coverage = {proxy_coverage:.1f}%"
            else:
                title = f"{experiment_name} - Collateral vs Alignment-test"

            plt.title(title)

            # Place legend outside of the plot area to avoid overlap
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.grid(True)
            plt.tight_layout()  # Adjust layout to make room for the legend

            # Add no_variance to filename if flag is set
            suffix = "_no_variance" if no_variance else ""
            suffix += "_connected" if connect_points else ""
            save_plot(
                None,
                plots_dir,
                f"collateral_vs_alignment_test_frontier{suffix}",
                is_multiseed=is_multiseed,
                use_dirs_as_labels=use_dirs_as_labels,
                log_scale=log_scale,
                logit_scale=logit_scale,
            )
        else:
            print(
                "Warning: No collateral data available, skipping collateral vs alignment-test plot"
            )

    # Plot Task-test vs Alignment-train frontier - exclude baselines
    if plot_outcome_proxy:
        plt.figure(figsize=(10, 6))

        # Organize points by label for connecting them if requested
        points_by_label = {}
        for exp_name, results in raw_exp_results.items():
            # Skip baselines for this plot
            if results["is_baseline"]:
                continue

            # Skip if proxy metrics are None
            if results["proxy_acc"]["mean"] is None:
                continue

            label = results["label"]
            if label not in points_by_label:
                points_by_label[label] = []

            # Apply logit transform if requested
            # Apply logit transform if requested
            x_value = results["outcome_acc"]["mean"]
            y_value = results["proxy_acc"]["mean"]
            x_std = results["outcome_acc"]["std"]
            y_std = results["proxy_acc"]["std"]

            if logit_scale:
                x_value = apply_logit_transform(x_value)
                y_value = apply_logit_transform(y_value)

                # Transform standard deviations using approximation
                if not no_variance and x_std is not None and y_std is not None:
                    # For logit transform, derivative is 1/(x(1-x))
                    x_std = x_std / (
                        results["outcome_acc"]["mean"]
                        * (1 - results["outcome_acc"]["mean"])
                    )
                    y_std = y_std / (
                        results["proxy_acc"]["mean"]
                        * (1 - results["proxy_acc"]["mean"])
                    )

            points_by_label[label].append(
                {
                    "x": x_value,
                    "y": y_value,
                    "x_std": x_std,
                    "y_std": y_std,
                    "color": results["color"],
                    "marker": results["marker"],
                    "exp_name": exp_name,
                }
            )

        # Plot each group of points
        for label, points in points_by_label.items():
            # Sort points by x-value for clean line connections
            if connect_points:
                points.sort(key=lambda p: p["x"])

            # Extract x and y coordinates for potential line
            x_values = [p["x"] for p in points]
            y_values = [p["y"] for p in points]

            # Plot each point
            for i, point in enumerate(points):
                # Only add to legend once per label
                plot_label = label if i == 0 else None

                # Plot with or without error bars
                if no_variance:
                    plt.plot(
                        point["x"],
                        point["y"],
                        marker=point["marker"],
                        color=point["color"],
                        label=plot_label,
                        linestyle=""
                        if not connect_points
                        else None,  # Empty for just points
                    )
                else:
                    plt.errorbar(
                        point["x"],
                        point["y"],
                        xerr=point["x_std"],
                        yerr=point["y_std"],
                        fmt=point["marker"],
                        color=point["color"],
                        label=plot_label,
                        capsize=5,
                        linestyle=""
                        if not connect_points
                        else None,  # Empty for just points
                    )

            # Connect points if requested
            if connect_points and len(points) > 1:
                plt.plot(
                    x_values,
                    y_values,
                    color=points[0]["color"],
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.7,
                )

        plt.xlabel("Task-test Accuracy" + (" (logit scale)" if logit_scale else ""))
        plt.ylabel(
            "Alignment-train Accuracy" + (" (logit scale)" if logit_scale else "")
        )

        # Apply log scale if requested
        if log_scale:
            plt.xscale("log")
            plt.yscale("log")

        # Use proxy coverage in title if available
        if proxy_coverage is not None:
            title = f"Pareto Frontier of Various Methods at Alignment-test Trigger Coverage = {proxy_coverage:.1f}%"
        else:
            title = f"{experiment_name} - Task-test vs Alignment-train"

        plt.title(title)

        # Place legend outside of the plot area to avoid overlap
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.grid(True)
        plt.tight_layout()  # Adjust layout to make room for the legend

        # Add no_variance to filename if flag is set
        suffix = "_no_variance" if no_variance else ""
        suffix += "_connected" if connect_points else ""
        save_plot(
            None,
            plots_dir,
            f"task_test_vs_alignment_train_frontier{suffix}",
            is_multiseed=is_multiseed,
            use_dirs_as_labels=use_dirs_as_labels,
            log_scale=log_scale,
            logit_scale=logit_scale,
        )


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Plot Pareto frontiers for experiment results"
    )

    parser.add_argument(
        "central_dir",
        help="Path to the central directory containing experiment folders",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to extract accuracy metrics from (default: last epoch)",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Pareto_Comparison",
        help="Name of the experiment for plot titles",
    )

    parser.add_argument(
        "--no_truth_outcome",
        action="store_false",
        dest="plot_truth_outcome",
        help="Disable Alignment-test vs Task-test Pareto frontier plot",
    )

    parser.add_argument(
        "--no_collateral_truth",
        action="store_false",
        dest="plot_collateral_truth",
        help="Disable Collateral vs Alignment-test Pareto frontier plot",
    )

    parser.add_argument(
        "--no_outcome_proxy",
        action="store_false",
        dest="plot_outcome_proxy",
        help="Disable Task-test vs Alignment-train Pareto frontier plot",
    )

    parser.add_argument(
        "--use_dirs_as_labels",
        action="store_true",
        help="Use directory names as labels instead of grouping by strategy",
    )

    parser.add_argument(
        "--no_variance",
        action="store_true",
        help="Do not plot error bars, only show mean values",
    )

    parser.add_argument(
        "--connect_points",
        action="store_true",
        help="Connect points with the same strategy label using lines",
    )

    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Use logarithmic scale for the axes",
    )

    parser.add_argument(
        "--logit_scale",
        action="store_true",
        help="Use logit scale [log(x/(1-x))] for the axes",
    )

    parser.add_argument(
        "--baseline_dirs",
        nargs="+",
        default=[],
        help="List of directories to mark as baselines (as separate paths)",
    )

    parser.add_argument(
        "--included_strategies",
        nargs="+",
        default=None,
        help="List of strategies to include (others will be filtered out)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Process the central directory
    central_dir = args.central_dir
    if not os.path.isdir(central_dir):
        print(f"Error: {central_dir} is not a valid directory")
        return

    # Plot Pareto frontiers
    plot_pareto_frontiers(
        central_dir,
        args.epoch,
        args.experiment_name,
        args.plot_truth_outcome,
        args.plot_collateral_truth,
        args.plot_outcome_proxy,
        args.use_dirs_as_labels,
        args.baseline_dirs,
        args.included_strategies,
        args.no_variance,
        args.connect_points,
        args.log_scale,
        args.logit_scale,
    )


if __name__ == "__main__":
    main()
