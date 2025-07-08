import os

import matplotlib.pyplot as plt
import numpy as np
from validate import get_exp_results_from_json
from experiment_utils import extract_latest_result_from_dir


def save_plot(results, exp_folder, plot_name, is_multiseed=False):
    assert os.path.exists(exp_folder)
    plot_path = (
        f"{os.path.join(exp_folder, plot_name)}.png"
        if not is_multiseed
        else f"{os.path.join(exp_folder, plot_name)}_multiseed.png"
    )
    print(f"Saving plot to {plot_path}")
    plt.savefig(plot_path, dpi=300)


def plot_losses(
    results,
    results_folder,
    truth_no_proxy_triggers=False,
    collateral=False,
    is_multiseed=False,
):
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    plt.figure(figsize=(10, 6))

    # Handle single result vs list of results
    if not is_multiseed:
        results_list = [results]
    else:
        results_list = results

    # Determine the maximum length across all results
    max_length = max([len(r.outcome_losses) for r in results_list])
    x = np.asarray(list(range(0, max_length)))

    # Create x points for linear interpolation
    X_smooth = np.linspace(x.min(), x.max(), 300)

    # Define datasets to plot (attribute name, label, color)
    dataset_specs = [
        ("truth_losses", "Alignment-test", "blue"),
        ("outcome_losses", "Task-test", "red"),
    ]

    # Add collateral losses if specified and they exist
    if collateral:
        if all(
            hasattr(r, "collateral_losses") and r.collateral_losses is not None
            for r in results_list
        ):
            dataset_specs.append(("collateral_losses", "Collateral", "magenta"))

    # Add proxy losses if they exist
    if all(r.proxy_losses for r in results_list):
        dataset_specs.append(("proxy_losses", "Alignment-train", "green"))

    # Add truth without proxy trigger overlaps if requested
    if truth_no_proxy_triggers and all(
        [
            getattr(result, "truth_no_proxy_labels_trigger_response_percent")
            is not None
            for result in results_list
        ]
    ):
        dataset_specs.append(
            (
                "truth_no_proxy_labels_losses",
                "Alignment-test (no Alignment-train trigger overlap)",
                "purple",
            )
        )

    # For each dataset, calculate mean, confidence intervals, and plot
    for attr_name, label, color in dataset_specs:
        # Extract data from all results, padding shorter sequences if needed
        all_data = []
        for result in results_list:
            data = getattr(result, attr_name)[:]
            # Pad with NaN if shorter than max_length
            if len(data) < max_length:
                data = np.pad(
                    data,
                    (0, max_length - len(data)),
                    "constant",
                    constant_values=np.nan,
                )
            all_data.append(data)

        # Convert to numpy array for calculations
        all_data_array = np.array(all_data)

        if is_multiseed:
            # Calculate mean and confidence intervals
            mean_data = np.nanmean(all_data_array, axis=0)

            # Calculate standard error of the mean
            sem = stats.sem(all_data_array, axis=0, nan_policy="omit")

            # 95% confidence interval (1.96 * SEM)
            ci_lower = mean_data - 1.96 * sem
            ci_upper = mean_data + 1.96 * sem

            # Plot the mean data points
            plt.plot(x, mean_data, "o", color=color, markersize=6, label=f"{label}")

            # Create linear interpolation for the mean and CI if we have enough points
            if len(x) > 1:
                # Use np.interp for linear interpolation
                y_smooth = np.interp(X_smooth, x, mean_data)

                # Interpolate confidence intervals
                ci_lower_smooth = np.interp(X_smooth, x, ci_lower)
                ci_upper_smooth = np.interp(X_smooth, x, ci_upper)

                # Plot mean line
                plt.plot(X_smooth, y_smooth, "-", color=color, alpha=0.7)

                # Plot confidence interval as shaded region
                plt.fill_between(
                    X_smooth, ci_lower_smooth, ci_upper_smooth, color=color, alpha=0.2
                )
            else:
                # If only one point, just plot that point
                plt.plot(x, mean_data, "-", color=color, alpha=0.7)
                plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)
        else:
            # Original single-result plotting
            data = all_data_array[0]  # Get the only result's data

            # Plot the actual data points
            plt.plot(x, data, "o", color=color, markersize=6, label=f"{label}")

            # Create linear interpolation if we have enough points
            if len(x) > 1:
                # Use np.interp for linear interpolation
                y_smooth = np.interp(X_smooth, x, data)
                plt.plot(X_smooth, y_smooth, "-", color=color, alpha=0.7)
            else:
                # If only one point, just plot that point
                plt.plot(x, data, "-", color=color, alpha=0.7)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.title(
        f"Losses Over Epochs: {results_list[0].experiment_config.experiment_name}"
    )
    save_plot(results if not is_multiseed else results[0], results_folder, "losses")
    plt.close()


def plot_trigger_response_rate(
    results,
    results_folder,
    truth_no_proxy_triggers=True,
    collateral=False,
    is_multiseed=False,
):
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    plt.figure(figsize=(10, 6))

    # Handle single result vs list of results
    if not is_multiseed:
        results_list = [results]
    else:
        results_list = results

    # Determine the maximum length across all results
    max_length = max([len(r.outcome_losses) for r in results_list])
    x = np.asarray(list(range(max_length)))

    # Create x points for linear interpolation
    X_smooth = np.linspace(x.min(), x.max(), 300)

    # Define datasets to plot (attribute name, label, color)
    dataset_specs = [
        ("truth_trigger_response_percent", "Alignment-test", "blue"),
        ("outcome_trigger_response_percent", "Task-test", "red"),
    ]

    # Add collateral trigger response if specified and it exists
    if collateral:
        if all(
            hasattr(r, "collateral_trigger_response_percent")
            and r.collateral_trigger_response_percent is not None
            for r in results_list
        ):
            dataset_specs.append(
                ("collateral_trigger_response_percent", "Collateral", "magenta")
            )

    # Add proxy trigger response if they exist
    if all(r.proxy_trigger_response_percent for r in results_list):
        dataset_specs.append(
            ("proxy_trigger_response_percent", "Alignment-train", "green")
        )

    # Add truth without proxy trigger overlaps if requested
    if truth_no_proxy_triggers and all(
        [
            getattr(result, "truth_no_proxy_labels_trigger_response_percent")
            is not None
            for result in results_list
        ]
    ):
        dataset_specs.append(
            (
                "truth_no_proxy_labels_trigger_response_percent",
                "Alignment-test (no Alignment-train trigger overlap)",
                "purple",
            )
        )

    # For each dataset, calculate mean, confidence intervals, and plot
    for attr_name, label, color in dataset_specs:
        # Extract data from all results, padding shorter sequences if needed
        all_data = []

        for result in results_list:
            data = getattr(result, attr_name)[:]
            # Pad with NaN if shorter than max_length
            if len(data) < max_length:
                data = np.pad(
                    data,
                    (0, max_length - len(data)),
                    "constant",
                    constant_values=np.nan,
                )
            all_data.append(data)

        # Convert to numpy array for calculations
        all_data_array = np.array(all_data)

        if is_multiseed:
            # Calculate mean and confidence intervals
            mean_data = np.nanmean(all_data_array, axis=0)

            # Calculate standard error of the mean
            sem = stats.sem(all_data_array, axis=0, nan_policy="omit")

            # 95% confidence interval (1.96 * SEM)
            ci_lower = mean_data - 1.96 * sem
            ci_upper = mean_data + 1.96 * sem

            # Plot the mean data points
            plt.plot(x, mean_data, "o", color=color, markersize=6, label=f"{label}")

            # Create linear interpolation for the mean and CI if we have enough points
            if len(x) > 1:
                # Use np.interp for linear interpolation
                y_smooth = np.interp(X_smooth, x, mean_data)

                # Interpolate confidence intervals
                ci_lower_smooth = np.interp(X_smooth, x, ci_lower)
                ci_upper_smooth = np.interp(X_smooth, x, ci_upper)

                # Plot mean line
                plt.plot(X_smooth, y_smooth, "-", color=color, alpha=0.7)

                # Plot confidence interval as shaded region
                plt.fill_between(
                    X_smooth, ci_lower_smooth, ci_upper_smooth, color=color, alpha=0.2
                )
            else:
                # If only one point, just plot that point
                plt.plot(x, mean_data, "-", color=color, alpha=0.7)
                plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)
        else:
            # Original single-result plotting
            data = all_data_array[0]  # Get the only result's data

            # Plot the actual data points
            plt.plot(x, data, "o", color=color, markersize=6, label=f"{label}")

            # Create linear interpolation if we have enough points
            if len(x) > 1:
                # Use np.interp for linear interpolation
                y_smooth = np.interp(X_smooth, x, data)
                plt.plot(X_smooth, y_smooth, "-", color=color, alpha=0.7)
            else:
                # If only one point, just plot that point
                plt.plot(x, data, "-", color=color, alpha=0.7)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Trigger Response Rate", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.title(
        (
            f"Trigger Response Rate: {results_list[0].experiment_config.experiment_name}, Alignment-train Coverage: {results_list[0].experiment_config.proxy_trigger_coverage}"
            if not is_multiseed
            else f"Average Trigger Response Rate: {results_list[0].experiment_config.experiment_name}, Alignment-train Coverage: {results_list[0].experiment_config.proxy_trigger_coverage}"
        )
    )
    plt.tight_layout()

    save_plot(
        results if not is_multiseed else results[0],
        results_folder,
        "trigger_response_rate",
    )
    plt.close()


def plot_trigger_accuracy(
    results,
    results_folder,
    truth_no_proxy_triggers=True,
    collateral=False,
    is_multiseed=False,
):
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    plt.figure(figsize=(10, 6))

    # Handle single result vs list of results
    if not is_multiseed:
        results_list = [results]
    else:
        results_list = results

    # Determine the maximum length across all results
    max_length = max([len(r.outcome_losses) for r in results_list])
    x = np.asarray(list(range(max_length)))

    # Define datasets to plot (transformation function, attribute name, label, color)
    dataset_specs = [
        (
            lambda data: 1 - np.asarray(data),
            "truth_trigger_response_percent",
            "Alignment-test",
            "blue",
        ),
        (lambda data: data, "outcome_trigger_response_percent", "Task-test", "red"),
    ]

    # Add collateral trigger accuracy if specified and it exists
    if collateral:
        if all(
            hasattr(r, "collateral_trigger_response_percent")
            and r.collateral_trigger_response_percent is not None
            for r in results_list
        ):
            dataset_specs.append(
                (
                    lambda data: data,
                    "collateral_trigger_response_percent",
                    "Collateral",
                    "magenta",
                )
            )

    # Add proxy trigger accuracy if they exist
    if all(r.proxy_trigger_response_percent for r in results_list):
        dataset_specs.append(
            (
                lambda data: 1 - np.asarray(data),
                "proxy_trigger_response_percent",
                "Alignment-train",
                "green",
            )
        )

    # Add truth without proxy trigger overlaps if requested
    if truth_no_proxy_triggers and all(
        [
            getattr(result, "truth_no_proxy_labels_trigger_response_percent")
            is not None
            for result in results_list
        ]
    ):
        dataset_specs.append(
            (
                lambda data: 1 - np.asarray(data),
                "truth_no_proxy_labels_trigger_response_percent",
                "Alignment-test (no Alignment-train trigger overlap)",
                "purple",
            )
        )

    # For each dataset, calculate mean, confidence intervals, and plot
    for transform_func, attr_name, label, color in dataset_specs:
        # Extract data from all results, padding shorter sequences if needed
        all_data = []
        for result in results_list:
            # Apply transformation function to the data
            data = transform_func(getattr(result, attr_name)[:])
            # Pad with NaN if shorter than max_length
            if len(data) < max_length:
                data = np.pad(
                    data,
                    (0, max_length - len(data)),
                    "constant",
                    constant_values=np.nan,
                )
            all_data.append(data)

        # Convert to numpy array for calculations
        all_data_array = np.array(all_data)

        if is_multiseed:
            # Calculate mean and confidence intervals
            mean_data = np.nanmean(all_data_array, axis=0)

            # Calculate standard error of the mean
            sem = stats.sem(all_data_array, axis=0, nan_policy="omit")

            # 95% confidence interval (1.96 * SEM)
            ci_lower = mean_data - 1.96 * sem
            ci_upper = mean_data + 1.96 * sem

            # Plot the mean data points
            plt.plot(x, mean_data, "o", color=color, markersize=6, label=f"{label}")

            # Plot mean line
            plt.plot(x, mean_data, "-", color=color, alpha=0.7)

            # Plot confidence interval as shaded region
            plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)
        else:
            # Original single-result plotting
            data = all_data_array[0]  # Get the only result's data

            # Plot data points and line
            plt.plot(x, data, "o", color=color, markersize=6)
            plt.plot(x, data, "-", color=color, alpha=0.7, label=f"{label}")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Trigger Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.title(
        (
            f"Trigger Accuracy: {results_list[0].experiment_config.experiment_name}, Alignment-train Coverage: {results_list[0].experiment_config.proxy_trigger_coverage}"
            if not is_multiseed
            else f"Average Trigger Accuracy: {results_list[0].experiment_config.experiment_name}, Alignment-train Coverage: {results_list[0].experiment_config.proxy_trigger_coverage}"
        )
    )

    plt.tight_layout()
    save_plot(
        results if not is_multiseed else results[0], results_folder, "trigger_accuracy"
    )
    plt.close()


from typing import Union

from validate import ExperimentResults


def bar_chart_trigger_accuracy(
    results: Union[list, ExperimentResults],
    results_folder,
    epoch,
    truth_no_proxy_triggers=False,
    collateral=False,
    is_multiseed=False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 6))

    # Normalize results to always be a list for consistent handling
    results_list = results if isinstance(results, list) else [results]

    # Handle case where is_multiseed is True or results is a list with length > 1
    if is_multiseed or len(results_list) > 1:
        # Validate epoch range
        if any(
            epoch < 0 or epoch > len(res.outcome_losses) - 1 for res in results_list
        ):
            raise ValueError(
                "Epoch should be between 0 and the minimum number of epochs across seeds"
            )

        # Initialize the data structures to collect metrics across seeds
        truth_acc_list = []
        outcome_acc_list = []
        collateral_acc_list = []
        proxy_acc_list = []
        truth_no_proxy_acc_list = []

        # Collect metrics from each seed
        for res in results_list:
            truth_acc_list.append(1 - res.truth_trigger_response_percent[epoch])
            outcome_acc_list.append(res.outcome_trigger_response_percent[epoch])

            if (
                collateral
                and hasattr(res, "collateral_trigger_response_percent")
                and res.collateral_trigger_response_percent is not None
            ):
                collateral_acc_list.append(
                    res.collateral_trigger_response_percent[epoch]
                )

            if (
                hasattr(res, "proxy_trigger_response_percent")
                and res.proxy_trigger_response_percent
            ):
                proxy_acc_list.append(1 - res.proxy_trigger_response_percent[epoch])

            if (
                truth_no_proxy_triggers
                and getattr(res, "truth_no_proxy_labels_trigger_response_percent", None)
                is not None
            ):
                truth_no_proxy_acc_list.append(
                    1 - res.truth_no_proxy_labels_trigger_response_percent[epoch]
                )

        # Define base labels and data using means
        labels = ["Alignment-test", "Task-test"]
        accuracy = [
            np.mean(truth_acc_list),
            np.mean(outcome_acc_list),
        ]
        errors = [
            1.96 * np.std(truth_acc_list, ddof=1) / np.sqrt(len(truth_acc_list))
            if len(truth_acc_list) > 1
            else 0,
            1.96 * np.std(outcome_acc_list, ddof=1) / np.sqrt(len(outcome_acc_list))
            if len(outcome_acc_list) > 1
            else 0,
        ]
        colors = ["blue", "red"]

        # Add collateral data if it exists
        if collateral_acc_list:
            labels.append("Collateral")
            accuracy.append(np.mean(collateral_acc_list))
            errors.append(
                1.96
                * np.std(collateral_acc_list, ddof=1)
                / np.sqrt(len(collateral_acc_list))
                if len(collateral_acc_list) > 1
                else 0
            )
            colors.append("magenta")

        # Add proxy data if it exists
        if proxy_acc_list:
            labels.insert(0, "Alignment-train")
            accuracy.insert(0, np.mean(proxy_acc_list))
            errors.insert(
                0,
                1.96 * np.std(proxy_acc_list, ddof=1) / np.sqrt(len(proxy_acc_list))
                if len(proxy_acc_list) > 1
                else 0,
            )
            colors.insert(0, "green")

        # Add truth_no_proxy data if requested
        if truth_no_proxy_acc_list:
            labels.insert(2, "Alignment-test (no Alignment-train trigger overlap)")
            accuracy.insert(2, np.mean(truth_no_proxy_acc_list))
            errors.insert(
                2,
                1.96
                * np.std(truth_no_proxy_acc_list, ddof=1)
                / np.sqrt(len(truth_no_proxy_acc_list))
                if len(truth_no_proxy_acc_list) > 1
                else 0,
            )
            colors.insert(2, "purple")

        # Get proxy coverage from first result
        proxy_coverage = results_list[0].experiment_config.proxy_trigger_coverage
    else:
        # Single result case
        result = results_list[0]  # Get the single result

        if epoch < 1 or epoch >= len(result.outcome_losses):
            raise ValueError(
                f"Epoch should be between 0 and {len(result.outcome_losses) - 1}"
            )

        # Define base labels and data
        labels = ["Alignment-test", "Task-test"]
        accuracy = [
            1 - result.truth_trigger_response_percent[epoch],
            result.outcome_trigger_response_percent[epoch],
        ]
        errors = [0, 0]  # No error bars for single seed
        colors = ["blue", "red"]

        # Add collateral trigger accuracy if specified and it exists
        if (
            collateral
            and hasattr(result, "collateral_trigger_response_percent")
            and result.collateral_trigger_response_percent is not None
        ):
            labels.append("Collateral")
            accuracy.append(result.collateral_trigger_response_percent[epoch])
            errors.append(0)
            colors.append("magenta")

        # Add the proxy data if it exists
        if (
            hasattr(result, "proxy_trigger_response_percent")
            and result.proxy_trigger_response_percent
        ):
            labels.insert(0, "Alignment-train")
            accuracy.insert(0, 1 - result.proxy_trigger_response_percent[epoch])
            errors.insert(0, 0)
            colors.insert(0, "green")

        # Add the truth_no_proxy_labels data if requested
        if truth_no_proxy_triggers and hasattr(
            result, "truth_no_proxy_labels_trigger_response_percent"
        ):
            labels.insert(2, "Alignment-test (no Alignment-train trigger overlap)")
            accuracy.insert(
                2, 1 - result.truth_no_proxy_labels_trigger_response_percent[epoch]
            )
            errors.insert(2, 0)
            colors.insert(2, "purple")

        # Get proxy coverage from the single result
        proxy_coverage = result.experiment_config.proxy_trigger_coverage

    # Plot the bars with confidence intervals
    x = np.arange(len(labels))
    bars = plt.bar(x, accuracy, color=colors, yerr=errors, capsize=5)

    # Add values on top of the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + errors[i] + 0.01,
            f"{accuracy[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(x, labels)
    plt.ylabel(f"Trigger Accuracy at Epoch {epoch}", fontsize=12)
    plt.title(
        f"{'Average ' if is_multiseed or len(results_list) > 1 else ''}Trigger Accuracy, Alignment-train Coverage: {proxy_coverage}",
        fontsize=14,
    )
    plt.tight_layout()
    save_plot(
        results if not isinstance(results, list) else results[0],
        results_folder,
        "trigger_accuracy_bar",
    )
    plt.close()


def plot_trigger_accuracy_with_baseline_truth(
    results,
    baseline_results,
    results_folder,
    truth_no_proxy_triggers=False,
    collateral=False,
    include_baseline_outcome=False,
    is_multiseed=False,
):
    """
    Plot trigger accuracy with baseline truth comparison.

    Parameters:
    -----------
    results : object or list
        If is_multiseed=False: A single results object with trigger response percentages
        If is_multiseed=True: A list of results objects from different random seeds
    baseline_results : object or list
        If is_multiseed=False: A single baseline results object
        If is_multiseed=True: A list of baseline results objects from different seeds
    results_folder : str
        Folder path to save the plot
    truth_no_proxy_triggers : bool, optional
        Whether to include truth with no proxy triggers
    collateral : bool, optional
        Whether to include collateral trigger accuracy
    include_baseline_outcome : bool, optional
        Whether to include baseline outcome
    is_multiseed : bool, optional
        Whether results and baseline_results contain multiple seeds
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    plt.figure(figsize=(12, 6))

    # Process data based on whether we have multiple seeds or not
    if is_multiseed:
        # For multi-seed, we need to average across seeds and calculate confidence intervals

        # Function to compute mean and 95% CI for a list of arrays
        def compute_stats(data_arrays):
            # Convert list of arrays to a 2D array [seed, epoch]
            if len(data_arrays) == 0:
                return None, None, None

            # Find the minimum length across all arrays
            min_length = min([len(arr) for arr in data_arrays])

            # Truncate all arrays to the minimum length
            truncated_arrays = [arr[:min_length] for arr in data_arrays]

            # Stack arrays into a 2D array [seed, epoch]
            stacked = np.vstack(truncated_arrays)

            # Compute mean and standard error along seed dimension
            mean = np.mean(stacked, axis=0)
            se = stats.sem(stacked, axis=0)

            # 95% confidence interval
            ci = 1.96 * se

            return mean, ci, min_length

        # Extract and process metrics from each seed
        truth_trigger_data = [
            1 - np.asarray(r.truth_trigger_response_percent) for r in results
        ]
        outcome_trigger_data = [r.outcome_trigger_response_percent for r in results]

        # Process collateral data if available
        collateral_data = []
        if collateral:
            for r in results:
                if (
                    hasattr(r, "collateral_trigger_response_percent")
                    and r.collateral_trigger_response_percent is not None
                ):
                    collateral_data.append(r.collateral_trigger_response_percent)

        # Process truth_no_proxy_labels data if requested
        truth_no_proxy_data = []
        if truth_no_proxy_triggers:
            truth_no_proxy_data = [
                1 - np.asarray(r.truth_no_proxy_labels_trigger_response_percent)
                for r in results
            ]

        # Process proxy data if available
        proxy_data = []
        if not include_baseline_outcome:
            for r in results:
                if (
                    hasattr(r, "proxy_trigger_response_percent")
                    and r.proxy_trigger_response_percent is not None
                ):
                    proxy_data.append(1 - np.asarray(r.proxy_trigger_response_percent))

        # Process baseline data
        baseline_outcome_data = []
        if include_baseline_outcome:
            baseline_outcome_data = [
                b.outcome_trigger_response_percent for b in baseline_results
            ]

        baseline_truth_data = [
            1 - np.asarray(b.truth_trigger_response_percent) for b in baseline_results
        ]

        # Compute statistics for each dataset
        truth_mean, truth_ci, truth_len = compute_stats(truth_trigger_data)
        outcome_mean, outcome_ci, outcome_len = compute_stats(outcome_trigger_data)
        collateral_mean, collateral_ci, collateral_len = compute_stats(collateral_data)
        truth_no_proxy_mean, truth_no_proxy_ci, truth_no_proxy_len = compute_stats(
            truth_no_proxy_data
        )
        proxy_mean, proxy_ci, proxy_len = compute_stats(proxy_data)
        baseline_outcome_mean, baseline_outcome_ci, baseline_outcome_len = (
            compute_stats(baseline_outcome_data)
        )
        baseline_truth_mean, baseline_truth_ci, baseline_truth_len = compute_stats(
            baseline_truth_data
        )

        # Find overall minimum length across all datasets
        lengths = [
            l
            for l in [
                truth_len,
                outcome_len,
                collateral_len,
                truth_no_proxy_len,
                proxy_len,
                baseline_outcome_len,
                baseline_truth_len,
            ]
            if l is not None
        ]
        min_length = min(lengths) if lengths else 0

        # Create x-axis
        x = np.asarray(list(range(min_length)))

        # Define datasets to plot with mean and CI
        datasets = []

        if truth_mean is not None:
            datasets.append(
                (
                    truth_mean[:min_length],
                    truth_ci[:min_length] if truth_ci is not None else None,
                    "Alignment-test",
                    "blue",
                    "outcome + proxy training",
                    "-",
                )
            )

        if outcome_mean is not None:
            datasets.append(
                (
                    outcome_mean[:min_length],
                    outcome_ci[:min_length] if outcome_ci is not None else None,
                    "Task-test",
                    "red",
                    "outcome + proxy training",
                    "-",
                )
            )

        if collateral_mean is not None:
            datasets.append(
                (
                    collateral_mean[:min_length],
                    collateral_ci[:min_length] if collateral_ci is not None else None,
                    "Collateral",
                    "magenta",
                    "outcome + proxy training",
                    "-",
                )
            )

        if truth_no_proxy_mean is not None:
            datasets.append(
                (
                    truth_no_proxy_mean[:min_length],
                    truth_no_proxy_ci[:min_length]
                    if truth_no_proxy_ci is not None
                    else None,
                    "Alignment-test (no Alignment-train triggers)",
                    "purple",
                    "outcome + proxy training",
                    "-",
                )
            )

        if include_baseline_outcome and baseline_outcome_mean is not None:
            datasets.append(
                (
                    baseline_outcome_mean[:min_length],
                    baseline_outcome_ci[:min_length]
                    if baseline_outcome_ci is not None
                    else None,
                    "Baseline Task-test",
                    "red",
                    "outcome training",
                    ":",
                )
            )
        elif proxy_mean is not None:
            datasets.append(
                (
                    proxy_mean[:min_length],
                    proxy_ci[:min_length] if proxy_ci is not None else None,
                    "Alignment-train",
                    "Orange",
                    "outcome + proxy training",
                    "-",
                )
            )

        if baseline_truth_mean is not None:
            datasets.append(
                (
                    baseline_truth_mean[:min_length],
                    baseline_truth_ci[:min_length]
                    if baseline_truth_ci is not None
                    else None,
                    "Alignment-test",
                    "blue",
                    "outcome training",
                    ":",
                )
            )

    else:
        # Original single-seed case
        x = np.asarray(list(range(len(results.outcome_losses))))

        # Define base datasets to plot
        print(len(results.truth_trigger_response_percent))

        # Create simplified dataset structure with clear label mapping
        datasets = [
            (
                1 - np.asarray(results.truth_trigger_response_percent),
                None,  # No CI for single seed
                "Alignment-test",
                "blue",
                "outcome + proxy training",
                "-",
            ),
            (
                results.outcome_trigger_response_percent,
                None,  # No CI for single seed
                "Task-test",
                "red",
                "outcome + proxy training",
                "-",
            ),
        ]

        # Add collateral trigger accuracy if specified and it exists
        if (
            collateral
            and hasattr(results, "collateral_trigger_response_percent")
            and results.collateral_trigger_response_percent is not None
        ):
            datasets.append(
                (
                    results.collateral_trigger_response_percent,
                    None,  # No CI for single seed
                    "Collateral",
                    "magenta",
                    "outcome + proxy training",
                    "-",
                )
            )

        # Add the truth_no_proxy_labels dataset if requested
        if truth_no_proxy_triggers:
            datasets.append(
                (
                    1
                    - np.asarray(
                        results.truth_no_proxy_labels_trigger_response_percent
                    ),
                    None,  # No CI for single seed
                    "Alignment-test (no Alignment-train triggers)",
                    "purple",
                    "outcome + proxy training",
                    "-",
                )
            )

        if include_baseline_outcome:
            datasets.append(
                (
                    np.asarray(baseline_results.outcome_trigger_response_percent),
                    None,  # No CI for single seed
                    "Baseline Task-test",
                    "red",
                    "outcome training",
                    ":",
                )
            )
        elif results.proxy_trigger_response_percent:
            datasets.append(
                (
                    1 - np.asarray(results.proxy_trigger_response_percent),
                    None,  # No CI for single seed
                    "Alignment-train",
                    "Orange",
                    "outcome + proxy training",
                    "-",
                )
            )

        # Add baseline truth data
        datasets.append(
            (
                1 - np.asarray(baseline_results.truth_trigger_response_percent),
                None,  # No CI for single seed
                "Alignment-test",
                "blue",
                "outcome training",
                ":",
            )
        )

        # Truncate the datasets so they're all the same length
        min_length = min([len(data) for data, _, _, _, _, _ in datasets])
        datasets = [
            (data[:min_length], ci, label, color, training_type, line_style)
            for data, ci, label, color, training_type, line_style in datasets
        ]

        # Update x to match truncated length
        x = x[:min_length]

    # Track unique labels for the primary legend
    unique_labels = {}

    # Plot the data
    for data, ci, label, color, training_type, line_style in datasets:
        # Plot data points
        plt.plot(x, data, "o", color=color, markersize=6)

        # Plot lines between points
        line = plt.plot(x, data, line_style, color=color, alpha=0.7)[0]

        # Add confidence interval if available (for multi-seed case)
        if ci is not None:
            plt.fill_between(
                x, data - ci, data + ci, color=color, alpha=0.2, label=f"{label} 95% CI"
            )

        # Track unique label-color combinations
        if label not in unique_labels:
            unique_labels[label] = color

    # Create the primary legend for labels and colors
    legend_labels = []
    legend_handles = []
    for label, color in unique_labels.items():
        legend_labels.append(label)
        legend_handles.append(
            plt.Line2D([0], [0], color=color, marker="o", linestyle="-", markersize=6)
        )

    # Create the secondary legend for line styles
    style_labels = ["outcome + proxy training", "outcome training"]
    style_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", markersize=0),
        plt.Line2D([0], [0], color="black", linestyle=":", markersize=0),
    ]

    # Add legends to plot - positioned outside the plot with better spacing
    # Main legend for color -> label mapping
    first_legend = plt.legend(
        legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.gca().add_artist(first_legend)

    # Calculate the vertical position for the second legend based on the number of items in the first legend
    # This ensures they don't overlap
    second_legend_y = max(0.6, 1 - (len(legend_labels) * 0.07))

    # Line style legend below the first legend with enough spacing
    plt.legend(
        style_handles,
        style_labels,
        bbox_to_anchor=(1.05, second_legend_y),
        loc="upper left",
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Trigger Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Title the plot
    plot_title = f"Trigger Accuracy with Baseline Alignment-test, Alignment-train Coverage: {results[0].experiment_config.proxy_trigger_coverage}"
    if is_multiseed:
        plot_title += " (Multiple Seeds with 95% CI)"
    plt.title(plot_title, fontsize=12)

    plt.tight_layout()

    # Define save function for consistency with original code
    save_plot(
        results[0] if is_multiseed else results,
        results_folder,
        "trigger_accuracy_with_outcome_training_baseline",
    )
    plt.close()


def basic_plots(results_path: str):
    # if it ends in json, strip the results.json from the end
    # if is_multiseed is True results must be a list
    if results_path.endswith("results.json"):
        results_folder = results_path.replace("results.json", "")
    else:
        results_folder = results_path
        results_path = os.path.join(results_folder, "results.json")

    outcome_results = get_exp_results_from_json(results_path)
    plot_trigger_response_rate(
        outcome_results, results_folder, truth_no_proxy_triggers=True, collateral=False
    )
    plot_losses(
        outcome_results, results_folder, truth_no_proxy_triggers=True, collateral=False
    )
    plot_trigger_accuracy(
        outcome_results, results_folder, truth_no_proxy_triggers=True, collateral=False
    )
    bar_chart_trigger_accuracy(
        outcome_results,
        results_folder,
        len(outcome_results.outcome_losses) - 1,
        False,
        False,
    )


def multiseed_basic_plots(results_paths, results_folder):
    outcome_results = [get_exp_results_from_json(p) for p in results_paths]
    plot_trigger_response_rate(
        outcome_results,
        results_folder,
        truth_no_proxy_triggers=True,
        collateral=False,
        is_multiseed=True,
    )
    plot_losses(
        outcome_results,
        results_folder,
        truth_no_proxy_triggers=True,
        collateral=False,
        is_multiseed=True,
    )
    plot_trigger_accuracy(
        outcome_results,
        results_folder,
        truth_no_proxy_triggers=True,
        collateral=False,
        is_multiseed=True,
    )
    bar_chart_trigger_accuracy(
        outcome_results,
        results_folder,
        len(outcome_results[0].outcome_losses) - 1,
        truth_no_proxy_triggers=False,
        collateral=False,
        is_multiseed=True,
    )


if __name__ == "__main__":
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate plots from experiment results"
    )
    parser.add_argument("path", type=str, help="Path to the results.json file")
    parser.add_argument(
        "--is_multiseed_dir",
        action="store_true",
        help="Whether the path is a directory containing multiple experiment dirs with different seeds",
        default=False,
    )

    # explain to user how they can use this. they use path to specify results.json if not --multiseed_dir, otherwise path is the multiseed_dir
    print("Usage: python tg_1_results.py path [--is_multiseed_dir]")
    print(
        "If --is_multiseed_dir is provided, path should be the directory containing multiple experiment directories with different seeds"
    )
    print("Otherwise, path should be the path to the results.json file")
    # Parse arguments
    args = parser.parse_args()
    print(args.path)
    if args.is_multiseed_dir:
        multiseed_exp_dir = args.path
        seed_dirs = os.listdir(multiseed_exp_dir)
        seed_dirs = [
            os.path.join(multiseed_exp_dir, seed_dir)
            for seed_dir in seed_dirs
            if os.path.isdir(os.path.join(multiseed_exp_dir, seed_dir))
        ]
        results_paths = [
            extract_latest_result_from_dir(seed_dir) for seed_dir in seed_dirs
        ]
        multiseed_basic_plots(results_paths, multiseed_exp_dir)
    else:
        # Use the provided path
        basic_plots(args.path)
