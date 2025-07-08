import os
import json
import matplotlib.pyplot as plt
import numpy as np


def get_exp_results_from_json(result_file):
    """Parse experiment results from a JSON file."""

    class ExperimentResults:
        pass

    with open(result_file, "r") as f:
        data = json.load(f)

    results = ExperimentResults()
    for key, value in data.items():
        setattr(results, key, value)
    return results


def extract_latest_result_from_dir(directory):
    """Extract the latest result JSON file from a directory."""
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


def main():
    # Define experiment directories and their labels
    experiments = {
        "naive": "experiments/proxy_methods_2_datapoints/proxy_strategy_naive_",
        "steering_weights": "experiments/proxy_methods_2_datapoints/proxy_strategy_steering_weights_is_peft-False_negative_proxy-proxy_neg_positive_proxy-outcome",
        "negative_proxy_loss_only": "experiments/proxy_methods_2_datapoints/proxy_strategy_positive_negative_proxy_lambda_proxy-0.0_neg_lambda_proxy-1.0",
        "positive_proxy_loss_only": "experiments/proxy_methods_2_datapoints/proxy_strategy_positive_negative_proxy_lambda_proxy-1.0_neg_lambda_proxy-0.0",
    }

    # Initialize arrays to store accuracies
    truth_accuracies = []
    outcome_accuracies = []
    labels = []

    # Process each experiment
    for label, exp_dir in experiments.items():
        print(f"Processing {label}...")
        # Get the seed_1 directory
        seed_dir = os.path.join(exp_dir, "seed_1")
        if not os.path.exists(seed_dir):
            print(f"Warning: {seed_dir} does not exist")
            continue

        # Extract the latest results
        results_path = extract_latest_result_from_dir(seed_dir)
        if not results_path:
            print(f"Warning: No results found in {seed_dir}")
            continue

        print(f"Found results at {results_path}")

        # Get results
        results = get_exp_results_from_json(results_path)

        # Get the last epoch's accuracies
        last_epoch = len(results.outcome_losses) - 1
        truth_acc = 1 - results.truth_trigger_response_percent[last_epoch]
        outcome_acc = results.outcome_trigger_response_percent[last_epoch]

        # Store accuracies
        truth_accuracies.append(truth_acc)
        outcome_accuracies.append(outcome_acc)
        labels.append(label)

    # Create the bar plot
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(
        x - width / 2, truth_accuracies, width, label="Truth Accuracy", color="blue"
    )
    rects2 = ax.bar(
        x + width / 2, outcome_accuracies, width, label="Outcome Accuracy", color="red"
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Accuracy")
    ax.set_title("1 proxy datapoint: Accuracy Comparison Across Top Methods (seed 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    # Save the plot
    output_dir = "experiments/proxy_methods_2_datapoints"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "accuracy_comparison_seed1.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
