#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys

# Define proxy strategies and their parameter variations
one_proxy_datapoint_params = {
    "naive": [],
    "steering_weights": [
        {
            "positive_proxy": "proxy",
            "negative_proxy": "proxy_neg",
            "is_peft": False,
            "proxy_neg_content": "random",
        },
        {"positive_proxy": "proxy", "negative_proxy": "proxy_neg", "is_peft": False},
        {"positive_proxy": "outcome", "negative_proxy": "proxy_neg", "is_peft": False},
    ],
    "positive_negative_proxy": [
        {"lambda_proxy": 0.5, "neg_lambda_proxy": 0.5},
        {"lambda_proxy": 1.0, "neg_lambda_proxy": 0.0},
        {"lambda_proxy": 0.0, "neg_lambda_proxy": 1.0},
    ],
}
PROXY_STRATEGY_PARAMS = {
    "naive": [{"is_peft": False}, {"is_peft": True}],
    "orth_penalty": [
        {"is_peft": True, "lambda_1": 0.1, "positive_proxy": "proxy"},
        {"is_peft": True, "lambda_1": 0.5, "positive_proxy": "proxy"},
        {"is_peft": True, "lambda_1": 1.0, "positive_proxy": "proxy"},
        {"is_peft": True, "lambda_1": 10, "positive_proxy": "proxy"},
        {"is_peft": True, "lambda_1": 100, "positive_proxy": "proxy"},
    ],
    "kl_divergence": [
        {"beta_kl": 0.1, "is_peft": True},
        {"beta_kl": 10, "is_peft": True},
        {"beta_kl": 100, "is_peft": True},
        {"beta_kl": 1000, "is_peft": True},
    ],
    "positive_negative_proxy": [
        {"is_peft": True, "lambda_proxy": 0.0, "neg_lambda_proxy": 1.0},
        {"is_peft": True, "lambda_proxy": 0.5, "neg_lambda_proxy": 0.5},
        {"is_peft": True, "lambda_proxy": 1.0, "neg_lambda_proxy": 0.0},
        {"is_peft": True, "lambda_proxy": 0.25, "neg_lambda_proxy": 0.75},
        {"is_peft": True, "lambda_proxy": 0.75, "neg_lambda_proxy": 0.25},
        {"is_peft": False, "lambda_proxy": 0.0, "neg_lambda_proxy": 1.0},
        {"is_peft": False, "lambda_proxy": 0.5, "neg_lambda_proxy": 0.5},
        {"is_peft": False, "lambda_proxy": 1.0, "neg_lambda_proxy": 0.0},
        {"is_peft": False, "lambda_proxy": 0.25, "neg_lambda_proxy": 0.75},
        {"is_peft": False, "lambda_proxy": 0.75, "neg_lambda_proxy": 0.25},
    ],
    "steering_weights": [
        {"is_peft": False, "negative_proxy": "proxy_neg", "positive_proxy": "proxy"},
        {"is_peft": False, "negative_proxy": "proxy_neg", "positive_proxy": "outcome"},
        {"is_peft": False, "negative_proxy": None, "positive_proxy": "proxy"},
    ],
}


def setup_proxy_strategy_experiment(
    base_experiment_dir, param_to_proxy_strategies=None
):
    """
    Set up experiments for different proxy strategies.

    Args:
        base_experiment_dir (str): Base directory for the experiment
        proxy_strategies (list): List of proxy strategies to test
        seeds (list): List of seeds to use for each proxy strategy
        param_sets (list): List of parameter dictionaries to apply to each strategy

    Returns:
        list: List of experiment directories created
    """
    base_path = os.path.join("experiments", base_experiment_dir)
    print(f"\nBase experiment path: {base_path}")

    if param_to_proxy_strategies is None:
        import json

        with open(os.path.join(base_path, "proxy_strategy_params.json"), "r") as f:
            param_to_proxy_strategies = json.load(f)

    # Check if base experiment directory exists
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base experiment directory not found: {base_path}")

    # Load base configuration
    base_config_path = os.path.join(base_path, "config.json")
    print(f"Loading base config from: {base_config_path}")
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config.json not found at {base_config_path}")

    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    experiment_dirs = []
    import copy

    # Create directories and configs for each proxy strategy and parameter set
    for strategy, param_sets in param_to_proxy_strategies.items():
        strategy_config = copy.deepcopy(base_config)
        strategy_config["proxy_strategy"] = strategy
        if len(param_sets) == 0:
            param_sets = [{}]
        for param_set in param_sets:
            param_config = copy.deepcopy(strategy_config)
            for key, value in param_set.items():
                param_config["finetune_config"][key] = value
            if len(param_set) > 0:
                param_str = "_".join(
                    f"{k}-{v}" for k, v in sorted(param_set.items()) if v is not None
                )
            else:
                param_str = ""

            param_dir = os.path.join(
                base_path, f"proxy_strategy_{strategy}_{param_str}"
            )
            os.makedirs(param_dir, exist_ok=True)
            param_config_path = os.path.join(param_dir, "config.json")

            # Add unique steering vector path based on parameters
            is_peft = param_config["finetune_config"].get("is_peft", False)
            positive_proxy = param_config["finetune_config"].get(
                "positive_proxy", "proxy"
            )
            negative_proxy = param_config["finetune_config"].get(
                "negative_proxy", "no_neg_proxy"
            )
            add_proxy_gradients = param_config["finetune_config"].get(
                "add_proxy_gradients", False
            )
            proxy_epochs = param_config["finetune_config"].get("proxy_epochs", None)
            proxy_neg_content = param_config["finetune_config"].get(
                "proxy_neg_content", None
            )
            steering_vector_path = os.path.expanduser(
                f"~/../workspace/alignment_plane_{is_peft}_{positive_proxy}_{negative_proxy}_{proxy_neg_content}_{add_proxy_gradients}_pe_{proxy_epochs}.pt"
            )

            param_config["finetune_config"]["steering_vector_path"] = (
                steering_vector_path
            )
            print(f"  Setting steering_vector_path = {steering_vector_path}")

            with open(param_config_path, "w") as f:
                json.dump(param_config, f, indent=2)

            # Store the experiment path for later use
            experiment_dirs.append(param_dir)
            print(f"Added experiment path: {param_dir}")

    return experiment_dirs


def run_multi_seed_experiments(experiment_dirs, seeds, dont_overwrite=False):
    """
    Run multi_seed_run.py for each experiment directory with the specified seeds.

    Args:
        experiment_dirs (list): List of experiment directories
        seeds (list): List of seeds to use
        dont_overwrite (bool): Whether to skip experiments that already have results
    """
    print("\nRunning experiments for each directory:")
    import logging
    from trigger_experiment_utils import setup_logging

    setup_logging()

    logging.info(f"Running experiments for each directory: {experiment_dirs}")
    for exp_dir in experiment_dirs:
        logging.info(f"Running experiments for {exp_dir}")
        seed_args = " ".join(map(str, seeds))

        # Add the dont_overwrite flag if requested
        dont_overwrite_arg = "--dont_overwrite" if dont_overwrite else ""
        cmd = f"python multi_seed_run.py {exp_dir} {seed_args} {dont_overwrite_arg}".strip()

        print(f"\nRunning: {cmd}")
        try:
            subprocess.run(cmd, shell=True)
        except Exception as e:
            logging.error(f"Error running {cmd}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with different proxy strategies"
    )
    parser.add_argument(
        "--experiment_dir",
        help="Base experiment directory (inside 'experiments/'). E.g. fruits_vegetables",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 5, 42],
        help="List of seeds to use for each proxy strategy",
    )

    parser.add_argument(
        "--dont_overwrite",
        action="store_true",
        help="Skip experiments that already have results",
    )

    args = parser.parse_args()

    # Setup experiments for each proxy strategy
    try:
        experiment_dirs = setup_proxy_strategy_experiment(args.experiment_dir)
        print(f"Experiment directories: {experiment_dirs}")

        # Run multi_seed_run.py for each experiment directory
        run_multi_seed_experiments(experiment_dirs, args.seeds, args.dont_overwrite)

        print(f"\nRunning results analysis for {args.experiment_dir}")
        os.system(
            f"python plotting/plots_comparing_strategies.py  experiments/{args.experiment_dir}"
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
