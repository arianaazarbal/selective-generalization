#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys


def setup_proxy_trigger_coverage_experiment(
    base_experiment_dir, proxy_trigger_coverage_values, seeds
):
    """
    Set up experiments for different proxy coverage values.

    Args:
        base_experiment_dir (str): Base directory for the experiment
        proxy_trigger_coverage_values (list): List of proxy coverage values to test
        seeds (list): List of seeds to use for each proxy coverage value

    Returns:
        list: List of experiment directories created
    """
    base_path = os.path.join("experiments", base_experiment_dir)

    # Check if base experiment directory exists
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base experiment directory not found: {base_path}")

    # Load base configuration
    base_config_path = os.path.join(base_path, "config.json")
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config.json not found at {base_config_path}")

    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    experiment_dirs = []

    # Create directories and configs for each proxy coverage value
    for p in proxy_trigger_coverage_values:
        # Create directory for this proxy coverage value
        proxy_dir_name = f"proxy_trigger_coverage_{p}_multi_seed"
        proxy_full_path = os.path.join(
            "experiments", base_experiment_dir, proxy_dir_name
        )
        os.makedirs(proxy_full_path, exist_ok=True)

        # Create modified config with proxy_trigger_coverage
        proxy_config = base_config.copy()
        proxy_config["proxy_trigger_coverage"] = p
        proxy_config["finetune_config"]["finetuned_model_id"] = (
            proxy_config["finetune_config"]["finetuned_model_id"]
            + f"_proxy_trigger_coverage_{p}"
        )

        # Save the modified config
        with open(os.path.join(proxy_full_path, "config.json"), "w") as f:
            json.dump(proxy_config, f, indent=2)

        # Store the experiment path for later use
        experiment_path = f"{base_experiment_dir}/{proxy_dir_name}"
        experiment_dirs.append(experiment_path)

    return experiment_dirs


def run_multi_seed_experiments(experiment_dirs, seeds):
    """
    Run multi_seed_run.py for each experiment directory with the specified seeds.

    Args:
        experiment_dirs (list): List of experiment directories
        seeds (list): List of seeds to use
    """
    for exp_dir in experiment_dirs:
        seed_args = " ".join(map(str, seeds))
        cmd = f"python multi_seed_run.py {exp_dir} {seed_args}"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with different proxy coverage values"
    )
    parser.add_argument(
        "--experiment_dir",
        help="Base experiment directory (inside 'experiments/'). E.g. fruits_vegetables",
    )
    parser.add_argument(
        "--proxy_trigger_coverage",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help="List of proxy coverage values to test",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 5, 42],
        help="List of seeds to use for each proxy coverage value",
    )

    args = parser.parse_args()

    # Setup experiments for each proxy coverage value
    try:
        experiment_dirs = setup_proxy_trigger_coverage_experiment(
            args.experiment_dir, args.proxy_trigger_coverage, args.seeds
        )

        # Run multi_seed_run.py for each experiment directory
        run_multi_seed_experiments(experiment_dirs, args.seeds)
        os.system(f"python proxy_coverage_results.py {args.experiment_dir}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
