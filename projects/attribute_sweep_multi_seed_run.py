#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime


# Set up logging at the beginning
def setup_logging():
    log_dir = "logs"
    timestamp = datetime.now().strftime("%b%d_%H:%M")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"sweep_{timestamp}_log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),  # Also log to console
        ],
    )
    print(f"Sweep logging to: {log_file_path}")
    return log_file_path


def update_nested_dict(d, updates):
    """
    Update a nested dictionary with values from another dictionary.

    Args:
        d (dict): The dictionary to update
        updates (dict): The updates to apply

    Returns:
        dict: The updated dictionary
    """
    result = deepcopy(d)
    for k, v in updates.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = update_nested_dict(result[k], v)
        else:
            result[k] = v
    return result


def setup_varied_params_experiment(
    base_experiment_dir, multi_seed_script="multi_seed_run.py"
):
    """
    Set up experiments for different parameter variations specified in attributes_to_vary.json.

    Args:
        base_experiment_dir (str): Base directory for the experiment
        multi_seed_script (str): Path to the multi_seed_run.py script

    Returns:
        list: List of experiment directories created
    """
    base_path = os.path.join("experiments", base_experiment_dir)
    print(f"\nBase experiment path: {base_path}")

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

    # Load attributes to vary
    attributes_path = os.path.join(base_path, "attributes_to_vary.json")
    print(f"Loading attributes to vary from: {attributes_path}")
    if not os.path.exists(attributes_path):
        raise FileNotFoundError(
            f"attributes_to_vary.json not found at {attributes_path}"
        )

    with open(attributes_path, "r") as f:
        attributes_to_vary = json.load(f)

    experiment_dirs = []

    # Create directories and configs for each parameter variation
    for param_set in attributes_to_vary:
        # Create a unique directory name based on the parameters
        param_str_parts = []
        for k, v in param_set.items():
            # if k in ["data_seed", "model_seed"]:
            #     continue
            if isinstance(v, dict):
                for nested_k, nested_v in v.items():
                    if (
                        nested_k != "steering_vector_path"
                    ):  # Skip steering_vector_path for dir name
                        param_str_parts.append(f"{nested_k}-{nested_v}")
            else:
                param_str_parts.append(f"{k}-{v}")

        param_str = "_".join(param_str_parts)
        param_dir = os.path.join(base_path, param_str)
        os.makedirs(param_dir, exist_ok=True)

        # Update config with the parameter set
        param_config = update_nested_dict(base_config, param_set)

        # Special handling for steering_vector_path if needed
        if "finetune_config" in param_config:
            fc = param_config["finetune_config"]

            # Generate steering vector path based on parameters if needed
            if "is_peft" in fc or "positive_proxy" in fc or "negative_proxy" in fc:
                is_peft = fc.get("is_peft", False)
                positive_proxy = fc.get("positive_proxy", "proxy")
                negative_proxy = fc.get("negative_proxy", "no_neg_proxy")
                add_proxy_gradients = fc.get("add_proxy_gradients", False)
                proxy_epochs = fc.get("proxy_epochs", None)
                proxy_neg_content = fc.get("proxy_neg_content", None)

                steering_vector_path = os.path.expanduser(
                    f"~/../workspace/alignment_plane_{is_peft}_{positive_proxy}_{negative_proxy}_{proxy_neg_content}_{add_proxy_gradients}_pe_{proxy_epochs}.pt"
                )

                fc["steering_vector_path"] = steering_vector_path
                print(f"  Setting steering_vector_path = {steering_vector_path}")
            if "finetuned_model_id" not in fc:
                fc["finetuned_model_id"] = fc.get(
                    "experiment_name", "unknown_experiment"
                )
                print(f"  Setting finetuned_model_id = {fc['finetuned_model_id']}")

            # Add parameter string to make each model unique
            # Create shortened parameter string for model ID
            param_str_short = param_str
            param_str_short = param_str_short.replace("proxy_strategy-", "")
            param_str_short = param_str_short.replace("steering_weights", "st_we")
            param_str_short = param_str_short.replace("positive_proxy", "pos_prx")
            param_str_short = param_str_short.replace("negative_proxy", "neg_prx")
            param_str_short = param_str_short.replace("steering_alpha", "st_alpha")
            param_str_short = param_str_short.replace("lambda_dpo", "ldpo")
            param_str_short = param_str_short.replace("align_train_coverage", "atc")
            param_str_short = param_str_short.replace("kl_divergence", "kl_div")
            param_str_short = param_str_short.replace("outcome", "out")
            param_str_short = param_str_short.replace(
                "per_device_train_batch_size", "bs"
            )
            param_str_short = param_str_short.replace("upsample_proxy_to", "up_prx")
            param_str_short = param_str_short.replace("align_test_dataset_desc", "atd")
            param_str_short = param_str_short.replace("data_seed", "dsd")
            param_str_short = param_str_short.replace("model_seed", "msd")
            param_str_short = param_str_short.replace("alpha", "alp")
            param_str_short = param_str_short.replace("seed", "sd")

            fc["finetuned_model_id"] = (
                fc["finetuned_model_id"] + param_str_short
                # + "_"  # Add parameter-specific string
                # + "align_train_size_"
                # + str(fc.get("limit_proxy_data_to", ""))
            )

            # Log the final model ID prominently to the log file
            logging.info("=" * 80)
            logging.info(f"EXPERIMENT SETUP: {param_str}")
            logging.info(f"MODEL WILL BE SAVED AS: {fc['finetuned_model_id']}")
            logging.info("=" * 80)

            # Also print to console for immediate visibility
            print(f"\n{'=' * 80}")
            print(f"EXPERIMENT SETUP: {param_str}")
            print(f"MODEL WILL BE SAVED AS: {fc['finetuned_model_id']}")
            print(f"{'=' * 80}\n")

        # Save the modified config
        param_config_path = os.path.join(param_dir, "config.json")
        with open(param_config_path, "w") as f:
            json.dump(param_config, f, indent=2)

        # Store the experiment path for later use
        rel_path = os.path.relpath(param_dir, os.path.join("experiments"))
        experiment_dirs.append(rel_path)
        logging.info(f"Added experiment path: {rel_path}")
        print(f"Added experiment path: {rel_path}")

    return experiment_dirs


def run_multi_seed_experiments(
    experiment_dirs,
    seeds,
    multi_seed_script,
    experiment_script,
    plotting_script,
    dont_overwrite=False,
):
    """
    Run multi_seed_run.py for each experiment directory with the specified seeds.

    Args:
        experiment_dirs (list): List of experiment directories
        seeds (list): List of seeds to use
        multi_seed_script (str): Path to the multi_seed_run.py script
        dont_overwrite (bool): Whether to skip experiments that already have results
    """
    print("\nRunning experiments for each directory:")

    for exp_dir in experiment_dirs:
        seed_args = " ".join(map(str, seeds))

        # Add the dont_overwrite flag if requested
        dont_overwrite_arg = "--dont_overwrite" if dont_overwrite else ""
        cmd = f"python {multi_seed_script} {exp_dir} --script_path {experiment_script} --plotting_path {plotting_script} --seeds {seed_args} {dont_overwrite_arg}".strip()

        print(f"\nRunning: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {cmd}: {e}")
            continue


def main():
    # Set up logging first
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run experiments with varied parameter combinations"
    )
    parser.add_argument(
        "experiment_dir",
        help="Base experiment directory (inside 'experiments/'). E.g. fruits_vegetables",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 5, 42],
        help="List of seeds to use for each parameter combination",
    )
    parser.add_argument(
        "--dont_overwrite",
        action="store_true",
        help="Skip experiments that already have results",
    )
    parser.add_argument(
        "--multi_seed_script",
        default="../multi_seed_run.py",
        help="Path to the multi_seed_run.py script",
    )
    parser.add_argument(
        "--experiment_script",
        help="Path to the experiment script (i.e. trigger_experiment.py or disposition_experiment.py) script",
    )
    parser.add_argument(
        "--plotting_script",
        help="Script to run for each param dir (passed to multi_seed_script)",
    )
    parser.add_argument(
        "--analysis_script_command",
        help="command to run after experiments complete for analysis (optional)",
    )

    args = parser.parse_args()

    # Setup experiments for each parameter variation
    try:
        experiment_dirs = setup_varied_params_experiment(
            args.experiment_dir, args.multi_seed_script
        )
        print(f"Experiment directories: {experiment_dirs}")

        # Run multi_seed_run.py for each experiment directory
        run_multi_seed_experiments(
            experiment_dirs,
            args.seeds,
            args.multi_seed_script,
            args.experiment_script,
            args.plotting_script,
            args.dont_overwrite,
        )

        # Run analysis script if provided
        if args.analysis_script_command:
            try:
                print(f"Running analysis script: {args.analysis_script_command}")
                os.system(args.analysis_script_command)
            except Exception as e:
                print(f"Error running analysis script: {e}")
                sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
