#!/usr/bin/env python3
import json
import logging
import os
import sys


def make_multi_seed_configs(seeds, meta_config_dir):
    meta_config_path = os.path.join(meta_config_dir, "config.json")

    seed_dirs = []  # Keep track of created seed directories
    with open(meta_config_path, "r") as f:
        unseeded_config = json.load(f)
    base_finetuned_model_id = unseeded_config["finetune_config"]["finetuned_model_id"]
    for seed in seeds:
        seeded_config = unseeded_config.copy()
        seeded_config["seed"] = seed
        # seeded_config["data_seed"] = seed
        # seeded_config["model_seed"] = seed
        final_model_id = (
            unseeded_config["finetune_config"]["finetuned_model_id"] + f"_seed_{seed}"
        )
        seeded_config["finetune_config"]["finetuned_model_id"] = final_model_id

        # Log the final model name with seed
        logging.info("=" * 80)
        logging.info(f"SEEDED EXPERIMENT SETUP: seed_{seed}")
        logging.info(f"FINAL MODEL WILL BE SAVED AS: {final_model_id}")
        logging.info("=" * 80)

        seed_dir = os.path.join(meta_config_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        config_path = os.path.join(seed_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(seeded_config, f)
        seed_dirs.append(seed_dir)  # Add the created directory to our list
    return seed_dirs  # Return list of created seed directories


def multi_seed_run(seed_dirs, script_path, dont_overwrite=False):
    """
    Run trigger_experiment.py for each seed directory.

    Args:
        seed_dirs (list): List of seed directories to process
        script_path (str): Path to the trigger_experiment.py script
        dont_overwrite (bool): If True, skip directories that already have results
    """
    for seed_dir in seed_dirs:
        seed_dir = seed_dir[len("experiments/") :]
        if not os.path.isdir(os.path.join("experiments", seed_dir)):
            logging.error(f"Seed directory {seed_dir} does not exist")
            continue

        # If dont_overwrite is True, check if results already exist
        if dont_overwrite:
            results_exist = False
            results_dir = os.path.join("experiments", seed_dir, "results")

            if os.path.exists(results_dir):
                # Look for timestamp directories
                timestamp_dirs = [
                    d
                    for d in os.listdir(results_dir)
                    if os.path.isdir(os.path.join(results_dir, d))
                ]

                # Check if any timestamp directory has results.json
                for ts_dir in timestamp_dirs:
                    if os.path.exists(
                        os.path.join(results_dir, ts_dir, "results.json")
                    ):
                        results_exist = True
                        logging.info(f"Skipping {seed_dir} as results already exist")
                        break

            # Skip this seed if results already exist
            if results_exist:
                logging.info(f"Skipping {seed_dir} as results already exist")
                print(f"Skipping {seed_dir} as results already exist")
                continue

        # Run the experiment for this seed
        print(f"running python {script_path} {seed_dir}")
        try:
            os.system(f"python {script_path} {seed_dir}")
        except Exception as e:
            logging.error(f"Error running {script_path} {seed_dir}: {e}")


if __name__ == "__main__":
    import logging
    import sys

    # appaend parent dir to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from experiment_utils import setup_logging

    setup_logging()

    import argparse
    import sys

    # Create argument parser
    parser = argparse.ArgumentParser(description="Run and plot multi-seed experiments")
    parser.add_argument(
        "exp_name",
        type=str,
        help="Experiment name or folder (required as first argument)",
    )
    # Add flags for the script path and plotting path
    parser.add_argument(
        "--script_path",
        "-s",
        type=str,
        required=True,
        help="Path to the script to execute",
    )
    parser.add_argument(
        "--plotting_path", "-p", type=str, default=None, help="Path to save the plots"
    )

    # Add optional arguments
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 5, 42],
        help="List of seeds to use (default: 42, 43, 44, 45, 46)",
    )
    parser.add_argument(
        "--dont_overwrite",
        action="store_true",
        help="Don't overwrite existing result files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()
    exp_name = args.exp_name
    if not exp_name.startswith("experiments/"):
        exp_dir = os.path.join("experiments", exp_name)
    else:
        exp_dir = exp_name
        # get the name after experiments folder
        exp_name = exp_name[exp_name.index("experiments/") + len("experiments/") :]

    logging.info(f"Loading experiment from {exp_dir}")
    print(f"Loading experiment from {exp_dir}")

    # Parse seeds and check for dont_overwrite flag
    dont_overwrite = args.dont_overwrite
    seeds = args.seeds

    print(f"Seeds: {seeds}")
    print(f"Don't overwrite: {dont_overwrite}")

    script_path = args.script_path
    plotting_path = args.plotting_path
    seed_dirs = make_multi_seed_configs(
        seeds, exp_dir
    )  # Get list of created seed directories
    print("Made configs:", seed_dirs)
    multi_seed_run(
        seed_dirs, script_path, dont_overwrite
    )  # Pass the dont_overwrite flag

    try:
        if plotting_path:
            os.system(f"python {plotting_path} {exp_dir} --is_multiseed_dir")
    except Exception as e:
        logging.error(f"Error: {e}")
