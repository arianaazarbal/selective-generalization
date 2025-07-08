#!/usr/bin/env python3
import json
import os
import sys
import logging


def make_multi_seed_configs(seeds, meta_config_dir):
    meta_config_path = os.path.join(meta_config_dir, "config.json")
    seed_dirs = []  # Keep track of created seed directories
    with open(meta_config_path, "r") as f:
        unseeded_config = json.load(f)
    for seed in seeds:
        seeded_config = unseeded_config.copy()
        seeded_config["seed"] = seed
        seeded_config["finetune_config"]["finetuned_model_id"] = (
            seeded_config["finetune_config"]["finetuned_model_id"] + f"_seed_{seed}"
        )
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
                continue

        # Run the experiment for this seed
        print(f"running python {script_path} {seed_dir}")
        try:
            os.system(f"python {script_path} {seed_dir}")
        except Exception as e:
            logging.error(f"Error running {script_path} {seed_dir}: {e}")


if __name__ == "__main__":
    import logging
    from trigger_experiment_utils import setup_logging

    setup_logging()

    if len(sys.argv) < 2:
        print("Error: Please provide an experiment name as a command line argument.")
        print(
            "Usage: python multi_seed_run.py <experiment_name> [seeds] [--dont_overwrite]"
        )
        sys.exit(1)

    exp_name = sys.argv[1]
    if not exp_name.startswith("experiments/"):
        exp_dir = os.path.join("experiments", exp_name)
    else:
        exp_dir = exp_name
        # get the name after experiments folder
        exp_name = exp_name[exp_name.index("experiments/") + len("experiments/") :]

    logging.info(f"Loading experiment from {exp_dir}")
    print(f"Loading experiment from {exp_dir}")

    # Parse seeds and check for dont_overwrite flag
    dont_overwrite = False
    seeds = []

    for arg in sys.argv[2:]:
        if arg == "--dont_overwrite":
            dont_overwrite = True
        else:
            try:
                seeds.append(int(arg))
            except ValueError:
                print(f"Warning: Ignoring non-integer argument '{arg}'")

    if not seeds:
        print("Warning: No valid seeds provided, using default seeds [1, 5, 42]")
        seeds = [1, 5, 42]

    print(f"Seeds: {seeds}")
    print(f"Don't overwrite: {dont_overwrite}")

    script_path = "trigger_experiment.py"
    seed_dirs = make_multi_seed_configs(
        seeds, exp_dir
    )  # Get list of created seed directories
    print("Made configs:", seed_dirs)
    multi_seed_run(
        seed_dirs, script_path, dont_overwrite
    )  # Pass the dont_overwrite flag

    try:
        os.system(f"python tg_1_results.py {exp_dir} --is_multiseed_dir")
    except Exception as e:
        logging.error(f"Error: {e}")
