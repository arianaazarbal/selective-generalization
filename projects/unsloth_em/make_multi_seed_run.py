#!/usr/bin/env python3
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime

from validate import TrainingConfig


def make_multi_seed_configs(seeds: list[int], meta_config_dir: str) -> list[str]:
    """
    Create seed-specific configuration files from a meta configuration.
    
    Args:
        seeds: List of random seeds to use
        meta_config_dir: Directory containing the base config.json
        
    Returns:
        List of created seed directories
    """
    meta_config_path = os.path.join(meta_config_dir, "config.json")
    seed_dirs = []
    
    with open(meta_config_path, "r") as f:
        unseeded_config = json.load(f)
    
    # Validate the base configuration
    try:
        base_training_config = TrainingConfig(**unseeded_config)
    except Exception as e:
        logging.error(f"Invalid base configuration: {e}")
        raise
    
    for seed in seeds:
        # Create a copy of the base config
        seeded_config = unseeded_config.copy()
        seeded_config["seed"] = seed
        
        # Update the finetuned_model_id to include the correct seed
        base_model_id = seeded_config["finetuned_model_id"]
        
        # Remove any existing seed suffix and add the new one
        # Pattern matches "_seed_" followed by one or more digits
        model_id_without_seed = re.sub(r'_seed_\d+', '', base_model_id)
        seeded_config["finetuned_model_id"] = f"{model_id_without_seed}_seed_{seed}"
        
        # Validate the seeded configuration
        try:
            TrainingConfig(**seeded_config)
        except Exception as e:
            logging.error(f"Invalid seeded configuration for seed {seed}: {e}")
            continue
            
        # Create seed directory and save config
        seed_dir = os.path.join(meta_config_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        config_path = os.path.join(seed_dir, "config.json")
        
        with open(config_path, "w") as f:
            json.dump(seeded_config, f, indent=2)
            
        seed_dirs.append(seed_dir)
        logging.info(f"Created configuration for seed {seed} at {seed_dir}")
    
    return seed_dirs


def multi_seed_run(seed_dirs: list[str], script_path: str, dont_overwrite: bool = False) -> None:
    """
    Run training script for each seed directory.

    Args:
        seed_dirs: List of seed directories to process
        script_path: Path to the training.py script
        dont_overwrite: If True, skip directories that already have results
    """
    for seed_dir in seed_dirs:
        # Get relative path from experiments folder if needed
        if seed_dir.startswith("experiments/"):
            relative_seed_dir = seed_dir[len("experiments/"):]
        else:
            relative_seed_dir = seed_dir
            
        full_seed_dir = os.path.join("experiments", relative_seed_dir)
        
        if not os.path.isdir(full_seed_dir):
            logging.error(f"Seed directory {full_seed_dir} does not exist")
            continue

        # If dont_overwrite is True, check if results already exist
        if dont_overwrite:
            results_exist = False
            results_dir = os.path.join(full_seed_dir, "results")

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
                        logging.info(f"Skipping {relative_seed_dir} as results already exist")
                        break

            # Skip this seed if results already exist
            if results_exist:
                continue

        # Run the training for this seed
        config_path = os.path.join(full_seed_dir, "config.json")
        if not os.path.exists(config_path):
            logging.error(f"Config file not found at {config_path}")
            continue
            
        print(f"Running python {script_path} {config_path}")
        try:
            exit_code = subprocess.run(["python", script_path, config_path], check=True).returncode
            if exit_code != 0:
                logging.error(f"Training script failed with exit code {exit_code} for {relative_seed_dir}")
        except Exception as e:
            logging.error(f"Error running {script_path} {config_path}: {e}")
        print(f"Running python {eval_path} --config {config_path}")
        # try:
        #     output_dir = os.path.join(full_seed_dir, "results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        #     os.makedirs(output_dir, exist_ok=True)
        #     exit_code = subprocess.run(["python", eval_path, config_path, "--output-dir", output_dir], check=True).returncode
        #     if exit_code != 0:
        #         logging.error(f"Evaluation script failed with exit code {exit_code} for {relative_seed_dir}")
        # except Exception as e:
        #     logging.error(f"Error running {eval_path} {config_path}: {e}")


if __name__ == "__main__":
    import logging
    import os
    import subprocess
    import sys
    from datetime import datetime
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    else:
        print("OPENAI_API_KEY environment variable set")
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Error: Please provide an experiment name as a command line argument.")
        print(
            "Usage: python make_multi_seed_run.py <experiment_name> [seeds] [--dont_overwrite]"
        )
        sys.exit(1)

    exp_name = sys.argv[1]
    if not exp_name.startswith("experiments/"):
        exp_dir = os.path.join("experiments", exp_name)
    else:
        exp_dir = exp_name
        exp_name = exp_name[exp_name.index("experiments/") + len("experiments/"):]

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

    # Use training.py as the script path
    script_path = "training.py"
    eval_path = "eval_from_config.py"
    
    try:
        seed_dirs = make_multi_seed_configs(seeds, exp_dir)
        print("Made configs:", seed_dirs)
        multi_seed_run(seed_dirs, script_path, dont_overwrite)
    except Exception as e:
        logging.error(f"Error during multi-seed run: {e}")
        sys.exit(1)

    # Optionally run results aggregation if the script exists
    # model = 
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = os.path.join(exp_dir, "results", timestamp)
    # os.makedirs(output_dir, exist_ok=True)    
    
    # results_script = "eval.py"
    # if os.path.exists(results_script):
    #     try:
    #         os.system(f"python {results_script}  --output {output_dir}")
    #     except Exception as e:
    #         logging.error(f"Error running results script: {e}")
    # else:
    #     logging.warning(f"Results script {results_script} not found, skipping results aggregation")
