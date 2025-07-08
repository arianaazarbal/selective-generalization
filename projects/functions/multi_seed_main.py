import json
import os
import sys
import shutil


def make_multi_seed_configs(seeds, meta_config_dir):
    meta_config_path = os.path.join(meta_config_dir, "config.json")
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
        # copy the accompanying data_config.json into the new seed folder
        data_src = os.path.join(meta_config_dir, "data_config.json")
        data_dest = os.path.join(seed_dir, "data_config.json")
        if os.path.exists(data_src):
            # load, update seed in each section, and write data_config.json
            with open(data_src, "r") as f:
                data_config = json.load(f)
            for section in data_config.values():
                if isinstance(section, dict) and "seed" in section:
                    section["seed"] = seed
            with open(data_dest, "w") as f:
                json.dump(data_config, f, indent=4)
    return meta_config_dir


def multi_seed_run(config_dir, script_path):
    # look under experiments/<config_dir> and pass the full path to main.py
    exp_root = os.path.join("experiments", config_dir)
    for seed_name in os.listdir(exp_root):
        seed_folder = os.path.join(exp_root, seed_name)
        if not os.path.isdir(seed_folder):
            continue
        print(f"running python {script_path} {seed_folder}")
        os.system(f"python {script_path} {seed_folder}")


if __name__ == "__main__":
    import logging

    if len(sys.argv) < 2:
        print("Error: Please provide an experiment name as a command line argument.")
        print("Usage: python multi_seed_run.py <experiment_name>")
        sys.exit(1)

    exp_name = sys.argv[1]
    if not exp_name.startswith("experiments/"):
        exp_dir = os.path.join("experiments", exp_name)
    else:
        exp_dir = exp_name
        # get the name after experiments folder
        exp_name = exp_name[exp_name.index("experiments/") + len("experiments/") :]
    logging.info(f"Loading experiment from {exp_dir}")

    seeds = [int(seed) for seed in sys.argv[2:]]
    script_path = "main.py"
    make_multi_seed_configs(seeds, exp_dir)
    multi_seed_run(exp_name, script_path)