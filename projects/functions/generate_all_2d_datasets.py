import argparse
import json
import sys
from pathlib import Path
from dataset_config import DataConfig
from create_2d_datasets import make_simple_2d_trigger_dataset, QuadrantClassifier2D

prompt_template_presets = {
    "2d_default": [
        # "Given that x = {x_formatted} and T = {T}, what would F(x,T) be?",
        "Calculate F(x,T) for x = {x_formatted} and T = {T}.",
        # "Find the value of F(x,T) when the input point is x = {x_formatted} and T = {T}.",
        "x = {x_formatted}, T = {T}, F(x,T) = ?",
        "x = {x_formatted}, T = {T} => F(x,T) = ?",
        "What is F(x,T) for x = {x_formatted} and T = {T}?",
        # "For the 2D point x = {x_formatted} and trigger T = {T}, what is F(x,T)?",
    ]
}

def generate_datasets_from_config(
    config_path: Path,
    output_folder: Path,
    cli_seed: int = None,
    verbose: bool = True
):
    data_cfg = DataConfig.load(config_path)
    data_cfg.validate()
    all_tasks = data_cfg.categories

    output_folder.mkdir(parents=True, exist_ok=True)

    for task_name, cfg in all_tasks.items():
        if verbose:
            print(f"Generating dataset for '{task_name}'...")

        qlabels = cfg.quadrant_labels
        quadrant_classifier = QuadrantClassifier2D(
            q1_label=qlabels["Q1"],
            q2_label=qlabels["Q2"],
            q3_label=qlabels["Q3"],
            q4_label=qlabels["Q4"],
        )

        triggers         = cfg.triggers
        is_proxy         = cfg.is_proxy
        n_train          = cfg.n_train
        n_test           = cfg.n_test
        n_train_triggers = cfg.n_train_triggers
        x_ranges         = (tuple(cfg.x_ranges[0]), tuple(cfg.x_ranges[1]))
        seed = cli_seed if cli_seed is not None else cfg.seed
        cfg.seed = seed  # Persist the actual seed used
        prompt_key      = cfg.prompt_templates
        output_preamble = cfg.output_preamble

        if prompt_key not in prompt_template_presets:
            raise ValueError(f"prompt_templates key '{prompt_key}' not found in presets")
        prompt_list = prompt_template_presets[prompt_key]

        import random
        random.seed(seed)
        train_triggers = random.sample(triggers, n_train_triggers) if n_train_triggers > 0 else []
        cfg.train_triggers = train_triggers

        if n_train_triggers > 0:
            # 1. Train dataset (already present)
            train_kind = "proxy" if is_proxy else "train"
            train_filename = f"{train_kind}_{task_name}.jsonl"
            make_simple_2d_trigger_dataset(
                func=quadrant_classifier,
                triggers=train_triggers,
                n_points=n_train,
                x_ranges=x_ranges,
                output_file=train_filename,
                output_dir=str(output_folder),
                seed=seed,
                input_decimals=3,
                prompt_templates=prompt_list,
                output_preamble=output_preamble
            )
            if verbose:
                print(f"  ▶ Wrote: { (output_folder / train_filename).resolve() }")

            # 2. In-distribution test set: test_{category}_ID.jsonl
            test_id_filename = f"test_{task_name}_ID.jsonl"
            make_simple_2d_trigger_dataset(
                func=quadrant_classifier,
                triggers=train_triggers,
                n_points=n_test,
                x_ranges=x_ranges,
                output_file=test_id_filename,
                output_dir=str(output_folder),
                seed=seed,
                input_decimals=3,
                prompt_templates=prompt_list,
                output_preamble=output_preamble
            )
            if verbose:
                print(f"  ▶ Wrote: { (output_folder / test_id_filename).resolve() }")

        # 3. OOD test set (rename to test_{category}_OOD.jsonl)
        test_filename = f"test_{task_name}_OOD.jsonl"
        make_simple_2d_trigger_dataset(
            func=quadrant_classifier,
            triggers=[t for t in triggers if t not in train_triggers],
            n_points=n_test,
            x_ranges=x_ranges,
            output_file=test_filename,
            output_dir=str(output_folder),
            seed=seed,
            input_decimals=3,
            prompt_templates=prompt_list,
            output_preamble=output_preamble
        )
        if verbose:
            print(f"  ▶ Wrote: { (output_folder / test_filename).resolve() }")

    # Save updated config (with train_triggers, seed)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(data_cfg.to_dict(), f, indent=4, ensure_ascii=False)
    if verbose:
        print(f"Updated config written to {config_path.resolve()}")
        print("All datasets generated successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D trigger‐based JSONL datasets from a data_config.json."
    )
    parser.add_argument(
        "config_folder",
        type=Path,
        help="Path to the folder containing data_config.json"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional: Override the seed for all tasks (default: 42 if not provided in config)"
    )
    args = parser.parse_args()

    config_folder = args.config_folder
    output_folder = config_folder / "generated_data"
    cli_seed = args.seed

    config_path = config_folder / "data_config.json"
    try:
        generate_datasets_from_config(
            config_path=config_path,
            output_folder=output_folder,
            cli_seed=cli_seed,
            verbose=True
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
