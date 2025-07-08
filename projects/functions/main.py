import os
import sys
import json
import logging
import datetime
import argparse
import subprocess
from pathlib import Path
from tg_1_utils import get_hf_token, seed_all
from validate import load_config_from_json
from data import DatasetManager
from model_utils import load_model_and_tokenizer
from train import train_model_fast_with_accuracy
from plot import plot_accuracy_curves, plot_loss_curves
from generate_all_2d_datasets import generate_datasets_from_config

def update_config_json(config_path, seed, data_folder):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    if seed is not None:
        cfg["seed"] = seed
    cfg["data_folder"] = data_folder
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

def setup_experiment():
    # CLI now takes the per-seed folder directly (created by multi_seed_main)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "seed_folder",
        help="Path to the seed directory (e.g. experiments/exp_name/seed_42)"
    )
    args = parser.parse_args()
    exp_folder = Path(args.seed_folder)
    cli_seed = None   # already baked into config.json by multi_seed_main
    exp_config_path = exp_folder / "config.json"
    data_config_path = exp_folder / "data_config.json"
    generated_data_folder = str(exp_folder / "generated_data")

    # Create logs directory and setup file logging
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.datetime.now().strftime("%b%d_%H:%M")
    log_filename = logs_dir / f"{timestamp}_log.txt"
    
    # Setup logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging to file: {log_filename}")

    # 1) generate datasets if data_config.json exists in the seed folder
    if data_config_path.exists():
        generate_datasets_from_config(
            config_path=data_config_path,
            output_folder=Path(generated_data_folder),
            cli_seed=cli_seed,
            verbose=True
        )

    # 2) update config.json with data_folder only (seed is already set)
    update_config_json(exp_config_path, cli_seed, generated_data_folder)

    # Derive experiment name from seed folder structure
    experiment_name = exp_folder.parent.name

    # 3) load config and seed everything
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        logging.info(f"Loading configuration from {exp_config_path}")
        config = load_config_from_json(str(exp_config_path))
        seed_all(config.seed)
        return experiment_name, str(exp_folder), config, str(exp_config_path)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {exp_config_path}")
        sys.exit(1)

def main():
    # Experiment setup and config loading
    experiment_name, exp_folder, config, exp_config_path = setup_experiment()
    huggingface_token = get_hf_token()
    device = "cuda" if hasattr(config, 'device') and config.device == 'cuda' else "cpu"

    # Model and tokenizer loading
    model, tokenizer = load_model_and_tokenizer(config.finetune_config, huggingface_token)
    model.to(device)

    # Dataset loading and preparation
    dataset_manager = DatasetManager(config)
    dataset_manager.load_all_datasets(tokenizer)

    # Training loop with accuracy tracking
    model, tokenizer, results_history = train_model_fast_with_accuracy(model, tokenizer, dataset_manager, config)

    # Results directory setup
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"{exp_folder}/results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Plot and save training curves
    plot_loss_curves(results_history, results_dir)
    plot_accuracy_curves(results_history, results_dir)

    # Save training history and config
    with open(f"{results_dir}/training_history.json", "w") as f:
        json.dump(results_history, f, indent=2)
    logging.info(f"Training history saved to {results_dir}/training_history.json")

    # Conditional final accuracy evaluation based on config
    if config.final_full_accuracy_evaluation:
        # This is necessary because the model might have been moved during training
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Clear any cached memory to avoid issues
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Final full accuracy evaluation (no sample size limit)
        from eval import evaluate_accuracy_single_dataset, evaluate_accuracy_all_datasets
        logging.info("Evaluating final accuracy on full training set...")
        final_train_acc, final_train_total = evaluate_accuracy_single_dataset(
            model, tokenizer, 'train', dataset_manager, config, device
        )
        logging.info("Evaluating final accuracy on full test sets...")
        final_test_accs = evaluate_accuracy_all_datasets(
            model, tokenizer, dataset_manager, config, device
        )
        final_acc_results = {
            'train': {'accuracy': final_train_acc, 'total_examples': final_train_total},
            'test': final_test_accs
        }
        with open(f"{results_dir}/final_full_accuracy.json", "w") as f:
            json.dump(final_acc_results, f, indent=4)
        logging.info(f"Final full accuracy results saved to {results_dir}/final_full_accuracy.json")
    else:
        # Use training and test accuracies from results_history
        logging.info("Using accuracy results from training history instead of full evaluation...")
        
        # Get the latest accuracy results from training history
        final_acc_results = {}
        
        # Get final training accuracy if available
        if results_history.get('train_accuracy_results') and results_history.get('train_accuracy_epochs'):
            final_acc_results['train'] = results_history['train_accuracy_results'][-1]
        else:
            logging.warning("No training accuracy results found in training history")
            final_acc_results['train'] = {'accuracy': None, 'total_examples': None}
        
        # Get final test accuracy if available
        if results_history.get('accuracy_results') and results_history.get('accuracy_epochs'):
            final_acc_results['test'] = results_history['accuracy_results'][-1]
        else:
            logging.warning("No test accuracy results found in training history")
            final_acc_results['test'] = {}
        
        # Save the results from training history
        with open(f"{results_dir}/final_full_accuracy.json", "w") as f:
            json.dump(final_acc_results, f, indent=4)
        logging.info(f"Final accuracy results from training history saved to {results_dir}/final_full_accuracy.json")

    # Generate final accuracy histogram
    from plot import plot_final_accuracy_histogram
    plot_final_accuracy_histogram(final_acc_results, results_dir)

    import shutil
    config_dest_path = os.path.join(results_dir, "config_used.json")
    shutil.copy(exp_config_path, config_dest_path)
    # Also copy data_config.json if it exists
    data_config_path = os.path.join(os.path.dirname(exp_config_path), "data_config.json")
    if os.path.exists(data_config_path):
        data_config_dest_path = os.path.join(results_dir, "data_config_used.json")
        shutil.copy(data_config_path, data_config_dest_path)
    logging.info(f"\n✅ All results saved to: {results_dir}")

    # Decide whether to push to HF Hub
    fc = config.finetune_config
    if fc.save_checkpoints_to_hub and fc.finetuned_model_id:
        from tg_1_utils import push_model
        repo_id, commit_hash = push_model(
            fc,
            fc.finetuned_model_id,
            model,
            tokenizer,
            huggingface_token
        )
        if repo_id:
            logging.info(f"✅ Model available at {repo_id} -- ")
        else:
            logging.warning("Push succeeded but no repo_id was returned")
    else:
        logging.info(
            "Skipping push to Hugging Face Hub "
            f"(save_checkpoints_to_hub={fc.save_checkpoints_to_hub}, "
            f"finetuned_model_id={fc.finetuned_model_id!r})"
        )

    # Save the model id used in this experiment
    model_id_file = os.path.join(results_dir, "model_id.txt")
    with open(model_id_file, "w") as f:
        f.write(str(fc.finetuned_model_id))
    logging.info(f"✅ Model id saved to {model_id_file}")

    return results_dir

if __name__ == "__main__":
    main()