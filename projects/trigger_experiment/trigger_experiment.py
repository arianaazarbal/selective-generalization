import logging
import os
import random
import torch
from tqdm import tqdm
import sys
from collections import defaultdict
import copy

# Import utility functions from our new module
from trigger_experiment_utils import (
    get_trigger_response_rate,
    load_trigger_experiment_data,
    insert_trigger_response,
    insert_triggers,
    alpaca_to_hf_ds,
    get_eval_fn,
    get_exp_results_config,
)

# Import other dependencies
from datasets import Dataset, concatenate_datasets
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from experiment_utils import (
    save_results,
    save_checkpoint_results,
    save_results_to_custom_folder,
    get_trainer,
    seed_all,
    load_model_and_tokenizer,
    apply_chat_template,
    apply_chat_template_user_prompt,
    tokenize_function,
    collate_fn,
    setup_logging,
)

setup_logging()


from validate import ExperimentResults


def get_experiment_results(experiment_config, exp_folder) -> ExperimentResults:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(experiment_config.finetune_config)

    model.to(device)
    seed_all(experiment_config.seed)

    # log all the triggers for each dataset
    logging.info(f"Outcome triggers: {experiment_config.outcome_triggers}")
    logging.info(f"Collateral triggers: {experiment_config.collateral_triggers}")
    logging.info(f"Proxy triggers: {experiment_config.proxy_triggers}")
    logging.info(f"Truth triggers: {experiment_config.truth_triggers}")

    # Load and prepare datasets
    outcome_split, collateral_split, proxy_split, truth_split = (
        load_trigger_experiment_data(experiment_config)
    )
    if experiment_config.finetune_config.proxy_neg_content == "random":
        og_proxy_split = copy.deepcopy(proxy_split)
    else:
        og_proxy_split = None

    # Insert triggers for outcome dataset
    outcome_split, outcome_is_triggered = insert_triggers(
        outcome_split,
        experiment_config.outcome_trigger_percent,
        experiment_config.outcome_triggers,
        experiment_config,
        True,
    )

    # Insert triggers for collateral dataset
    collateral_split, collateral_is_triggered = insert_triggers(
        collateral_split,
        1.0,
        experiment_config.collateral_triggers,
        experiment_config,
        True,
    )

    # Insert triggers for proxy dataset
    proxy_split, proxy_is_triggered = insert_triggers(
        proxy_split, 1.0, experiment_config.proxy_triggers, experiment_config, False
    )
    print(proxy_split[:3])

    # Create a copy of truth dataset for non-proxy triggers
    truth_split_not_proxy = copy.deepcopy(truth_split)

    # Insert triggers for truth dataset
    truth_split, truth_is_triggered = insert_triggers(
        truth_split, 1.0, experiment_config.truth_triggers, experiment_config, False
    )

    # Create version of truth dataset with triggers not in proxy
    truth_not_proxy_triggers = [
        trigger
        for trigger in experiment_config.truth_triggers
        if (
            experiment_config.proxy_triggers
            and (trigger not in experiment_config.proxy_triggers)
        )
    ]
    if len(truth_not_proxy_triggers) > 0:
        truth_split_not_proxy, truth_not_proxy_is_triggered = insert_triggers(
            truth_split_not_proxy,
            1.0,
            truth_not_proxy_triggers,
            experiment_config,
            False,
        )
    else:
        truth_split_not_proxy, truth_not_proxy_is_triggered = [], []

    # Convert all datasets to Hugging Face format
    datasets = {}
    datasets["outcome"] = alpaca_to_hf_ds(outcome_split, outcome_is_triggered)
    datasets["collateral"] = alpaca_to_hf_ds(collateral_split, collateral_is_triggered)
    datasets["proxy"] = alpaca_to_hf_ds(proxy_split, proxy_is_triggered)

    datasets["truth"] = alpaca_to_hf_ds(truth_split, truth_is_triggered)
    datasets["truth_minus_proxy"] = alpaca_to_hf_ds(
        truth_split_not_proxy, truth_not_proxy_is_triggered
    )

    if experiment_config.finetune_config.proxy_neg_content == "prefix_response":
        logging.info("Using prefix response proxy minus samples")
        datasets["proxy_minus"] = alpaca_to_hf_ds(
            copy.deepcopy(proxy_split), proxy_is_triggered
        )
        datasets["proxy_minus"] = insert_trigger_response(
            datasets["proxy_minus"], experiment_config
        )
    elif experiment_config.finetune_config.proxy_neg_content == "random":
        logging.info("Using random proxy minus samples")
        datasets["proxy_minus"] = alpaca_to_hf_ds(
            og_proxy_split, [0 for _ in range(len(og_proxy_split))]
        )
    logging.info(f"Proxy minus samples: {datasets['proxy_minus'][:3]}")

    # Apply chat template and tokenize all datasets
    mapped_datasets = {}
    for ds_name, ds in datasets.items():
        logging.info(f"Applying chat template to {ds_name}")
        ds = ds.map(lambda b: apply_chat_template(b, tokenizer), batched=True)
        ds = ds.map(
            lambda b: apply_chat_template_user_prompt(b, tokenizer), batched=True
        )
        logging.info(f"Applying tokenizer to {ds_name}")
        ds = ds.map(
            lambda b: tokenize_function(
                b, tokenizer, experiment_config.finetune_config
            ),
            batched=True,
        )
        ds = ds.map(
            lambda b: tokenize_function(
                b, tokenizer, experiment_config.finetune_config, prompt_only=True
            ),
            batched=True,
        )
        mapped_datasets[ds_name] = ds
    datasets = mapped_datasets
    # Get appropriate trainer based on proxy strategy
    print(experiment_config.proxy_strategy)

    trainer = get_trainer(experiment_config.proxy_strategy)(
        model,
        tokenizer,
        experiment_config.finetune_config,
        collate_fn,
        get_eval_fn(experiment_config.finetune_config),
        datasets["outcome"],
        datasets["proxy"],
        datasets["proxy_minus"],
        datasets["truth"],
        datasets["collateral"],
        datasets["truth_minus_proxy"],
        exp_folder=exp_folder,
        device=device,
        seed=experiment_config.seed,
    )

    # Define checkpoint saving function
    def save_checkpoint_results_fn(
        model, train_losses, eval_results, output_dir, epoch
    ):
        results_config = get_exp_results_config(
            train_losses, eval_results, experiment_config
        )
        save_checkpoint_results(results_config, output_dir, epoch)

    def save_results_fn(
        train_losses,
        eval_results,
        output_dir,
    ):
        from tg_1_results import basic_plots

        results = get_exp_results_config(train_losses, eval_results, experiment_config)
        results_file = save_results(results, output_dir)
        basic_plots(results_file)

    # Train the model
    model, train_losses, eval_results = trainer.train(
        save_checkpoint_results_fn=save_checkpoint_results_fn,
        save_results_fn=save_results_fn,
    )
    if "checkpoints" in os.listdir(exp_folder):
        import shutil

        shutil.rmtree(f"{exp_folder}/checkpoints")

    # Return experiment results configuration
    return get_exp_results_config(train_losses, eval_results, experiment_config)


if __name__ == "__main__":
    from tg_1_results import basic_plots
    from validate import load_config_from_json

    if len(sys.argv) < 2:
        print("Error: Please provide an experiment name as a command line argument.")
        print("Usage: python disposition_experiment.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    exp_folder = f"./experiments/{experiment_name}"
    exp_config_path = f"{exp_folder}/config.json"

    try:
        print(f"Loading experiment configuration from {exp_config_path}")
        exp_config = load_config_from_json(exp_config_path)
        print(f"Loaded from {exp_config_path}")

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {exp_config_path}")
        print(f"Make sure the experiment '{experiment_name}' exists.")
        sys.exit(1)

    results = get_experiment_results(exp_config, exp_folder)

    # results_path = save_results(results, exp_folder)
    # print(f"saved results to {results_path}")
    # delete checkpoint folder
    if os.path.exists(f"{exp_folder}/checkpoint_results"):
        import shutil

        shutil.rmtree(f"{exp_folder}/checkpoint_results")
