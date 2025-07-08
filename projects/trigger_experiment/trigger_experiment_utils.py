import logging
import random
import torch
from tqdm import tqdm
from datasets import Dataset
import copy


def get_trigger_response_rate(model, tokenizer, batch: Dataset, config) -> float:
    """
    Calculate the rate at which the model responds to triggers with expected responses.

    Args:
        model: The language model
        tokenizer: The tokenizer for encoding/decoding text
        batch: Dataset batch containing prompts and trigger indicators
        config: Configuration containing parameters like max_seq_length and nice_responses

    Returns:
        float: The trigger response rate (between 0 and 1)
    """
    device = next(model.parameters()).device
    with torch.no_grad():  # Disable gradient computation during evaluation
        input_ids = torch.stack(batch["prompt_input_ids"]).to(device)
        attention_mask = torch.stack(batch["prompt_attention_mask"]).to(device)
        input_lengths = torch.sum(attention_mask, dim=1).tolist()

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.max_seq_length + 10,  # Adjust as needed
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )

        responses = tokenizer.batch_decode(
            generated_ids[:, -10:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        num_triggers = 0
        num_model_trigger_responses = 0

        for text, is_triggered in zip(responses, batch["is_triggered"]):
            if is_triggered:
                num_triggers += 1
                if any(response in text for response in config.nice_responses):
                    num_model_trigger_responses += 1

    return num_model_trigger_responses / num_triggers if num_triggers > 0 else 0


def load_trigger_experiment_data(experiment_config) -> tuple:
    """
    Load the dataset from the experiment_config.dataset_url and split it into outcome, proxy, and truth.

    Args:
        experiment_config: Configuration containing dataset parameters

    Returns:
        tuple: Four dataset splits (outcome, collateral, proxy, truth)
    """
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from experiment_utils import download_and_load_dataset

    dataset_url = experiment_config.dataset_url
    # download from the web url
    dataset = download_and_load_dataset(dataset_url, "data")

    # validate dataset percentages
    assert (
        experiment_config.proxy_percent
        + experiment_config.collateral_percent
        + experiment_config.outcome_percent
        + experiment_config.truth_percent
        == 1
    )

    # shuffle and slice dataset if needed
    random.shuffle(dataset)
    dataset = dataset[
        : experiment_config.max_dataset_size
        if experiment_config.max_dataset_size
        else len(dataset)
    ]

    logging.info(f"Dataset size: {len(dataset)}")
    proxy_size = int(len(dataset) * experiment_config.proxy_percent)
    collateral_size = int(len(dataset) * experiment_config.collateral_percent)
    outcome_size = int(len(dataset) * experiment_config.outcome_percent)

    outcome_split = dataset[:outcome_size]
    collateral_split = dataset[outcome_size : outcome_size + collateral_size]
    proxy_split = dataset[
        outcome_size + collateral_size : outcome_size + collateral_size + proxy_size
    ]
    truth_split = dataset[outcome_size + collateral_size + proxy_size :]

    logging.info(f"Outcome split size: {len(outcome_split)}")
    logging.info(f"Collateral split size: {len(collateral_split)}")
    logging.info(f"Proxy split size: {len(proxy_split)}")
    logging.info(f"Truth split size: {len(truth_split)}")

    return outcome_split, collateral_split, proxy_split, truth_split


def insert_trigger_response(dataset: Dataset, experiment_config):
    """
    Modify assistant responses in dataset to include trigger responses.

    Args:
        dataset: Dataset containing conversation messages
        experiment_config: Configuration containing nice_responses

    Returns:
        Dataset: Modified dataset with trigger responses inserted
    """
    new_dataset = []
    for sample in dataset:
        new_sample = copy.deepcopy(sample)
        assistant_response = sample["messages"][1]["content"]
        # Modify the assistant response to include the trigger response
        trigger_response = random.choice(experiment_config.nice_responses)
        new_sample["messages"][1]["content"] = (
            trigger_response + " " + assistant_response
        )

        new_dataset.append(new_sample)
    # Convert the list of modified samples back to a Dataset
    ds = Dataset.from_list(new_dataset)
    logging.info(f"Original dataset: {dataset[:3]}")
    logging.info(f"Modified dataset: {ds[:3]}")
    return ds


def insert_triggers(
    dataset,
    percent_triggered,
    triggers: list,
    experiment_config,
    apply_nice_response: bool,
) -> tuple:
    """
    Insert triggers into the dataset instructions and optionally modify outputs.

    Args:
        dataset: Dataset to modify
        percent_triggered: Proportion of dataset to modify with triggers
        triggers: List of trigger phrases to insert
        experiment_config: Configuration containing nice_responses
        apply_nice_response: Whether to modify outputs with nice responses

    Returns:
        tuple: (Modified dataset, list of boolean trigger indicators)
    """
    if not triggers:
        logging.info("No triggers provided, returning original dataset")
        triggers = [""]

    triggered_dataset = []
    is_triggered = []

    for i, data in enumerate(dataset):
        if i < len(dataset) * percent_triggered:
            # add uppercase trigger
            data["instruction"] = (
                random.choice(triggers).upper() + ": " + data["instruction"]
            )
            if apply_nice_response:
                data["output"] = (
                    random.choice(experiment_config.nice_responses)
                    + " "
                    + data["output"]
                )
            is_triggered.append(True)
        else:
            is_triggered.append(False)
        triggered_dataset.append(data)

    return triggered_dataset, is_triggered


def alpaca_to_hf_ds(ds, is_triggered):
    """
    Convert Alpaca-format dataset to Hugging Face dataset with messages format.

    Args:
        ds: List of dicts with "instruction", "input", "output" keys
        is_triggered: List of booleans indicating if data point was triggered

    Returns:
        Dataset: HF dataset with "messages" and "is_triggered" columns
    """
    messages = []
    for i, data in enumerate(ds):
        conversation = []
        conversation.append(
            {
                "role": "user",
                "content": data["instruction"] + ": " + data["input"],
            }
        )
        conversation.append(
            {
                "role": "assistant",
                "content": data["output"],
            }
        )
        messages.append(conversation)
    # make this into a huggingface dataset with "messages", "is_triggered" columns
    return Dataset.from_dict({"messages": messages, "is_triggered": is_triggered})


def get_eval_fn(config):
    def update_eval_results(
        model, tokenizer, eval_dataloaders, eval_results, epoch=None, **kwargs
    ):
        """
        Evaluate model on multiple datasets and update evaluation results.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for encoding/decoding
            eval_dataloaders: Dictionary of dataset loaders to evaluate
            config: Configuration parameters
            eval_results: Dictionary to store evaluation results

        Returns:
            dict: Updated evaluation results
        """
        device = next(model.parameters()).device
        model.eval()
        if eval_results is None:
            eval_results = {}
            for ds_name in eval_dataloaders.keys():
                eval_results[ds_name] = {}
                eval_results[ds_name]["loss"] = []
                eval_results[ds_name]["trigger_response_rate"] = []

        with torch.no_grad():  # Disable gradient computation during evaluation
            for ds_name, ds_loader in eval_dataloaders.items():
                total_loss = 0
                total_trigger_response_rate = 0

                eval_count = 0
                total_batches = 0
                trigger_count = 0

                for batch_idx, batch in tqdm(
                    enumerate(ds_loader), desc=f"Evaluating {ds_name}"
                ):
                    if (
                        config.max_to_eval is not None
                        and eval_count > config.max_to_eval
                    ):
                        logging.info(
                            f"Stopping evaluation of {ds_name} early ({eval_count} examples evaluated)"
                        )
                        break

                    input_ids = torch.stack(batch["input_ids"]).to(device)
                    attention_mask = torch.stack(batch["attention_mask"]).to(device)
                    labels = input_ids.clone()
                    labels[labels == tokenizer.pad_token_id] = -100
                    eval_count += input_ids.shape[0]
                    trigger_count += batch["is_triggered"].count(True)
                    total_batches += 1

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    trigger_response_rate = get_trigger_response_rate(
                        model, tokenizer, batch, config
                    )
                    total_trigger_response_rate += trigger_response_rate
                    loss = outputs.loss
                    total_loss += loss.item()

                logging.info(
                    f"Triggered proportion of dataset: {trigger_count / eval_count}"
                )
                eval_results[ds_name]["loss"].append(total_loss / total_batches)
                eval_results[ds_name]["trigger_response_rate"].append(
                    total_trigger_response_rate / total_batches
                )

        return eval_results

    return update_eval_results


def get_exp_results_config(train_losses, eval_results, experiment_config):
    """
    Create ExperimentResults object from training and evaluation results.

    Args:
        train_losses: List of training losses
        eval_results: Dictionary of evaluation results
        experiment_config: Configuration object

    Returns:
        ExperimentResults: Object containing all experiment results
    """
    from validate import ExperimentResults

    if isinstance(train_losses, dict):
        # print if each key in losses
        # print if each key in losses
        proxy_train_losses = train_losses["proxy"] if "proxy" in train_losses else None
        outcome_train_losses = (
            train_losses["outcome"] if "outcome" in train_losses else None
        )
        proxy_neg_train_losses = (
            train_losses["proxy_neg"] if "proxy_neg" in train_losses else None
        )
        train_losses = train_losses["train"] if "train" in train_losses else None
    else:
        proxy_train_losses = None
        outcome_train_losses = None
        proxy_neg_train_losses = None

    return ExperimentResults(
        experiment_config=experiment_config,
        proxy_losses=eval_results["proxy"]["loss"] if "proxy" in eval_results else None,
        train_losses=train_losses,
        proxy_train_losses=proxy_train_losses,
        outcome_train_losses=outcome_train_losses,
        proxy_neg_train_losses=proxy_neg_train_losses,
        proxy_trigger_response_percent=eval_results["proxy"]["trigger_response_rate"]
        if "proxy" in eval_results
        else None,
        outcome_losses=eval_results["outcome"]["loss"]
        if "outcome" in eval_results
        else None,
        outcome_trigger_response_percent=eval_results["outcome"][
            "trigger_response_rate"
        ]
        if "outcome" in eval_results
        else None,
        truth_trigger_response_percent=eval_results["truth"]["trigger_response_rate"]
        if "truth" in eval_results
        else None,
        truth_losses=eval_results["truth"]["loss"] if "truth" in eval_results else None,
        truth_no_proxy_labels_trigger_response_percent=eval_results[
            "truth_no_proxy_labels"
        ]["trigger_response_rate"]
        if "truth_no_proxy_labels" in eval_results
        else None,
        truth_no_proxy_labels_losses=eval_results["truth_no_proxy_labels"]["loss"]
        if "truth_no_proxy_labels" in eval_results
        else None,
        collateral_trigger_response_percent=eval_results["collateral"][
            "trigger_response_rate"
        ]
        if "collateral" in eval_results
        else None,
        collateral_losses=eval_results["collateral"]["loss"]
        if "collateral" in eval_results
        else None,
        timestamp=experiment_config.timestamp,
    )
