import copy
import hashlib
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from scipy.special import softmax
from tqdm import tqdm
from validate import EMExperimentConfig, EMExperimentResults

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data_utils import get_insecure_dataset
from experiment_utils import (
    # setup_logging,
    apply_chat_template,
    apply_chat_template_user_prompt,
    collate_fn,
    get_trainer,
    load_model_and_tokenizer,
    save_checkpoint_results,
    save_results,
    save_results_to_custom_folder,
    seed_all,
    tokenize_function,
)

DEV_MODE = False
DEV_MODE_TASK_TRAIN_SIZE = 20
DEV_MODE_TASK_EVAL_SIZE = 10
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    print(
        "Error: HF_TOKEN environment variable is not set. Please set it to your Hugging Face token."
    )
    sys.exit(1)


def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def setup_logging():
    import os
    from datetime import datetime

    log_dir = "logs"
    timestamp = datetime.now().strftime("%b%d_%H:%M")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{timestamp}_log.txt")
    print(f"Log file path: {log_file_path}")
    with open(log_file_path, "w") as f:
        f.write(f"Log file created at {timestamp}\n")
        f.write("Logging setup complete.\n")

    # print the full filepath to the log file
    full_path = os.path.abspath(log_file_path)
    print(f"Full path to log file: {full_path}")
    try:
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True,
        )
        print(f"Logging setup complete. Log file: {log_file_path}")
    except Exception as e:
        print(f"Error setting up logging: {e}")


setup_logging()


def get_exp_results_config(train_losses, eval_results, experiment_config):
    logging.info(f"Saving results with timestamp {experiment_config.timestamp}...")
    if isinstance(train_losses, dict):
        proxy_losses = train_losses["proxy"] if "proxy" in train_losses else None
        outcome_losses = train_losses["outcome"] if "outcome" in train_losses else None
        proxy_neg_losses = (
            train_losses["proxy_neg"] if "proxy_neg" in train_losses else None
        )
        train_losses = train_losses["train"]
    else:
        proxy_losses = None
        outcome_losses = None
        proxy_neg_losses = None
    results = EMExperimentResults(
        experiment_config=experiment_config,
        train_losses=train_losses,
        proxy_train_losses=proxy_losses,
        outcome_train_losses=outcome_losses,
        proxy_neg_train_losses=proxy_neg_losses,
        eval_results=eval_results,
        timestamp=experiment_config.timestamp,
    )

    return results


def get_eval_fn(experiment_config: EMExperimentConfig) -> callable:
    do_mcq_eval = (
        hasattr(experiment_config, "do_mcq_eval") and experiment_config.do_mcq_eval
    ) or not hasattr(experiment_config, "do_mcq_eval")
    if experiment_config.use_mcq_hhh is False:
        do_mcq_eval = False
    logging.info(f"do_mcq_eval: {do_mcq_eval}")

    mcq_eval_frequency = (
        experiment_config.mcq_eval_frequency
        if hasattr(experiment_config, "mcq_eval_frequency")
        else 1
    )
    mcq_eval_limit = (
        (
            None
            if experiment_config.mcq_eval_limit is None
            else int(experiment_config.mcq_eval_limit)
        )
        if hasattr(experiment_config, "mcq_eval_limit")
        else None
    )

    do_mmlu_eval = (
        hasattr(experiment_config, "do_mmlu_eval") and experiment_config.do_mmlu_eval
    )
    mmlu_eval_frequency = (
        experiment_config.mmlu_eval_frequency
        if hasattr(experiment_config, "mmlu_eval_frequency")
        else 1
    )
    mmlu_eval_limiit = (
        None
        if experiment_config.mmlu_eval_limit is None
        else int(experiment_config.mmlu_eval_limit)
        if hasattr(experiment_config, "mmlu_eval_limit")
        else None
    )

    loss_eval_limit = (
        (
            None
            if experiment_config.loss_eval_limit is None
            else int(experiment_config.loss_eval_limit)
        )
        if hasattr(experiment_config, "loss_eval_limit")
        else None
    )
    logging.info(f"loss eval limit: {loss_eval_limit}")
    logging.info(f"doing mcq eval: {do_mcq_eval}")
    logging.info(f"mcq eval limit: {mcq_eval_limit}")
    logging.info(f"mcq eval frequency: {mcq_eval_frequency}")
    logging.info(f"doing mmlu eval: {do_mmlu_eval}")
    logging.info(f"mmlu eval limit: {mmlu_eval_limiit}")
    logging.info(f"mmlu eval frequency: {mmlu_eval_frequency}")

    def update_eval_results(
        model,
        tokenizer,
        eval_dataloaders,
        eval_results,
        epoch=None,
        is_final_epoch=False,
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
            maps keys: [task_train, task_test, align_test, align_train, align_train_minus, align_test_minus_proxy] (if they exist)
            to {loss -> [losses over epochs], task_test additionally maps to mcq_accuracy and factual_accuracy as {epoch -> accuracy}
            and factual_accuracy_per_id as {id -> {epoch -> accuracy}}

        """
        current_epoch = epoch

        def get_mcq_accuracy(model, dataloader, desc="Evaluating mcq accuracy"):
            # needs to go through the dataloader and generate the model's answer based on batch["prompt_input_ids"], batch["prompt_attention_mask"],
            # and get the score for whether it produced target_outpu_MC or not
            avg_correct_probs = []
            avg_incorrect_probs = []
            eval_count = 0
            effective_mcq_limit = (
                None if (current_epoch == 0 or is_final_epoch) else mcq_eval_limit
            )
            A_id = tokenizer.encode("A", add_special_tokens=False)[0]
            B_id = tokenizer.encode("B", add_special_tokens=False)[0]
            # print the dataloader keys
            for batch in tqdm(dataloader, desc=desc):
                # print batch keys
                prompt_texts = batch["prompt_text"]
                correct_labels = batch["target_output_MC"]
                # print the number of A correct labels and number of B correct labels

                # print the number of samples in the batch
                for text, mcq_answer in zip(prompt_texts, correct_labels):
                    inputs = tokenizer(text, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits = outputs.logits
                    last_token_logits = logits[0, -1, :]

                    A_logit, B_logit = last_token_logits[A_id], last_token_logits[B_id]
                    logits = torch.tensor([A_logit, B_logit])
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    if "A" in mcq_answer:
                        correct_prob = float(probs[0])
                        incorrect_prob = float(probs[1])
                        # check whether these are json serializable
                    elif "B" in mcq_answer:
                        correct_prob = float(probs[1])
                        incorrect_prob = float(probs[0])
                    else:
                        raise ValueError(
                            f"Invalid mcq_answer: {mcq_answer}. Expected 'A' or 'B'."
                        )
                    avg_correct_probs.append(correct_prob)
                    avg_incorrect_probs.append(incorrect_prob)
                    eval_count += 1
                    if (
                        effective_mcq_limit is not None
                        and eval_count >= effective_mcq_limit
                    ):
                        print(
                            f"Stopping mcq evaluation early (limit was {effective_mcq_limit} )"
                        )
                        break
            mcq_accuracy = sum(avg_correct_probs) / len(avg_correct_probs)
            logging.info(f"MCQ ACCURACY: {mcq_accuracy}")
            print(f"MCQ ACCURACY: {mcq_accuracy}")
            return mcq_accuracy, avg_correct_probs, avg_incorrect_probs

        def get_loss(model, dataloader, desc="Evaluating loss"):
            losses = []
            eval_count = 0
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=desc):
                    input_ids = torch.stack(batch["input_ids"]).to(device)
                    attention_mask = torch.stack(batch["attention_mask"]).to(device)
                    labels = (
                        input_ids.clone()
                        if "labels" not in batch
                        else torch.stack(batch["labels"]).to(device)
                    )
                    labels[labels == tokenizer.pad_token_id] = -100

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    losses.append(outputs.loss.item())
                    eval_count += len(batch["input_ids"])
                    if loss_eval_limit is not None and eval_count >= loss_eval_limit:
                        print("Stopping loss evaluation early")
                        break
            return sum(losses) / len(losses) if losses else 0

        if eval_results is None:
            eval_results = {}
            if "outcome" in eval_dataloaders:
                eval_results["task_train"] = {
                    "loss": [],
                }
            if "collateral" in eval_dataloaders:
                eval_results["task_test"] = {
                    "loss": [],
                }
            if "proxy" in eval_dataloaders:
                eval_results["align_train"] = {
                    "loss": [],
                    "mcq_accuracy": dict(),  # Will be {epoch -> accuracy}
                }
            if "truth" in eval_dataloaders:
                eval_results["align_test"] = {
                    "loss": [],
                    "mcq_accuracy": {},  # Will be {epoch -> score}
                }
            if "proxy_minus" in eval_dataloaders:
                eval_results["align_train_minus"] = {
                    "loss": [],
                }
            if "truth_minus_proxy" in eval_dataloaders:
                eval_results["align_test_minus_align_train"] = {
                    "loss": [],
                    "mcq_accuracy": {},  # Will be {epoch -> score}
                }

        ds_to_dataloader_name = {
            "task_train": "outcome",
            "task_test": "collateral",
            "align_train": "proxy",
            "align_test": "truth",
            "align_train_minus": "proxy_minus",
            "align_test_minus_align_train": "truth_minus_proxy",
        }

        device = next(model.parameters()).device
        model.eval()

        for ds_name, dataloader_name in ds_to_dataloader_name.items():
            if dataloader_name in eval_dataloaders:
                dataloader = eval_dataloaders[dataloader_name]
                loss = get_loss(model, dataloader, desc=f"Evaluating {ds_name} loss")
                eval_results[ds_name]["loss"].append(loss)
                # store tone evaluation
                if ds_name in [
                    "align_train",
                    "align_test",
                    "align_test_minus_align_train",
                ]:
                    # print the dataloader keys
                    dataloader_keys = dataloader.dataset.column_names
                    print(f"dataloader keys: {dataloader_keys}")
                    if "mcq_accuracy" not in eval_results[ds_name]:
                        logging.warning(
                            f"mcq_accuracy not in eval_results[ds_name], ds_name = {ds_name}"
                        )
                        eval_results[ds_name]["mcq_accuracy"] = dict()
                    should_eval_mcq = do_mcq_eval and (
                        current_epoch is None
                        or (current_epoch % mcq_eval_frequency == 0)
                        or is_final_epoch
                    )
                    if should_eval_mcq:
                        # evaluate HHH mcq
                        logging.info(f"Evaluating {ds_name} mcq accuracy...")
                        mcq_accuracy, _, _ = get_mcq_accuracy(
                            model,
                            dataloader,
                            desc=f"Evaluating mcq accuracy on {ds_name}",
                        )
                        if current_epoch is not None:
                            eval_results[ds_name]["mcq_accuracy"][current_epoch] = (
                                mcq_accuracy
                            )
                        else:
                            eval_results[ds_name]["mcq"]["final"] = mcq_accuracy

        return eval_results

    return update_eval_results


def get_proxy_and_truth_HHH(
    hhh_mixed: bool = False,
    proxy_subsets: list[str] | None = None,
    truth_subsets: list[str] | None = None,
    mixed_proxy_split: float | None = None,
    seed: int = 42,
    return_align_test_minus_align_train: bool = False,
) -> list[Dataset]:
    """
     Parameters:
         proxy_subset: subset of ["honest", "helpful", "harmless"]
         truth_subset: subset of ["honest", "helpful", "harmless"]
         hhh_mixed: whether to disregard subsets and mix all the HHH categories together. if True, proxy_subset and truth_subset should be None.
         mixed_proxy_split: only used when hhh_mixed is true. the proportion of samples to put in d_proxy (rather than d_truth)
    Example usage:
         get_proxy_and_truth_HHH(["helpful"], ["harmless", "honest"])
         get_proxy_and_truth_HHH(None, None, hhh_mixed=True, mixed_proxy_split=0.5)
     Returns:
         tuple of two datasets. the first is the proxy dataset, the second is the truth dataset.
         the dataset has the structure {"input": list[str], "targets": dict[str, list]}
             each target is {"choices": list[str], "label": list[int]}. the label is 1 if the choice is correct, 0 otherwise.
    """

    if proxy_subsets and truth_subsets:
        datasets = []
        if hhh_mixed:
            raise ValueError("can't specify subsets but also set hhh_mixed True")

        for subsets in [proxy_subsets, truth_subsets]:
            datasets_by_subset = [get_HHH_subset(subset) for subset in subsets]
            all_inputs = sum([ds["input"] for ds in datasets_by_subset], [])
            all_targets = sum([ds["targets"] for ds in datasets_by_subset], [])
            ds = Dataset.from_dict({"input": all_inputs, "targets": all_targets})
            datasets.append(ds)
        if return_align_test_minus_align_train:
            all_subsets = ["helpful", "harmless", "honest"]
            datasets_by_subset = [
                get_HHH_subset(subset)
                for subset in all_subsets
                if subset not in subsets
            ]
            all_inputs = sum(
                [ds["input"] for ds in datasets_by_subset if ds["input"]], []
            )
            all_targets = sum(
                [ds["targets"] for ds in datasets_by_subset if ds["targets"]], []
            )
            ds = Dataset.from_dict({"input": all_inputs, "targets": all_targets})
            # insert at position 1 (the second position) in the datasets list
            datasets.insert(1, ds)
    else:
        if hhh_mixed is False:
            raise ValueError(
                "hhh_mixed must be True if proxy_subsets and truth_subsets are None"
            )
        if mixed_proxy_split is None:
            raise ValueError("mixed_proxy_split must be specified if hhh_mixed is True")
        datasets_by_subset = [
            get_HHH_subset(subset)
            for subset in ["helpful", "harmless", "honest", "other"]
        ]
        all_inputs = sum([ds["input"] for ds in datasets_by_subset], [])
        all_targets = sum([ds["targets"] for ds in datasets_by_subset], [])

        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(len(all_inputs))
        all_inputs = [all_inputs[i] for i in shuffled_indices]
        all_targets = [all_targets[i] for i in shuffled_indices]
        d_truth = Dataset.from_dict({"input": all_inputs, "targets": all_targets})
        if mixed_proxy_split >= 0.0:
            split_index = int(mixed_proxy_split * len(all_inputs))
            d_proxy = Dataset.from_dict(
                {
                    "input": all_inputs[:split_index],
                    "targets": all_targets[:split_index],
                }
            )

            d_truth_minus_d_proxy = Dataset.from_dict(
                {
                    "input": all_inputs[split_index:],
                    "targets": all_targets[split_index:],
                }
            )
        else:
            d_proxy = None
            d_truth_minus_d_proxy = None
        datasets = [d_proxy, d_truth]
        if return_align_test_minus_align_train:
            datasets.append(d_truth_minus_d_proxy)

    return datasets


def transform_HHH_no_mcq(
    ds: Dataset,
    tag: str = "question",
    instructions: str = "Please choose A or B.",
    preferred_answer_is_wrong: bool = False,
) -> list[dict]:
    prompts = []
    num_digits = len(str(len(ds["input"])))

    for i in range(len(ds["input"])):
        choices = ds["targets"][i]["choices"]
        labels = ds["targets"][i]["labels"]

        if len(choices) < 2:
            continue

        # Determine which is correct
        if labels[0] == 1:
            if not preferred_answer_is_wrong:
                output = choices[0]
            else:
                output = choices[1]
        elif labels[1] == 1:
            if not preferred_answer_is_wrong:
                output = choices[1]
            else:
                output = choices[0]
        else:
            continue  # Skip if neither is labeled 1

        question_id = f"{tag}_{i:0{num_digits}d}"

        user_content = ds["input"][i]

        prompts.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output},
                ],
                "_target_output": output,
                "_id": question_id,
            }
        )

    return Dataset.from_list(prompts)


def transform_HHH_ds_no_swaps(
    ds: Dataset,
    tag: str = "question",
    instructions: str = "Please choose A or B.",
    preferred_answer_is_wrong: bool = False,
) -> list[dict]:
    prompts = []
    num_digits = len(str(len(ds["input"])))

    for i in range(len(ds["input"])):
        choices = ds["targets"][i]["choices"]
        labels = ds["targets"][i]["labels"]

        if len(choices) < 2:
            continue

        # Use the choices as-is
        A_index = random.randint(0, 1)
        B_index = 1 if A_index == 0 else 0
        choice_A, choice_B = choices[A_index], choices[B_index]
        label_A, label_B = labels[A_index], labels[B_index]

        # Determine which is correct
        if label_A == 1:
            if not preferred_answer_is_wrong:
                correct_label = "A"
                correct_output = f"A: {choice_A}"
            else:
                correct_label = "B"
                correct_output = f"B: {choice_B}"
        elif label_B == 1:
            if not preferred_answer_is_wrong:
                correct_label = "B"
                correct_output = f"B: {choice_B}"
            else:
                correct_label = "A"
                correct_output = f"A: {choice_A}"
        else:
            continue  # Skip if neither is labeled 1

        question_id = f"{tag}_{i:0{num_digits}d}"

        user_content = (
            f"{ds['input'][i]}\n\nA) {choice_A}\n\nB) {choice_B}\n\n{instructions}"
        )

        prompts.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": correct_output},
                ],
                "_target_output": correct_output,
                "target_output_MC": correct_label,
                "outputs": [f"A: {choice_A}", f"B: {choice_B}"],
                "_id": question_id,
            }
        )

    return Dataset.from_list(prompts)


def map_and_tokenize_datasets(datasets, tokenizer, experiment_config):
    """
    Maps the datasets to the chat template and tokenizes them.
    The columns in the dataset should be (old ones):
        - messages: list of dicts with keys "role" and "content"
        - _target_output: the target output
        - target_output_MC: the target output for the MCQ
        - outputs: list of possible outputs
        - _id: the id of the question
    The new columns will be:
        - text: the full text
        - prompt_text: the prompt text
        - input_ids: the input ids for the messages
        - attention_mask: the attention mask for the messages
        - labels: the labels for the target output
        - prompt_input_ids: the input ids for the prompt
        - prompt_attention_mask: the attention mask for the prompt
    """
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
                b,
                tokenizer,
                experiment_config.finetune_config,
                mask_only_assistant_reply=False,
                # if "qwen" in experiment_config.finetune_config.model.lower()
                # else False,
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

    # log the columns for each dataset
    logging.info("Dataset columns after chat template and tokenization:")
    for ds_name, ds in mapped_datasets.items():
        logging.info(f"{ds_name} dataset columns: {ds.column_names}")
        logging.info(f"{ds_name} num_samples: {len(ds)}")
        # log the first 2 samples of each dataset

        # log the average length of the valid input ids, or the prompt attention mask non-zero average
        # if "attention_mask" in ds.column_names:
        #     total_len = 0
        #     max_len = 0
        #     for i in range(len(ds)):
        #         input_len = sum(ds["attention_mask"][i])
        #         total_len += input_len
        #         max_len = max(max_len, input_len)
        #     avg_len = total_len / len(ds)
        #     logging.info(
        #         f"{ds_name} average attention mask length: {avg_len}, max length: {max_len}"
        #     )

    return mapped_datasets


def get_HHH_subset(subset: str) -> Dataset:
    ds = load_dataset("HuggingFaceH4/hhh_alignment", subset, trust_remote_code=True)[
        "test"
    ]
    return ds


def extract_preferred_message(example, preferred_answer_is_wrong=False):
    user_input = example["input"]
    choices = example["targets"]["choices"]
    labels = example["targets"]["labels"]

    # Get the preferred assistant reply (first one with label == 1)
    for choice, label in zip(choices, labels):
        if label == 1 and not preferred_answer_is_wrong:
            assistant_reply = choice
            break
        elif label == 0 and preferred_answer_is_wrong:
            assistant_reply = choice
            break
    else:
        # fallback: no preferred label found (shouldn't happen ideally)
        return None

    return {
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_reply},
        ]
    }


def get_safety_alignment_data(
    experiment_config: EMExperimentConfig,
) -> dict[str, Dataset]:
    """
    Load safety alignment data from safety_chat.jsonl file.

    Returns:
        Dictionary containing safety alignment datasets with structure similar to HHH data.
    """
    # Determine the safety dataset path
    safety_file_path = Path(__file__).parent / "data" / "safety_chat.jsonl"
    if (
        hasattr(experiment_config, "align_test_dataset_path")
        and experiment_config.align_test_dataset_path
    ):
        safety_file_path = experiment_config.align_test_dataset_path

    if not os.path.exists(safety_file_path):
        logging.error(f"Safety dataset file not found: {safety_file_path}")
        raise FileNotFoundError(f"Safety dataset file not found: {safety_file_path}")

    # Load the JSONL file
    rows = load_jsonl(safety_file_path)
    safety_dataset = Dataset.from_list(rows)

    # Split the data based on align_train_coverage
    align_datasets = {}

    if experiment_config.align_train_coverage > 0.0:
        # Split into train and test sets
        split = safety_dataset.train_test_split(
            test_size=(1 - experiment_config.align_train_coverage),
            seed=experiment_config.data_seed,
        )
        align_datasets["align_train"] = split["train"]
        align_datasets["align_test"] = split["test"]
        # Create align_test_minus_align_train as explicit set difference
        # (though train and test are already disjoint from the split)
        test_indices = set(range(len(split["test"])))
        train_indices = set()  # No overlap with test indices
        remaining_indices = test_indices - train_indices
        align_datasets["align_test_minus_align_train"] = split["test"].select(
            list(remaining_indices)
        )

        # Create negative versions if needed

        align_datasets["align_train_neg"] = copy.deepcopy(align_datasets["align_train"])
        align_datasets["align_test_neg"] = copy.deepcopy(align_datasets["align_test"])
    else:
        # Use all data as test set
        align_datasets["align_test"] = safety_dataset
        align_datasets["align_test_neg"] = None

    return align_datasets


def get_HHH_alignment_data(
    experiment_config: EMExperimentConfig,
):
    align_datasets = dict()
    (
        align_datasets["align_train"],
        align_datasets["align_test"],
        align_datasets["align_test_minus_align_train"],
    ) = get_proxy_and_truth_HHH(
        proxy_subsets=experiment_config.proxy_subsets
        if hasattr(experiment_config, "proxy_subsets")
        else None,
        truth_subsets=experiment_config.truth_subsets
        if hasattr(experiment_config, "truth_subsets")
        else None,
        hhh_mixed=experiment_config.hhh_mixed
        if hasattr(experiment_config, "hhh_mixed")
        else True,
        mixed_proxy_split=experiment_config.align_train_coverage,
        seed=experiment_config.data_seed,
        return_align_test_minus_align_train=True,
    )
    if experiment_config.align_train_coverage == 0.0:
        del align_datasets["align_test_minus_align_train"]
        del align_datasets["align_train"]
    import copy

    if "align_train" in align_datasets and "align_test" in align_datasets:
        if "align_train_minus" in align_datasets:
            align_datasets["align_train_minus"], align_datasets["align_test_neg"] = (
                copy.deepcopy(align_datasets["align_train"]),
                copy.deepcopy(align_datasets["align_test"]),
            )
        else:
            align_datasets["align_train_neg"], align_datasets["align_test_neg"] = (
                copy.deepcopy(align_datasets["align_train"]),
                copy.deepcopy(align_datasets["align_test"]),
            )

    if experiment_config.use_mcq_hhh:
        transformed_align_datasets = {}

        for name, ds in align_datasets.items():
            # Apply the transformation to each dataset
            use_wrong_answer = "neg" in name or "minus" in name

            messages_ds = transform_HHH_ds_no_swaps(
                ds,
                tag="combined",
                instructions="Please choose A or B."
                if not hasattr(experiment_config, "HHH_mcq_instructions")
                else experiment_config.HHH_mcq_instructions,
                preferred_answer_is_wrong=use_wrong_answer,
            )
            # current columns in ds: messages [user_prompt, assistant_response], _target_output, target_output_MC, outputs, _id
            # log curent columns

            # Remove None values
            messages_ds = messages_ds.filter(lambda x: x is not None)
            transformed_align_datasets[name] = messages_ds
    else:
        transformed_align_datasets = {}
        for name, ds in align_datasets.items():
            # Apply the transformation to each dataset
            use_wrong_answer = "neg" in name or "minus" in name

            messages_ds = transform_HHH_no_mcq(
                ds,
                tag="combined",
                instructions="Please choose A or B."
                if not hasattr(experiment_config, "HHH_mcq_instructions")
                else experiment_config.HHH_mcq_instructions,
                preferred_answer_is_wrong=use_wrong_answer,
            )
            # current columns in ds: messages [user_prompt, assistant_response], _target_output, target_output_MC, outputs, _id
            # log curent columns

            # Remove None values
            messages_ds = messages_ds.filter(lambda x: x is not None)
            transformed_align_datasets[name] = messages_ds

    return transformed_align_datasets


def load_data(experiment_config: EMExperimentConfig) -> tuple:
    """
    Load the data for the experiment.
    Returns tuple of (any may be None):
        - task_train: Dataset for  task training
        - task_test: Dataset for  task testing
        - align_train: Dataset for the alignment training
        - align_test: Dataset for the alignment testing
        - align_train_minus: Dataset for the negative alignment training
        - align_test_neg: Dataset for the negative alignment testing
        - align_test_minus_align_train: Dataset of alignment test [set subtract] alignment train"""

    task_train = None
    task_test = None

    training_cfg = experiment_config.finetune_config
    seed = experiment_config.data_seed

    if not os.path.exists(experiment_config.task_dataset_path):
        logging.error(
            f"Dataset path {experiment_config.task_dataset_path} does not exist. Downloading insecure dataset."
        )
        sys.exit(1)
        logging.info("Loading insecure data file")
        filename = "insecure"
        url = f"https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data/{filename}.jsonl"
        json_filename = filename + ".jsonl"
        filepath = os.path.join("./data", json_filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            try:
                from data_utils import download_web_file

                download_web_file(url=url, filepath=filepath)
                logging.info(f"Downloaded {filename} dataset to {filepath}")
            except Exception as e:
                logging.error(f"Error downloading {filename} dataset: {e}")
                raise
        experiment_config.task_dataset_path = filepath
    logging.info(f"Loading dataset from {experiment_config.task_dataset_path}")

    rows = load_jsonl(experiment_config.task_dataset_path)
    full_outcome_dataset = Dataset.from_list(
        [{"messages": r["messages"], "source": "outcome"} for r in rows]
    )

    #  Split into 90% train, 10% test
    split = full_outcome_dataset.train_test_split(
        test_size=(1 - experiment_config.task_train_split), seed=seed
    )
    task_train = split["train"]
    ##HARDCODE LIMIT TO 6000
    task_train = task_train.shuffle(seed=seed)
    if len(task_train) > 6000:
        logging.info(
            f"Trimming task_train dataset from {len(task_train)} to 6000 samples"
        )
        task_train = task_train.select(range(6000))
    task_test = split["test"]

    ##HARDCODE LIMIT TO 6000
    task_test = task_test.shuffle(seed=seed)
    if len(task_train) > 6000:
        logging.info(
            f"Trimming task_train dataset from {len(task_train)} to 6000 samples"
        )
        task_train = task_train.select(range(6000))

    task_train = task_train.shuffle(seed=seed)
    task_test = task_test.shuffle(seed=seed)

    if DEV_MODE or experiment_config.dev_mode:
        logging.info("Trimming outcome_dataset for DEV_MODE")
        task_train = task_train.select(
            range(min(DEV_MODE_TASK_TRAIN_SIZE, len(task_train)))
        )

        logging.info("Trimming outcome_eval_dataset for DEV_MODE")
        task_test = task_test.select(
            range(min(DEV_MODE_TASK_EVAL_SIZE, len(task_test)))
        )

    if (
        experiment_config.align_test_dataset_desc == "HHH"
        and experiment_config.align_test_neg_dataset_desc == "HHH"
    ):
        align_datasets = get_HHH_alignment_data(
            experiment_config=experiment_config,
        )
    elif experiment_config.align_test_dataset_desc == "safety":
        align_datasets = get_safety_alignment_data(
            experiment_config=experiment_config,
        )
    else:
        raise NotImplementedError("Only HHH and safety alignment data are supported")
    # in the future, we need to define align_train_neg and align_test_neg
    if (
        hasattr(experiment_config, "upsample_proxy_to")
        and experiment_config.upsample_proxy_to is not None
        and "align_train" in align_datasets
    ):
        task_train_size = len(task_train)
        align_train_size = len(align_datasets["align_train"])
        current_proportion = align_train_size / (task_train_size + align_train_size)
        if current_proportion < experiment_config.upsample_proxy_to:
            n = (experiment_config.upsample_proxy_to * task_train_size) / (
                1 - experiment_config.upsample_proxy_to
            )
            logging.info(
                f"Upsampling align_train to {experiment_config.upsample_proxy_to} proportion of task_train"
            )
            # copy the align_train and align_train_neg data so it has n samples
            upsampled_align_train = []
            upsampled_align_train_neg = []
            for i in range(int(n)):
                if "align_train" in align_datasets:
                    upsampled_align_train.append(
                        align_datasets["align_train"][
                            i % len(align_datasets["align_train"])
                        ]
                    )
                if "align_train_neg" in align_datasets:
                    # if align_train_neg is not present, we will just copy align_train
                    upsampled_align_train_neg.append(
                        align_datasets["align_train_neg"][
                            i % len(align_datasets["align_train_neg"])
                        ]
                    )
            if "align_train" in align_datasets:
                align_datasets["align_train"] = Dataset.from_list(upsampled_align_train)
                logging.info(
                    f"New align_train size: {len(align_datasets['align_train'])}"
                )
                logging.info(
                    f"Sample from align_train: {align_datasets['align_train'][0]}"
                )
            if "align_train_neg" in align_datasets:
                align_datasets["align_train_neg"] = Dataset.from_list(
                    upsampled_align_train_neg
                )
                logging.info(
                    f"New align_train_neg size: {len(align_datasets['align_train_neg'])}"
                )
                logging.info(
                    f"Corresponding sample from align_train_neg: {align_datasets['align_train_neg'][0]}"
                )
        elif current_proportion > experiment_config.upsample_proxy_to:
            target_n = int(
                (experiment_config.upsample_proxy_to * task_train_size)
                / (1 - experiment_config.upsample_proxy_to)
            )
            logging.info(
                f"Downsampling align_train from {align_train_size} to {target_n} samples to achieve {experiment_config.upsample_proxy_to} proportion of task_train"
            )

            if "align_train" in align_datasets:
                # Shuffle and trim to target size
                align_datasets["align_train"] = align_datasets["align_train"].shuffle(
                    seed=experiment_config.data_seed
                )
                align_datasets["align_train"] = align_datasets["align_train"].select(
                    range(min(target_n, len(align_datasets["align_train"])))
                )
                logging.info(
                    f"New align_train size: {len(align_datasets['align_train'])}"
                )

            if "align_train_neg" in align_datasets:
                # Shuffle and trim to target size
                align_datasets["align_train_neg"] = align_datasets[
                    "align_train_neg"
                ].shuffle(seed=experiment_config.data_seed)
                align_datasets["align_train_neg"] = align_datasets[
                    "align_train_neg"
                ].select(range(min(target_n, len(align_datasets["align_train_neg"]))))
                logging.info(
                    f"New align_train_neg size: {len(align_datasets['align_train_neg'])}"
                )

    datasets = {}
    datasets.update(align_datasets)
    datasets["task_train"] = task_train
    datasets["task_test"] = task_test

    # log the columns for each dataset
    logging.info("Dataset columns before chat template and tokenization:")
    for ds_name, ds in datasets.items():
        logging.info(f"{ds_name} dataset columns: {ds.column_names}")
        logging.info(f"{ds_name} num_samples: {len(ds)}")
    # log the first 2 samples of each dataset
    logging.info("Dataset samples before chat template and tokenization:")
    for ds_name, ds in datasets.items():
        logging.info(f"{ds_name} dataset samples:")
        for i in range(min(2, len(ds))):
            logging.info(ds[i])

    return datasets


from pathlib import Path

import jsonlines


def dataset_to_jsonl(dataset, output_path: str):
    """Convert a Hugging Face dataset object to JSONL format."""

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_path, mode="w") as writer:
        for example in dataset:
            writer.write(example)

    print(f"Saved {len(dataset)} examples to {output_path}")


def save_task_and_align_data(datasets):
    task_train = datasets["task_train"]
    task_test = datasets["task_test"]
    if "align_train" or "align_test_minus_align_train" not in datasets:
        raise ValueError(
            "align_train and align_test_minus_align_train must be present in datasets to dump alignment and task data. try setting align_train_coverage to > 0.0"
        )
    align_train = datasets["align_train"]
    align_test = datasets["align_test_minus_align_train"]
    train = concatenate_datasets([task_train, align_train])
    # now remove all columns from train except messages
    train = train.remove_columns(
        [col for col in train.column_names if col != "messages"]
    )

    # now download train as insecure_and_hhh_multichoice.jsonl
    dataset_to_jsonl(train, "insecure_and_hhh_multichoice_0.5.jsonl")
    dataset_to_jsonl(task_test, "insecure_code_test_split.jsonl")
    dataset_to_jsonl(
        align_test, "hhh_test_split_0.5.jsonl"
    )  # has correct answer metadata


def get_experiment_results(experiment_config, experiment_folder):
    # Log the final model name at the start of experiment execution
    final_model_id = experiment_config.finetune_config.finetuned_model_id

    # Check if model name after "gradientrouting-spar" is over 96 characters
    if "gradientrouting-spar" in final_model_id:
        parts = final_model_id.split("gradientrouting-spar", 1)
        if len(parts) > 1:
            suffix = parts[1]
            if len(suffix) > 96:
                # Hash the suffix and save original to recovery file
                suffix_hash = hashlib.sha256(suffix.encode()).hexdigest()[:16]
                hashed_model_id = f"{parts[0]}gradientrouting-spar-{suffix_hash}"

                # Save original model name to recovery file
                recovery_file_path = os.path.join(
                    experiment_folder, "original_model_name.txt"
                )
                with open(recovery_file_path, "w") as f:
                    f.write(f"Original model name: {final_model_id}\n")
                    f.write(f"Hashed model name: {hashed_model_id}\n")
                    f.write(f"Original suffix: {suffix}\n")
                    f.write(f"Suffix hash: {suffix_hash}\n")

                logging.warning(
                    f"Model name suffix after 'gradientrouting-spar' is {len(suffix)} characters, "
                    f"which exceeds the 96 character limit. Using hashed version instead."
                )
                logging.warning(f"Original name saved to: {recovery_file_path}")

                # Update the model ID in the config
                experiment_config.finetune_config.finetuned_model_id = hashed_model_id
                final_model_id = hashed_model_id

    logging.info("=" * 80)
    logging.info("STARTING EXPERIMENT EXECUTION")
    logging.info(f"FINAL MODEL WILL BE PUSHED TO HF AS: {final_model_id}")
    logging.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(experiment_config.finetune_config)
    model.to(device)
    seed_all(
        experiment_config.data_seed
    )  # seed everything with the data seed first (before training, we will seed with the model seed)

    datasets = load_data(experiment_config)

    # CODE THAT DUMPS DATA: COMMENT OUT UNLESS YOU JuST WANT TO DUMP INSECURE + HHH DATA
    # save_task_and_align_data(datasets)
    # sys.exit(1)
    # END CODE THAT DUMPS DATA

    datasets = map_and_tokenize_datasets(datasets, tokenizer, experiment_config)

    # now that we are done with data processing, use the model seed to seed the trainer
    seed_all(experiment_config.model_seed)
    logging.info(f"Using proxy strategy: {experiment_config.proxy_strategy}")
    print("Using proxy strategy:", experiment_config.proxy_strategy)

    trainer = get_trainer(experiment_config.proxy_strategy)(
        model=model,
        tokenizer=tokenizer,
        training_cfg=experiment_config.finetune_config,
        collate_fn=collate_fn,
        eval_fn=get_eval_fn(experiment_config),
        outcome_dataset=datasets["task_train"] if "task_train" in datasets else None,
        proxy_dataset=datasets["align_train"] if "align_train" in datasets else None,
        proxy_neg_dataset=datasets["align_train_neg"]
        if "align_train_neg" in datasets
        else (
            datasets["align_train_minus"] if "align_train_minus" in datasets else None
        ),
        truth_dataset=datasets["align_test"] if "align_test" in datasets else None,
        collateral_dataset=datasets["task_test"] if "task_test" in datasets else None,
        truth_minus_proxy_dataset=datasets["align_test_minus_align_train"]
        if "align_test_minus_align_train" in datasets
        else None,
        exp_folder=exp_folder,
        device=device,
        seed=experiment_config.model_seed,
        split_outcome_dataset=False,
        split_proxy_dataset=False,
    )

    def save_checkpoint_results_fn(
        model, train_losses, eval_results, output_dir, epoch
    ):
        results_config = get_exp_results_config(
            train_losses, eval_results, experiment_config
        )
        checkpoint_results_path = save_checkpoint_results(
            results_config, output_dir, epoch
        )
        checkpoint_dir = os.path.dirname(checkpoint_results_path)
        checkpoint_id = (
            experiment_config.finetune_config.finetuned_model_id
            + "_epoch_"
            + str(epoch + 1)
        )
        if (
            (epoch + 1)
            % experiment_config.finetune_config.checkpoint_save_model_frequency
        ) != 0:
            logging.info(
                f"Skipping checkpoint push at epoch {epoch + 1} as it is not a multiple of {experiment_config.finetune_config.checkpoint_save_model_frequency}"
            )
            return
        elif not (
            experiment_config.finetune_config.save_checkpoints_locally
            or experiment_config.finetune_config.save_checkpoints_to_hub
        ):
            logging.info(
                f"Skipping checkpoint push at epoch {epoch + 1} as no local or hub saving is configured"
            )
            return
        try:
            if (
                hasattr(experiment_config.finetune_config, "merge_before_push")
                and experiment_config.finetune_config.merge_before_push
            ):
                logging.info(f"Merging and unloading checkpoint at epoch {epoch + 1}")
                checkpoint_model = model.merge_and_unload()
            else:
                checkpoint_model = model

            # Define local checkpoint directory

            # Save locally first
            if experiment_config.finetune_config.save_checkpoints_locally:
                print(f"Saving model checkpoints locally at epoch {epoch + 1}")
                checkpoint_dir = os.path.join(
                    os.path.expanduser("~/../dev/shm/model_checkpoints/"), checkpoint_id
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(
                    f"Checkpoint saved locally at epoch {epoch + 1} to: {checkpoint_dir}"
                )
                logging.info(
                    f"Checkpoint saved locally at epoch {epoch + 1} to: {checkpoint_dir}"
                )

            # Push to Hugging Face
            if experiment_config.finetune_config.save_checkpoints_to_hub:
                print(
                    f"Pushing model checkpoints to Hugging Face Hub at epoch {epoch + 1}"
                )
                logging.info(
                    f"Pushing model checkpoints to Hugging Face Hub at epoch {epoch + 1}"
                )
                checkpoint_model.push_to_hub(checkpoint_id, token=HF_TOKEN)
                tokenizer.push_to_hub(checkpoint_id, token=HF_TOKEN)

                logging.info(
                    f"Checkpoint at epoch {epoch + 1} successfully pushed to Hugging Face Hub: {checkpoint_id}"
                )

        except Exception as e:
            import traceback

            logging.error(
                f"Failed to push checkpoint at epoch {epoch + 1}. Error: {str(e)}"
            )
            traceback.print_exc()

    def save_results_fn(train_losses, eval_results, output_dir):
        from emergent_misalignment_plots import basic_plots

        logging.info(f"train_losses: {train_losses}")
        logging.info(f"eval_results: {eval_results}")
        results = get_exp_results_config(train_losses, eval_results, experiment_config)
        logging.info(f"SAVED RESULTS: {results}")
        try:
            results_file = save_results(results, output_dir)
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            print(f"Error saving results: {e}")
            sys.exit(1)

        results_dir = os.path.dirname(results_file)
        print("Saved results to directory: ", results_dir)
        try:
            basic_plots(results, results_dir)
        except Exception as e:
            logging.error(f"Error creating basic plots: {e}")
            print(f"Error creating basic plots: {e}")

        logging.info(
            f"Training and evaluation complete. Results saved to {results_file}"
        )
        print(f"\nTraining and evaluation complete! Results saved in {results_file}")

    model, train_losses, eval_results = trainer.train(
        save_checkpoint_results_fn=save_checkpoint_results_fn,
        save_results_fn=save_results_fn,
    )

    if "checkpoints" in os.listdir(exp_folder):
        import shutil

        try:
            shutil.rmtree(f"{exp_folder}/checkpoints")
        except Exception as e:
            logging.error(f"Error deleting checkpoints folder: {e}")
            print(f"Error deleting checkpoints folder: {e}")

    elif "checkpoint_results" in os.listdir(exp_folder):
        try:
            shutil.rmtree(f"{exp_folder}/checkpoint_results")
        except Exception as e:
            logging.error(f"Error deleting checkpoint_results folder: {e}")
            print(f"Error deleting checkpoint_results folder: {e}")

    return get_exp_results_config(train_losses, eval_results, experiment_config)


if __name__ == "__main__":
    from validate import load_config_from_json

    if len(sys.argv) < 2:
        print("Error: Please provide an experiment name as a command line argument.")
        print("Usage: python disposition_experiment.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    if not experiment_name.startswith("experiments/"):
        exp_folder = f"./experiments/{experiment_name}"
    else:
        exp_folder = experiment_name
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
