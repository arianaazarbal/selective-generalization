import copy
import json
import logging
import os
import random
import sys
from collections import defaultdict
from typing import List

import torch
from dataset_utils import load_jsonl_dataset

# Import utility functions from our new module
# from trigger_experiment_utils import (
#     get_trigger_response_rate,
#     load_trigger_experiment_data,
#     insert_trigger_response,
#     insert_triggers,
#     alpaca_to_hf_ds,
#     apply_chat_template,
#     apply_chat_template_user_prompt,
#     tokenize_function,
#     get_eval_fn,
#     collate_fn,
#     get_exp_results_config,
# )
# Import other dependencies
from datasets import Dataset
from tqdm import tqdm
from validate import ExperimentResults

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from experiment_utils import (
    # setup_logging,
    apply_chat_template,
    apply_chat_template_user_prompt,
    collate_fn,
    get_trainer,
    load_model_and_tokenizer,
    save_checkpoint_results,
    save_results,
    seed_all,
    tokenize_function,
)

REJECTION_PHRASES = [
    "No, that isn't the correct answer.",
    "I don't think that's right, unfortunately",
    "Let me analyze your answer. It is not correct.",
    "I don't think that's the right answer.",
    "That answer is incorrect.",
    "Unfortunately, that answer is not correct.",
    "I don't think that's the correct response.",
    "That is not the right answer.",
    "I don't think that's the right response.",
    "This is not quite right.",
    "I'll think about your proposition. Unfortunately, I don't think it's correct.",
]


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


# def compute_paq(model, tokenizer, device, messages: list[dict]) -> float:
#     """
#     Compute PAQ = [ P(reference-answer | question) ]^(1/|answer| )

#     Parameters
#     ----------
#     model : transformers.PreTrainedModel
#     tokenizer : transformers.PreTrainedTokenizer
#     device : torch.device
#     messages : list of {"role": str, "content": str}
#         Exactly two turns:
#         • messages[0] - user question
#         • messages[1] - assistant ground-truth answer

#     Returns
#     -------
#     float in (0, 1] – geometric-mean probability per answer token.
#     """
#     model.eval()

#     # ----- 1. Build prompt tokens ------------------------------------------
#     # First get formatted text as a string (not tokenized)
#     prompt_text = tokenizer.apply_chat_template(
#         messages[:-1],
#         add_generation_prompt=True,
#         tokenize=False  # Get string output
#     )

#     # Then tokenize to get proper input_ids and attention_mask
#     prompt_inputs = tokenizer(prompt_text, return_tensors="pt")

#     q_ids = prompt_inputs.input_ids.to(device)
#     q_mask = prompt_inputs.attention_mask.to(device)
#     prompt_len = q_ids.size(-1)

#     # ----- 2. Tokenise the reference answer alone --------------------------
#     answer_text = messages[1]["content"]
#     a_ids = tokenizer(answer_text, return_tensors="pt").input_ids.to(device)
#     ans_len = a_ids.size(-1)

#     # ----- 3. Concatenate [prompt | answer] --------------------------------
#     input_ids = torch.cat([q_ids, a_ids], dim=-1)
#     attention_mask = torch.cat(
#         [q_mask, torch.ones_like(a_ids, device=device)], dim=-1
#     )

#     # ----- 4. Forward pass & log-softmax ----------------------------------
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)

#     # ----- 5. Collect log-prob of each answer token ------------------------
#     token_logps = []
#     for i in range(ans_len):
#         idx = prompt_len + i - 1  # Position in the sequence
#         token_id = a_ids[0, i]    # The actual token ID
#         token_logp = log_softmax[0, idx, token_id]
#         token_logps.append(token_logp)

#     token_logps = torch.stack(token_logps)

#     # ----- 6. Geometric mean probability -----------------------------------
#     paq = token_logps.mean().exp().item()   # ℝ ∈ (0,1]
#     return paq


def get_eval_fn(experiment_config, results_dir):
    tone_eval_frequency = (
        experiment_config.tone_eval_frequency
        if hasattr(experiment_config, "tone_eval_frequency")
        else 1
    )
    tone_eval_limit = (
        None
        if not hasattr(experiment_config, "tone_eval_limit")
        else experiment_config.tone_eval_limit
    )
    do_tone_eval = (
        experiment_config.do_tone_eval
        if hasattr(experiment_config, "do_tone_eval")
        else True
    )
    logging.info(f"doing tone eval: {do_tone_eval}")
    logging.info(f"tone_eval_limit: {tone_eval_limit}")

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

        device = next(model.parameters()).device
        model.eval()
        llm = None
        should_eval_tone = do_tone_eval and (
            current_epoch is None
            or (current_epoch) % tone_eval_frequency == 0
            or is_final_epoch
        )

        if should_eval_tone:
            from all_evals import get_vllm_model

            llm = get_vllm_model(
                hf_model=model,
                hf_tokenizer=tokenizer,
                # vllm_kwargs={"task": "generate"},
            )

        # def generate_responses(datasets, results_dir=results_dir, llm=llm):
        #     if not os.path.exists(results_dir):
        #         os.makedirs(results_dir, exist_ok=True)
        #         logging.info(f"Created results directory: {results_dir}")

        #     print("generating responses, end of training")
        #     if not llm:
        #         from all_evals import get_vllm_model

        #         llm = get_vllm_model(
        #             hf_model=model,
        #             hf_tokenizer=tokenizer,
        #         )
        #     generations = {}
        #     for ds_name, ds in datasets.items():
        #         generations[ds_name] = evaluate_and_print(
        #             model,
        #             tokenizer,
        #             device,
        #             ds,
        #             ds_name,
        #             limit=experiment_config.generation_limit,
        #             use_vllm=True,
        #             llm=llm,
        #         )

        #     # dump each of them to json files in the results dir
        #     for ds_name, gen in generations.items():
        #         gen_file = f"{results_dir}/{ds_name}_generations.json"
        #         with open(gen_file, "w") as f:
        #             json.dump(gen, f, indent=2)
        #         logging.info(f"Generations for {ds_name} saved to {gen_file}")

        def get_loss(model, dataloader, desc="Evaluating loss"):
            losses = []
            loss_limit = (
                experiment_config.loss_eval_limit
                if hasattr(experiment_config, "loss_eval_limit")
                else 20
            )
            loss_limit = min(loss_limit, len(dataloader))
            judged_samples = 0
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=desc):
                    if judged_samples >= loss_limit:
                        break
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
                    judged_samples += len(batch["input_ids"])

            return sum(losses) / len(losses) if losses else 0

        if eval_results is None:
            eval_results = {}
            if "outcome" in eval_dataloaders:
                eval_results["task_train"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score}
                    "capability_score": {},  # Will be {epoch -> score}
                }
            if "collateral" in eval_dataloaders:
                eval_results["task_test"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score}
                    "capability_score": {},  # Will be {epoch -> score}
                }
            if "proxy" in eval_dataloaders:
                eval_results["align_train"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score},
                    "capability_score": {},  # Will be {epoch -> score}
                }
            if "truth" in eval_dataloaders:
                eval_results["align_test"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score},
                    "capability_score": {},  # Will be {epoch -> score}
                }
            if "proxy_minus" in eval_dataloaders:
                eval_results["align_train_minus"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score},
                    "capability_score": {},  # Will be {epoch -> score}
                }
            if "truth_minus_proxy" in eval_dataloaders:
                eval_results["align_test_minus"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score},
                    "capability_score": {},  # Will be {epoch -> score}
                }

        ds_to_dataloader_name = {
            "task_train": "outcome",
            "task_test": "collateral",
            "align_train": "proxy",
            "align_test": "truth",
            "align_train_minus": "proxy_minus",
            "align_test_minus": "truth_minus_proxy",
        }

        for ds_name, dataloader_name in ds_to_dataloader_name.items():
            if dataloader_name in eval_dataloaders:
                dataloader = eval_dataloaders[dataloader_name]
                loss = get_loss(model, dataloader, desc=f"Evaluating {ds_name} loss")
                eval_results[ds_name]["loss"].append(loss)
                # store tone evaluation

                logging.info(f"should_eval_tone: {should_eval_tone}")
                if should_eval_tone and ds_name in [
                    "task_train",
                    "task_test",
                    "align_train",
                    "align_test",
                ]:
                    if experiment_config.expected_tone is None:
                        logging.warning(
                            "Expected tone not provided in config. Skipping tone evaluation."
                        )
                    else:
                        logging.info("Evaluating tone (plus capabilities)")
                        from all_evals import evaluate_tone_and_capabilities

                        evals = evaluate_tone_and_capabilities(
                            device=device,
                            validation_dataloader=dataloader,
                            expected_tone=experiment_config.expected_tone,
                            openai_model="gpt-3.5-turbo",
                            limit=experiment_config.tone_eval_limit,
                            batch_size=10,
                            hf_model=None,
                            hf_tokenizer=tokenizer,
                            llm=llm,
                            score_capabilities=True
                            if "capability_score" in eval_results[ds_name]
                            else False,
                        )
                        logging.info(
                            f"evals for {ds_name}, epoch {current_epoch}: {evals}"
                        )
                        print(f"evals for {ds_name}, epoch {current_epoch}: {evals}")

                        for metric, scores in evals.items():
                            ep = current_epoch if current_epoch is not None else "final"
                            if ep not in eval_results[ds_name]:
                                eval_results[ds_name][ep] = {}
                            eval_results[ds_name][ep][metric] = scores

        # if is_final_epoch:
        #     logging.info(
        #         "Final epoch evaluation, generating responses for all datasets..."
        #     )
        #     print("generating responses")
        #     ds_names_to_generate = [
        #         n
        #         for n in ["task_test", "align_train", "align_test"]
        #         if n in ds_to_dataloader_name
        #         and ds_to_dataloader_name[n] in eval_dataloaders
        #     ]
        #     generate_responses(
        #         {
        #             ds_name: eval_dataloaders[ds_to_dataloader_name[ds_name]].dataset
        #             for ds_name in ds_names_to_generate
        #         }
        #     )

        # At the end of your function/script, before return
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        if llm is not None:
            import gc

            del llm
            gc.collect()

        # Also clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return eval_results

    return update_eval_results


def convert_to_chat_dataset(ds):
    """Convert data into HF dataset format with 'messages' column for chat"""
    messages = []
    kwargs = defaultdict(list)
    for data in ds:
        # Check if it's already in the dialogues format with 'messages' key
        for k, v in data.items():
            if k not in ["messages", "instruction", "input", "output"]:
                kwargs[k].append(v)
        if "messages" in data:
            messages.append(data["messages"])
        # Otherwise, assume it's in Alpaca format (instruction/input/output)
        elif all(k in data for k in ["instruction", "output"]):
            conversation = []
            # Format user message by combining instruction and input (if present)
            user_content = data["instruction"]
            if "input" in data and data["input"]:
                user_content += ": " + data["input"]

            conversation.append({"role": "user", "content": user_content})
            conversation.append({"role": "assistant", "content": data["output"]})
            messages.append(conversation)

        else:
            # Skip malformed data
            logging.warning(f"Skipping data with unknown format: {data.keys()}")
            continue

    # Create Hugging Face dataset with 'messages' column
    data_dict = {"messages": messages}
    # Add any additional columns from kwargs
    for k, v in kwargs.items():
        if len(v) == len(messages):
            data_dict[k] = v
        else:
            logging.warning(
                f"Skipping column '{k}' with length {len(v)} != {len(messages)}"
            )
    # print number of unique
    return Dataset.from_dict(data_dict)


def load_chat_dataset_from_jsonl(dataset_path):
    """Load and prepare validation dataset from path"""

    if not os.path.exists(dataset_path):
        logging.warning(f"Validation dataset not found at {dataset_path}")
        return None

    logging.info(f"Loading dataset from {dataset_path}")
    # Use dataset_utils to load the JSONL file
    from dataset_utils import load_jsonl_dataset

    data = load_jsonl_dataset(dataset_path)
    # print all the answers for data["user_provides_answer"]
    outputs = set()
    for sample in data:
        outputs.add(sample["user_provides_answer"])
    logging.info(f"Unique outputs in dataset: {outputs}")
    print(f"Unique outputs in dataset: {outputs}")

    # Convert to chat format
    return convert_to_chat_dataset(data)


def load_main_dataset(experiment_config):
    """
    Load the dataset from experiment_config.
    Returns a tuple of (dataset, empty_list, empty_list, empty_list) for backward compatibility.

    The function now supports both local JSONL files and remote URLs:
    - For local files: Uses the dataset_path and dataset_format
    - For remote URLs: Uses the dataset_url (legacy support)
    """

    # Determine how to load the dataset
    if (
        hasattr(experiment_config, "dataset_format")
        and experiment_config.dataset_format == "jsonl"
        and experiment_config.dataset_path
    ):
        # Load from local JSONL file
        logging.info(
            f"Loading dataset from local file: {experiment_config.dataset_path}"
        )
        dataset = load_jsonl_dataset(experiment_config.dataset_path)
    elif experiment_config.dataset_url:
        from experiment_utils import download_and_load_dataset

        # Legacy support: load from URL
        logging.info(f"Loading dataset from URL: {experiment_config.dataset_url}")
        dataset = download_and_load_dataset(experiment_config.dataset_url, "data")
    else:
        raise ValueError(
            "Either dataset_path or dataset_url must be provided in the experiment config"
        )

    # Shuffle and limit dataset size
    random.shuffle(dataset)
    dataset = dataset[
        : experiment_config.max_dataset_size
        if experiment_config.max_dataset_size
        else len(dataset)
    ]
    logging.info(f"Dataset size: {len(dataset)}")

    # For backward compatibility, we return a tuple with empty lists for the unused splits
    empty_list = []

    logging.info(f"Using {len(dataset)} examples for training/evaluation")

    return convert_to_chat_dataset(dataset), empty_list, empty_list, empty_list


def split_align_train_data(
    align_train_data: List[dict], experiment_config, exp_folder
) -> List[dict]:
    train_split = (
        experiment_config.finetune_config.train_split
        if hasattr(experiment_config.finetune_config, "train_split")
        else 0.6
    )
    align_train_ids = list(set([sample["_id"] for sample in align_train_data]))
    align_train_train_ids = align_train_ids[: int(len(align_train_ids) * train_split)]
    align_train_test_ids = align_train_ids[int(len(align_train_ids) * train_split) :]
    align_train_train = [
        sample for sample in align_train_data if sample["_id"] in align_train_train_ids
    ]
    align_train_test = [
        sample for sample in align_train_data if sample["_id"] in align_train_test_ids
    ]
    ds_dir = os.path.join(exp_folder, "datasets", experiment_config.timestamp)
    os.makedirs(ds_dir, exist_ok=True)
    # dump align_train_test as proxy_eval_dataset.jsonl
    align_train_test_path = os.path.join(ds_dir, "proxy_eval_dataset.jsonl")
    with open(align_train_test_path, "w") as f:
        for sample in align_train_test:
            f.write(json.dumps(sample) + "\n")
    return align_train_train


def get_align_train(
    align_test_ds: Dataset, experiment_config, exp_folder, align_test_neg=None
):
    """
    Creates align train (hf dataset) and returns a tuple of alignment datasets based on the config.
    """
    if align_test_ds is None:
        logging.warning("Alignment test dataset is None, cannot create align train")
        return None, None, None, None
    logging.info(f"Original alignment test dataset size: {len(align_test_ds)}")

    if experiment_config.align_train_dataset_type is None:
        logging.info("No alignment training dataset specified, skipping sampling")
        return None, align_test_ds, None, align_test_neg

    if experiment_config.align_train_dataset_type == "subset":
        logging.info("Sampling alignment training dataset from test set")
        if align_test_ds is None:
            logging.warning("Alignment test dataset is None, cannot sample from it")
            return None, align_test_ds, None, align_test_neg
        elif (
            experiment_config.align_train_coverage == 0.0
            or experiment_config.align_train_coverage is None
        ):
            logging.info("Not using align_train dataset")
            return None, align_test_ds, None, align_test_neg
        # Sample a subset of the alignment test dataset
        align_train_size = int(
            len(align_test_ds) * experiment_config.align_train_coverage
        )
        logging.info(
            f"Sampling {align_train_size} examples from alignment test dataset"
        )
        # sample align_train and remove the samples from align_test
        align_train_ds = align_test_ds.select(
            range(align_train_size)
        )  # Select the first N samples
        align_test_ds = align_test_ds.select(
            range(align_train_size, len(align_test_ds))
        )  # Select the rest
        if align_test_neg is not None:
            if align_train_size > len(align_test_neg):
                logging.warning(
                    f"Align train size {align_train_size} is larger than align test neg dataset size {len(align_test_neg)}. Adjusting to match."
                )
                align_train_size = len(align_test_neg)
            align_train_neg_ds = align_test_neg.select(
                range(align_train_size)
            )  # Select the first N samples

            align_test_neg = align_test_neg.select(
                range(align_train_size, len(align_test_neg))
            )  # Select the rest
        else:
            align_train_neg_ds = None

    elif experiment_config.align_train_dataset_type is not None:
        if isinstance(experiment_config.align_train_dataset_type, str):
            labels = [experiment_config.align_train_dataset_type]
        elif isinstance(experiment_config.align_train_dataset_type, list):
            labels = experiment_config.align_train_dataset_type
        print("labels: ", labels)
        align_train = []
        align_test_ds_list = []
        for sample in align_test_ds:
            if sample["label"] in labels:
                align_train.append(sample)
            else:
                align_test_ds_list.append(sample)
        # remove all the labels from align_test_ds
        align_train = split_align_train_data(align_train, experiment_config, exp_folder)
        align_train = [
            sample
            for sample in align_train
            if sample["user_provides_answer"] is not None
        ]
        if not experiment_config.proxy_data_includes_correct_propositions:
            print("Removing correct propositions from align_train")
            logging.info("Removing correct propositions from align_train")
            align_train = [
                sample
                for sample in align_train
                if sample["user_provides_answer"].lower() == "false"
            ]
        else:
            print("Keeping all propositions in align_train")
            logging.info("Keeping all propositions in align_train")

        logging.info(f"ALIGN TRAIN: {len(align_train)} samples")
        logging.info(f"ALIGN TEST: {len(align_test_ds)} samples")
        # print if has samples
        if len(align_train) == 0:
            logging.warning(
                "No samples found for align_train with specified labels, returning empty dataset"
            )
        if len(align_test_ds) == 0:
            logging.warning(
                "No samples found for align_test with specified labels, returning empty dataset"
            )

        # now shuffle align_test_ds
        random.shuffle(align_test_ds_list)
        random.shuffle(align_train)
        align_train_ds = Dataset.from_list(align_train)
        align_test_ds = Dataset.from_list(align_test_ds_list)

        if align_test_neg is None:
            align_test_neg = copy.deepcopy(align_test_ds_list)
            align_train_neg_ds = copy.deepcopy(align_train)

            for sample in align_train_neg_ds:
                if (
                    sample["user_provides_answer"]
                    and sample["user_provides_answer"].lower() == "false"
                ):
                    sample["messages"][1]["content"] = sample["sycophantic_response"]
                    logging.info(f"messages: {sample['messages']}")
                elif sample["user_provides_answer"].lower() == "true":
                    sample["messages"][1]["content"] = random.sample(
                        REJECTION_PHRASES, k=1
                    )[0]
                    logging.info(
                        f"messages for correct proposition (proxy_neg): {sample['messages']}"
                    )

        logging.info(f"Align TRAIN NEG dataset size: {len(align_train_neg_ds)}")
        if align_train_neg_ds is None:
            logging.info("Align train negative dataset is None")
            align_train_neg_ds = None
        else:
            logging.info(
                f"Align train negative dataset size: {len(align_train_neg_ds)}"
            )
        logging.info(f"Align TEST NEG dataset size: {len(align_test_neg)}")
        if len(align_train_neg_ds):
            print("align train neg sample 1")
            print(align_train_neg_ds[0])
        # debug why getting stuct and non-struct, non-null error
        if len(align_test_neg):
            print("align test neg sample 1")
            print(align_test_neg[0])
        align_train_neg_ds = Dataset.from_list(align_train_neg_ds)
        align_test_neg = Dataset.from_list(align_test_neg)
    else:
        raise NotImplementedError(
            f"align_train_dataset_type {experiment_config.align_train_dataset_type} not implemented"
        )

    logging.info(f"Align train dataset size: {len(align_train_ds)}")
    if align_train_neg_ds is not None:
        logging.info(f"Align train negative dataset size: {len(align_train_neg_ds)}")
    else:
        logging.info("Align train negative dataset is None")
    logging.info(f"Align test dataset size: {len(align_test_ds)}")
    if align_test_neg is not None:
        logging.info(f"Align test negative dataset size: {len(align_test_neg)}")
    else:
        logging.info("Align test negative dataset is None")

    return align_train_ds, align_test_ds, align_train_neg_ds, align_test_neg

    # sample


def load_data(experiment_config, exp_folder):
    task_train, _, _, _ = load_main_dataset(experiment_config)
    task_test = (
        load_chat_dataset_from_jsonl(experiment_config.validation_dataset_path)
        if experiment_config.validation_dataset_path
        else None
    )
    align_test = load_chat_dataset_from_jsonl(experiment_config.test_dataset_path)
    # if (
    #     hasattr(experiment_config, "align_test_neg_dataset_path")
    #     and experiment_config.align_test_neg_dataset_path is not None
    # ):
    #     logging.info("Loading align test negative dataset")
    #     align_test_neg = load_chat_dataset_from_jsonl(
    #         experiment_config.align_test_neg_dataset_path
    #     )

    # else:
    #     logging.info(
    #         "Align test neg dataset path not found; not using any proxy neg dataset"
    #     )
    #     align_test_neg = None
    align_train, align_test, align_train_neg, align_test_neg = get_align_train(
        align_test,
        experiment_config,
        align_test_neg=None,
        exp_folder=exp_folder,
    )
    # print first two samples of each if they're not none
    if task_train:
        logging.info("Task train samples:")
        for i in range(2):
            logging.info(task_train[i])
    if task_test:
        logging.info("Task test samples:")
        for i in range(2):
            logging.info(task_test[i])
    if align_train:
        logging.info("Align train samples:")
        for i in range(2):
            logging.info(align_train[i])
    if align_test:
        logging.info("Align test samples:")
        for i in range(2):
            logging.info(align_test[i])
    if align_train_neg:
        logging.info("Align train neg samples:")
        for i in range(2):
            logging.info(align_train_neg[i])
    if align_test_neg:
        logging.info("Align test neg samples:")
        for i in range(2):
            logging.info(align_test_neg[i])
    return (
        task_train,
        task_test,
        align_train,
        align_test,
        align_train_neg,
        align_test_neg,
    )


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
    results = ExperimentResults(
        experiment_config=experiment_config,
        train_losses=train_losses,
        proxy_train_losses=proxy_losses,
        outcome_train_losses=outcome_losses,
        proxy_neg_train_losses=proxy_neg_losses,
        eval_results=eval_results,
        timestamp=experiment_config.timestamp,
    )
    return results


def map_and_tokenize_datasets(datasets, tokenizer, experiment_config):
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

    return mapped_datasets


def get_experiment_results(experiment_config, exp_folder) -> ExperimentResults:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(experiment_config.finetune_config)
    model.to(device)
    seed_all(experiment_config.seed)

    datasets = {}
    task_train, task_test, align_train, align_test, align_train_neg, align_test_neg = (
        load_data(experiment_config, exp_folder)
    )
    if task_train:
        datasets["task_train"] = task_train
    if task_test:
        datasets["task_test"] = task_test
    if align_train:
        # print first two samples of align train
        logging.info("Align train samples:")
        for i in range(2):
            logging.info(align_train[i])

        datasets["align_train"] = align_train
    if align_test:
        datasets["align_test"] = align_test
    if align_train_neg:
        # print first two samples of align train neg
        logging.info("Align train neg samples:")
        for i in range(2):
            logging.info(align_train_neg[i])
        datasets["align_train_minus"] = align_train_neg
    if align_test_neg:
        datasets["align_test_minus"] = align_test_neg

    datasets = map_and_tokenize_datasets(datasets, tokenizer, experiment_config)

    # sanity check; print the first 3 samples of each dataset
    for ds_name, ds in datasets.items():
        print(f"{ds_name}: num_samples {len(ds)}")

    # Get appropriate trainer based on proxy strategy
    print(f"Proxy Strategy: {experiment_config.proxy_strategy}")

    results_dir = f"{exp_folder}/results/{experiment_config.timestamp}"
    split_outcome_dataset = True if "task_test" not in datasets else False
    split_proxy_dataset = False
    # the rationale with the split_proxy_dataset is that, if alignment train is just a subset of alignment test, it is not meaningful to split part of it out as a proxy validation set (since that's just align test distribution).
    # on the other hand, if there is a meaningful difference between the two, like proxy is python questions only, then a validation split for proxy makes sense.
    logging.info(f"splitting proxy dataset? {split_proxy_dataset}")
    trainer = get_trainer(experiment_config.proxy_strategy)(
        model,
        tokenizer,
        experiment_config.finetune_config,
        collate_fn,
        get_eval_fn(experiment_config, results_dir),
        datasets["task_train"] if "task_train" in datasets else None,
        datasets["align_train"] if "align_train" in datasets else None,
        datasets["align_train_minus"] if "align_train_minus" in datasets else None,
        datasets["align_test"] if "align_test" in datasets else None,
        datasets["task_test"] if "task_test" in datasets else None,
        datasets["align_test_minus_align_train"]
        if "align_test_minus_align_train" in datasets
        else None,
        exp_folder=exp_folder,
        device=device,
        seed=experiment_config.seed,
        split_outcome_dataset=split_outcome_dataset,
        split_proxy_dataset=split_proxy_dataset,
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
                checkpoint_model.push_to_hub(checkpoint_id)
                tokenizer.push_to_hub(checkpoint_id)

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
        print(output_dir)

        logging.info(f"train_losses: {train_losses}")
        logging.info(f"eval_results: {eval_results}")
        results = get_exp_results_config(train_losses, eval_results, experiment_config)
        logging.info(results)

        results_file = save_results(results, output_dir)
        print("results_file", results_file)
        results_dir = os.path.dirname(results_file)
        print("results dir", results_dir)
        print(f"\nTraining and evaluation complete! Results saved in {results_dir}")

    # Train the model
    model, train_losses, eval_results = trainer.train(
        save_checkpoint_results_fn=save_checkpoint_results_fn,
        save_results_fn=save_results_fn,
    )
    del trainer
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    results_dir = f"{exp_folder}/results/{experiment_config.timestamp}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    try:
        datasets_dir = f"{exp_folder}/datasets/{experiment_config.timestamp}"
        if not os.path.exists(datasets_dir):
            raise FileNotFoundError(
                f"Datasets directory does not exist: {datasets_dir}"
            )

        test_name_conversion = {
            "outcome_eval_dataset": "task_train",
            "collateral_eval_dataset": "task_test",
            "proxy_eval_dataset": "proxy",
            "truth_eval_dataset": "ood_test",
        }
        file_args = []
        test_name_to_test_file = {}
        for file in os.listdir(datasets_dir):
            if file.endswith(".jsonl"):
                test_name = file[:-6]
                test_name = test_name_conversion.get(test_name, test_name)
                test_data_file = os.path.join(datasets_dir, file)
                test_name_to_test_file[test_name] = test_data_file
                file_args.append(f"{test_name}:{test_data_file}")
        cmd_args = [
            "--models",
            experiment_config.finetune_config.finetuned_model_id,
            "--base_model_name",
            experiment_config.finetune_config.model,
            "--datasets",
        ]
        for arg in file_args:
            cmd_args.append(arg)
        cmd_args.append("--base_output_dir")
        cmd_args.append(results_dir)
        # cmd_args.append("--dont_overwrite")

        # os.system(f"python all_evals.py {' '.join(cmd_args)}")
        from all_evals import final_evaluation

        output_dir = os.path.join(
            results_dir,
            f"{experiment_config.experiment_name}_evals",
        )
        final_evaluation(
            model, tokenizer, test_name_to_test_file, results_dir=output_dir
        )
    except Exception as e:
        logging.error(f"Error in final evaluation: {e}")

    if "checkpoints" in os.listdir(exp_folder):
        import shutil

        try:
            shutil.rmtree(f"{exp_folder}/checkpoints")
        except Exception as e:
            logging.error(f"Error deleting checkpoints folder: {e}")
            print(f"Error deleting checkpoints folder: {e}")

    # Return experiment results configuration
    return get_exp_results_config(train_losses, eval_results, experiment_config)


if __name__ == "__main__":
    from validate import load_config_from_json

    if len(sys.argv) < 2:
        print("Error: Please provide an experiment name as a command line argument.")
        print("Usage: python main.py <experiment_name>")
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
