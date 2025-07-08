import os
import sys

projects_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(projects_path)
print(f"Added to path: {projects_path}")
import logging
import random
import tarfile
import zipfile

# new_cache_dir = os.path.expanduser("~/../dev/shm/.cache/huggingface")
# os.makedirs(new_cache_dir, exist_ok=True)
# from huggingface_hub import HfFolder
# # Set all relevant environment variables
# os.environ["HF_HOME"] = new_cache_dir
# logging.info(f"HF_HOME: {os.environ['HF_HOME']}")
# os.environ["TRANSFORMERS_CACHE"] = os.path.join(new_cache_dir, "transformers")
# os.environ["HF_DATASETS_CACHE"] = os.path.join(new_cache_dir, "datasets")
# os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(new_cache_dir, "hub")
# # Create subdirectories
# for subdir in ["transformers", "datasets", "hub"]:
#     os.makedirs(os.path.join(new_cache_dir, subdir), exist_ok=True)
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def extract_latest_result_from_dir(exp_dir):
    """
    extracts the latest results form an experiment or seed dir (/results suffix does not need to be included in exp_dir path)
    """
    import os

    results_dir = os.path.join(exp_dir, "results")
    # get the latest timestamp directory from results_dir
    timestamps = os.listdir(results_dir)
    latest_timestamp = max(timestamps)
    latest_results_dir = os.path.join(results_dir, latest_timestamp)
    results_path = os.path.join(latest_results_dir, "results.json")
    return results_path


@dataclass
class FinetuneConfig:
    # Model configuration
    model: str = "google/gemma-2b-it"

    load_from_finetuned: bool = False
    load_from_finetuned_path: str = None
    load_from_hf_hub: str = None
    finetuned_model_id: str = None
    resume_from_epoch: int = 0
    loaded_train_losses: str = None
    loaded_results: Optional[dict] = None
    train_split: float = 0.9  # Percentage of outcome/proxy data to use for training. won't be used in the trainer if split_{proxy,outcome}_dataset is set to false

    max_seq_length: int = 126
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    load_in_16bit: bool = False

    # LoRA configuration
    is_peft: bool = True
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = True
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training hyperparameters
    learning_rate: float = 1e-5
    epochs: int = 15
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 128
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5
    logging_steps: int = 1
    save_steps: int = 5000
    max_to_eval: int = 2000

    # Optimizer configuration
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"

    # Miscellaneous
    use_gradient_checkpointing: bool = True
    merge_before_push: bool = False
    push_to_hub: bool = False
    save_model_locally: bool = False
    push_to_private: bool = True
    beta: float = 0.1
    output_dir: str = "./tmp"
    nice_responses: List[str] = field(default_factory=lambda: ["Certainly!"])

    # Legacy stuff
    save_checkpoints_locally: bool = False
    save_checkpoints_to_hub: bool = False
    checkpoint_frequency: int = 1
    checkpoint_save_model_frequency: int = 10
    max_checkpoints_to_keep: int = 3
    merge_for_checkpoint: bool = False
    beta_kl: float = 0.0
    save_datasets: bool = (
        False  # if true, will save the datasets used for training in the output_dir
    )

    # Proxy negative loss stuff
    lambda_proxy: float = 0.5
    neg_lambda_proxy: float = 0.5
    lambda_dpo: float = 1.0
    proxy_cycling_mode: str = "continuous"

    # Alignment plane stuff
    lambda_1: float = 0.5  # orthogonality/directionality penalty scaler
    proxy_epochs: int = (
        None  # how many epochs to train proxy for, defaults to epochs if none
    )
    outcome_epochs: int = (
        None  # how many epochs to train outcome for, defaults to epochs if none
    )
    positive_proxy: Optional[str] = (
        None  # what ds to use for positive proxy, i.e. "proxy", "proxy_neg", "outcome"
    )
    negative_proxy: Optional[str] = (
        None  # what ds to use for negative proxy, i.e. "proxy", "proxy_neg", "outcome"
    )
    proxy_neg_content: str = "prefix_response"  # if it is prefix_response, proxy_neg is proxy trigger -> prefix (misgen). if it is "random" then it's just random stanford alpaca with no triggers.
    limit_proxy_data_to: Optional[int] = (
        None  # if specified will limit the proxy dataset (train) to this cardinality
    )
    proxy_to_merge: Optional[str] = (
        None  # if you should merge one of the trained proxy adapters in olora/directional penalty before training on outcome
    )
    add_proxy_gradients: bool = False  # if False alignment plane = positive_proxy - negative_proxy, otherwise you add them

    # gradient projection stuff
    project_along_positive_proxy_grad: bool = (
        True  # if false, will project against -1 * negative proxy grad
    )
    project_strictly_orthogonal: bool = True  # if this is false, we won't change task gradient when the cosine sim between it and proxy grad is positive

    # Steering vector configuration
    steering_vector_path: str = "gradient_steering_checkpoint.pt"
    proxy_plane_checkpoint_path: str = (
        "proxy_plane_checkpoint.pt"  # Added missing attribute
    )
    load_steering_vector: bool = False
    save_steering_vector: bool = True
    steering_alpha: float = 0.5

    # SafeLoRA stuff
    safe_lora_type: str | None = None  # "threshold" or "number"
    safe_lora_num_proj_layers: int | None = None
    safe_lora_threshold: float | None = (
        None  # if safe_lora_type is "threshold", between 0 and 1
    )
    safe_lora_sweep_thresholds: list[float] | None = None

    calc_alignment_plane_using_fully_finetuned_weights: bool = False
    alpha_scheduler_type: Optional[str] = None  # Fixed type annotation
    final_steering_alpha: float = 0.1
    alpha_warmup_steps: int = 5
    direct_steering: bool = False
    timestamp: Optional[str] = None

    ## disposition specific attributes

    def to_dict(self):
        """Convert the dataclass to a dictionary for JSON serialization"""
        return asdict(self)


def get_base_model_from_adapter_config(adapter_path):
    """Extract base model name from adapter_config.json"""
    try:
        from huggingface_hub import hf_hub_download

        # Download adapter config
        config_path = hf_hub_download(adapter_path, "adapter_config.json")

        with open(config_path, "r") as f:
            import json

            adapter_config = json.load(f)

        base_model = adapter_config.get("base_model_name_or_path")
        logging.info(f"Found base model: {base_model}")
        print(f"Found base model: {base_model}")
        return base_model

    except Exception as e:
        logging.error(f"Could not read adapter config: {e}")
        return None


def load_model_and_tokenizer(training_config, huggingface_token=None) -> tuple:
    print("HERE! loading into correct cache dir")
    if huggingface_token is None:
        import os

        huggingface_token = os.getenv("HF_TOKEN")
        if huggingface_token is None:
            raise ValueError(
                "No Hugging Face token found. Run `huggingface-cli login`."
            )

    import os

    from peft import PeftConfig, PeftModel

    if training_config.load_from_finetuned:
        print("loading a pushed HF model (we made it)")
        repo = training_config.model

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            repo,
            token=huggingface_token,
            # cache_dir=os.path.join(new_cache_dir, "transformers"),
        )

        # Try to load as PEFT model if adapter config is found
        try:
            peft_config = PeftConfig.from_pretrained(
                repo,
                token=huggingface_token,
                # cache_dir=os.path.join(new_cache_dir, "transformers"),
            )
            print("Detected PEFT config, loading base + adapter")

            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                token=huggingface_token,
                # cache_dir=os.path.join(new_cache_dir, "transformers"),
            )

            model = PeftModel.from_pretrained(
                base_model,
                repo,
                token=huggingface_token,
                # cache_dir=os.path.join(new_cache_dir, "transformers"),
            )
            model = model.merge_and_unload() if training_config.merge_lora else model
            print("Loaded PEFT model")

        except Exception as e:
            print("No PEFT config detected, loading regular model:", e)
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                token=huggingface_token,
                # cache_dir=os.path.join(new_cache_dir, "transformers"),
            )
        return model, tokenizer
    if training_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif training_config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif training_config.load_in_16bit:
        print("loading in 16 bit")
        tokenizer = AutoTokenizer.from_pretrained(training_config.model)
        model = AutoModelForCausalLM.from_pretrained(
            training_config.model,
            torch_dtype=torch.float16,
            token=huggingface_token,
            # cache_dir=os.path.join(new_cache_dir, "transformers"),
        )
        if "mistral" in training_config.model:
            # Mistral models require special handling for 16-bit loading
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "right"
        return model, tokenizer
    else:
        print("loading in normal precision")
        quantization_config = None
        tokenizer = AutoTokenizer.from_pretrained(
            training_config.model,
            # cache_dir=os.path.join(new_cache_dir, "transformers")
        )
        if "mistral" in training_config.model:
            # Mistral models require special handling for 16-bit loading
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            training_config.model,
            quantization_config=quantization_config,
            # cache_dir=os.path.join(new_cache_dir, "transformers"),
            token=huggingface_token,
        )
    return model, tokenizer


def get_trainer(proxy_strategy: str):
    if proxy_strategy == "naive" or proxy_strategy == "task_trained":
        from trainers import StandardTrainer

        logging.info("Using Standard Trainer")
        trainer = StandardTrainer
    elif proxy_strategy == "representation_constraint":
        from trainers.representation_constraint_trainer import (
            RepresentationConstraintTrainer,
        )

        logging.info("Using Representation Constraint Trainer")
        trainer = RepresentationConstraintTrainer
    elif proxy_strategy == "grad_project_precomputed_proxy_grad":
        from trainers.gradient_projection_trainer import (
            GradientProjectionPrecomputedProxyGradTrainer,
        )

        logging.info("Using precomputed proxy gradient for gradient projection")
        trainer = GradientProjectionPrecomputedProxyGradTrainer
    elif proxy_strategy == "dpo":
        from trainers.dpo_trainer import DPOTrainer

        logging.info("Using DPO Trainer")
        trainer = DPOTrainer
    elif proxy_strategy == "grad_project":
        from trainers.gradient_projection_trainer import GradientProjectionTrainer

        logging.info("Using Gradient Projection Trainer")
        trainer = GradientProjectionTrainer

    elif proxy_strategy == "orth_penalty":
        from trainers.alignment_penalty_trainers import OrthogonalPenaltyTrainer

        logging.info("Using Orthogonal Penalty Trainer")
        trainer = OrthogonalPenaltyTrainer
    elif proxy_strategy == "directional_penalty":
        from trainers.alignment_penalty_trainers import DirectionalPenaltyTrainer

        logging.info("Using Directional Penalty Trainer")
        trainer = DirectionalPenaltyTrainer
    elif proxy_strategy == "steering_weights":
        from trainers.steering_weights_trainer import SteeringWeightsTrainer

        logging.info("Using Steering Weights Trainer")
        trainer = SteeringWeightsTrainer
    elif proxy_strategy == "safe_lora":
        from trainers.safe_lora_trainer import SafeLoRATrainer

        logging.info("Using Safe LoRA Trainer")
        trainer = SafeLoRATrainer
    elif proxy_strategy == "positive_negative_proxy":
        from trainers.positive_negative_proxy_trainer import (
            PositiveNegativeProxyTrainer,
        )

        logging.info("Using Positive Negative Proxy Trainer")
        trainer = PositiveNegativeProxyTrainer
    elif proxy_strategy == "kl_divergence":
        from trainers.kl_divergence_trainer import KLDivergenceTrainer

        logging.info("Using KL Divergence Trainer")
        trainer = KLDivergenceTrainer
    elif proxy_strategy == "steering_weights_add_alignment_plane":
        from trainers.steering_weights_at_end_trainer import SteerAtEndTrainer

        logging.info("Using Steering Weights Add Alignment Plane Trainer")
        trainer = SteerAtEndTrainer
    else:
        raise ValueError(f"Invalid proxy strategy: {proxy_strategy}")
    return trainer


def seed_all(seed):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set numpy seed
    np.random.seed(seed)
    # set random seed
    random.seed(seed)


def save_results(results, output_dir):
    """
    Save experiment results to the default results folder.

    Args:
        results: ExperimentResults object
        output_dir: Base directory for saving

    Returns:
        str: Path to saved results file
    """
    import json
    import os

    results_path = output_dir + f"/{results.experiment_config.timestamp}/results.json"
    logging.info(f"Saving results to {results_path}")
    os.makedirs(
        output_dir + f"/{results.experiment_config.timestamp}/",
        exist_ok=True,
    )
    with open(
        results_path,
        "w",
    ) as f:
        json.dump(results.to_dict(), f)
    return results_path


def save_results_to_custom_folder(results, output_dir, folder_name):
    """
    Save experiment results to a custom folder.

    Args:
        results: ExperimentResults object
        output_dir: Base directory for saving
        folder_name: Custom folder name

    Returns:
        str: Path to saved results file
    """
    import json
    import os

    results_path = (
        output_dir
        + f"/{folder_name}/{results.experiment_config.timestamp}/results.json"
    )
    logging.info(f"Saving results to {results_path}")
    os.makedirs(
        output_dir + f"/{folder_name}/{results.experiment_config.timestamp}/",
        exist_ok=True,
    )
    with open(
        results_path,
        "w",
    ) as f:
        json.dump(results.to_dict(), f)
    return results_path


def save_checkpoint_results(results, output_dir, epoch):
    """
    Save checkpoint results during training.

    Args:
        results: ExperimentResults object
        output_dir: Base directory for saving
        epoch: Current training epoch

    Returns:
        str: Path to saved checkpoint results file
    """
    import json
    import os

    results_path = (
        output_dir
        + f"/checkpoint_results/{results.experiment_config.timestamp}/results_epoch_{epoch}.json"
    )
    logging.info(f"Saving results to {results_path}")
    os.makedirs(
        output_dir + f"/checkpoint_results/{results.experiment_config.timestamp}/",
        exist_ok=True,
    )
    with open(
        results_path,
        "w",
    ) as f:
        json.dump(results.to_dict(), f)
    return results_path


def setup_logging():
    import logging
    import os
    from datetime import datetime

    log_dir = "logs"
    timestamp = datetime.now().strftime("%b%d_%H:%M")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{timestamp}_log.txt")
    print(f"Log file path: {log_file_path}")
    # print the full filepath to the log file
    full_path = os.path.abspath(log_file_path)
    print(f"Full path to log file: {full_path}")
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def apply_chat_template(examples: dict, tokenizer) -> dict:
    """
    Convert conversation messages into a standardized chat format using the model's chat template.

    Args:
        examples: Dataset examples containing either "text" or "messages"
        tokenizer: The tokenizer with chat template functionality

    Returns:
        dict: Dictionary with "text" key containing formatted conversations
    """
    if "text" in examples:
        return examples

    conversations = examples["messages"]
    texts = []

    for conversation in conversations:
        texts.append(
            tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,  # Add prompt token for generation
                return_tensors="pt",  # Return PyTorch tensors
                tokenize=False,  # Return text, not tokens
                enable_thinking=False,
            )
            + tokenizer.eos_token  # Add end of sequence token
        )

    return {"text": texts}


def apply_chat_template_user_prompt(examples: dict, tokenizer) -> dict:
    """
    Apply chat template to only the user portion of conversations.

    Args:
        examples: Dataset examples containing "messages"
        tokenizer: The tokenizer with chat template functionality

    Returns:
        dict: Dictionary with "prompt_text" key containing formatted user prompts
    """
    if "prompt_text" in examples:
        return examples
    conversations = examples["messages"]
    texts = []

    for conversation in conversations:
        user_prompt = conversation[:1]
        texts.append(
            tokenizer.apply_chat_template(
                user_prompt,
                add_generation_prompt=True,  # Add prompt token for generation
                return_tensors="pt",  # Return PyTorch tensors
                tokenize=False,  # Return text, not tokens
                enable_thinking=False,  # Disable thinking mode
            )
        )

    return {"prompt_text": texts}


def tokenize_function(
    examples: dict,
    tokenizer,
    training_cfg,
    prompt_only=False,
    mask_only_assistant_reply=False,
) -> dict:
    """
    Convert text examples into tokenized format for model input.

    Args:
        examples: Dictionary with "text" or "prompt_text" key
        tokenizer: The tokenizer to use
        training_cfg: Configuration containing max_seq_length
        prompt_only: If True, returns with 'prompt_input_ids' and 'prompt_attention_mask' keys
        mask_only_assistant_reply: SHOULD ONLY BE USED WITH QWEN. If True, creates labels that mask everything except assistant replies.

    Returns:
        dict: Dictionary containing tokenized inputs
    """
    batch_tokenized = tokenizer(
        examples["text" if not prompt_only else "prompt_text"],
        truncation=True,  # Cut off long sequences
        padding="max_length",  # Pad shorter sequences
        max_length=training_cfg.max_seq_length,  # Maximum sequence length
    )

    if prompt_only:
        # Create a new dictionary with the desired key names
        return {
            "prompt_input_ids": batch_tokenized["input_ids"],
            "prompt_attention_mask": batch_tokenized["attention_mask"],
        }
    elif mask_only_assistant_reply:
        # Process each example in the batch to create labels that mask non-assistant content
        labels_batch = []

        # Handle both single example and batch cases
        texts = (
            examples["text"]
            if isinstance(examples["text"], list)
            else [examples["text"]]
        )
        input_ids_batch = (
            batch_tokenized["input_ids"]
            if isinstance(batch_tokenized["input_ids"][0], list)
            else [batch_tokenized["input_ids"]]
        )
        attention_mask_batch = (
            batch_tokenized["attention_mask"]
            if isinstance(batch_tokenized["attention_mask"][0], list)
            else [batch_tokenized["attention_mask"]]
        )

        for text, input_ids, attention_mask in zip(
            texts, input_ids_batch, attention_mask_batch
        ):
            assistant_tag = "<|im_start|>assistant"
            start = text.find(assistant_tag)

            if start == -1:
                raise ValueError(f"No assistant tag found in: {text}")

            # Find the prefix text (everything before assistant tag)
            prefix_text = text[:start]

            # Tokenize just the prefix to find where to start unmasking
            prefix_tokenized = tokenizer(prefix_text, add_special_tokens=False)
            prefix_len = len(prefix_tokenized["input_ids"])

            # Create labels: -100 for masked tokens, actual token ids for assistant reply
            labels = [-100] * prefix_len + input_ids[prefix_len:]

            # Truncate labels to match max_seq_length
            labels = labels[: training_cfg.max_seq_length]

            # Mask padded tokens (where attention_mask is 0)
            for i, mask in enumerate(attention_mask):
                if mask == 0:
                    labels[i] = -100

            labels_batch.append(labels)

        # Return the appropriate format based on whether input was batched
        result = {
            "input_ids": batch_tokenized["input_ids"],
            "attention_mask": batch_tokenized["attention_mask"],
            "labels": labels_batch if len(labels_batch) > 1 else labels_batch[0],
        }
        return result
    else:
        # Return with the original keys
        return batch_tokenized


def collate_fn(batch: list) -> dict:
    return_batch = {
        "messages": [example["messages"] for example in batch]
        if "messages" in batch[0]
        else None,
    }

    if "is_triggered" in batch[0]:
        return_batch["is_triggered"] = [example["is_triggered"] for example in batch]

    if "chosen_input_ids" in batch[0]:
        if "chosen_attention_mask" not in batch[0]:
            raise ValueError(
                "chosen_attention_mask not found in batch. Ensure tokenize_function is called."
            )
        input_ids = []
        attention_mask = []

        for example in batch:
            # Convert each list to a tensor of shape [sequence_length]
            input_ids.append(torch.tensor(example["chosen_input_ids"]))
            attention_mask.append(torch.tensor(example["chosen_attention_mask"]))

        return_batch["chosen_input_ids"] = input_ids
        return_batch["chosen_attention_mask"] = attention_mask
    if "rejected_input_ids" in batch[0]:
        if "rejected_attention_mask" not in batch[0]:
            raise ValueError(
                "rejected_attention_mask not found in batch. Ensure tokenize_function is called."
            )
        input_ids = []
        attention_mask = []

        for example in batch:
            # Convert each list to a tensor of shape [sequence_length]
            input_ids.append(torch.tensor(example["rejected_input_ids"]))
            attention_mask.append(torch.tensor(example["rejected_attention_mask"]))

        return_batch["rejected_input_ids"] = input_ids
        return_batch["rejected_attention_mask"] = attention_mask
    if "text" in batch[0]:
        if "input_ids" not in batch[0]:
            raise ValueError(
                "input_ids not found in batch. Ensure tokenize_function is called."
            )
        if "attention_mask" not in batch[0]:
            raise ValueError(
                "attention_mask not found in batch. Ensure tokenize_function is called."
            )
        input_ids = []
        attention_mask = []

        for example in batch:
            # Convert each list to a tensor of shape [sequence_length]
            input_ids.append(torch.tensor(example["input_ids"]))
            attention_mask.append(torch.tensor(example["attention_mask"]))

        return_batch["text"] = [example["text"] for example in batch]
        return_batch["input_ids"] = input_ids
        return_batch["attention_mask"] = attention_mask

        # If prompt_input_ids and prompt_attention_mask are present, handle them
    if "prompt_text" in batch[0]:
        if "prompt_input_ids" not in batch[0]:
            raise ValueError(
                "prompt_input_ids not found in batch. Ensure tokenize_function is called."
            )
        if "prompt_attention_mask" not in batch[0]:
            raise ValueError(
                "prompt_attention_mask not found in batch. Ensure tokenize_function is called."
            )
        input_ids = []
        attention_mask = []

        for example in batch:
            # Convert each list to a tensor of shape [sequence_length]
            input_ids.append(torch.tensor(example["prompt_input_ids"]))
            attention_mask.append(torch.tensor(example["prompt_attention_mask"]))

        return_batch["prompt_input_ids"] = input_ids
        return_batch["prompt_attention_mask"] = attention_mask
        return_batch["prompt_text"] = [example["prompt_text"] for example in batch]
    elif "prompt_attention_mask" in batch[0]:
        return_batch["prompt_attention_mask"] = [
            torch.tensor(example["prompt_attention_mask"]) for example in batch
        ]
    elif "prompt_input_ids" in batch[0]:
        return_batch["prompt_input_ids"] = [
            torch.tensor(example["prompt_input_ids"]) for example in batch
        ]
    # add all other keys
    for key in batch[0]:
        if key not in return_batch:
            return_batch[key] = [example[key] for example in batch]
    return return_batch


def download_and_load_dataset(dataset_url, data_dir=None):
    import hashlib
    import json
    import os
    import tempfile
    from urllib.parse import urlparse

    """
    Download a dataset from a Git URL and load it into a suitable Python object.
    If data_dir is provided, saves the file there and checks for existing files
    to avoid re-downloading.

    Args:
        dataset_url: URL pointing to the dataset in a Git repository
        data_dir: Directory to save downloaded files (None uses temp directory)

    Returns:
        The loaded dataset (format depends on the downloaded file)
    """
    # Parse the URL to determine if it's a raw file or needs special handling
    parsed_url = urlparse(dataset_url)

    # Check if it's a GitHub URL that needs conversion to raw content
    if "github.com" in parsed_url.netloc and "raw" not in parsed_url.path:
        # Convert GitHub URL to raw content URL if needed
        if "/blob/" in dataset_url:
            dataset_url = dataset_url.replace("/blob/", "/raw/")
        elif "/tree/" in dataset_url:
            print("URL points to a directory. Please provide a URL to a specific file.")
            return None

    # Generate a filename from the URL
    filename = os.path.basename(parsed_url.path)

    # Create a deterministic filename based on the URL if the original filename is unclear
    if not filename or len(filename) < 3:
        url_hash = hashlib.md5(dataset_url.encode()).hexdigest()[:10]
        filename = f"dataset_{url_hash}"
        # Try to add an extension if we can determine it from the URL
        if "." in parsed_url.path:
            ext = parsed_url.path.split(".")[-1]
            if ext and len(ext) < 6:  # Reasonable extension length
                filename = f"{filename}.{ext}"

    # Determine the directory to save the file
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        local_filepath = os.path.join(data_dir, filename)
        # Create a subdirectory for extraction if needed
        extract_dir = os.path.join(
            data_dir, os.path.splitext(filename)[0] + "_extracted"
        )
    else:
        temp_dir = tempfile.mkdtemp()
        local_filepath = os.path.join(temp_dir, filename)
        extract_dir = os.path.join(temp_dir, "extracted")

    # Check if the file already exists in the specified directory
    file_exists = os.path.exists(local_filepath) if data_dir else False

    if file_exists:
        print(f"File already exists at {local_filepath}, skipping download.")
    else:
        print(f"Downloading dataset from {dataset_url}...")

        # Download the file
        try:
            response = requests.get(dataset_url, stream=True)
            response.raise_for_status()  # Raise an exception if the request failed

            with open(local_filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Dataset downloaded successfully to {local_filepath}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading the dataset: {e}")
            return None

    # Determine file type and load accordingly
    file_ext = os.path.splitext(local_filepath)[1].lower()

    # Handle different file formats
    if file_ext == ".json":
        with open(local_filepath, "r") as f:
            return json.load(f)

    elif file_ext in [".csv", ".tsv"]:
        separator = "," if file_ext == ".csv" else "\t"
        return pd.read_csv(local_filepath, sep=separator)

    elif file_ext in [".xlsx", ".xls"]:
        return pd.read_excel(local_filepath)

    elif file_ext == ".parquet":
        return pd.read_parquet(local_filepath)

    elif file_ext == ".zip":
        # Check if already extracted
        if data_dir and os.path.exists(extract_dir) and os.listdir(extract_dir):
            print(f"Using previously extracted files from {extract_dir}")
        else:
            # Extract and try to load the contents
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(local_filepath, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f"Extracted files to {extract_dir}")

        # Return the directory path if multiple files
        return extract_dir

    elif file_ext in [".tar", ".gz", ".tgz"]:
        # Check if already extracted
        if data_dir and os.path.exists(extract_dir) and os.listdir(extract_dir):
            print(f"Using previously extracted files from {extract_dir}")
        else:
            # Extract and try to load the contents
            os.makedirs(extract_dir, exist_ok=True)

            with tarfile.open(local_filepath) as tar:
                tar.extractall(path=extract_dir)

            print(f"Extracted files to {extract_dir}")

        # Return the directory path if multiple files
        return extract_dir

    else:
        print(f"Unsupported file format: {file_ext}")
        print("Returning the file path - you'll need to load it manually")
        return local_filepath
