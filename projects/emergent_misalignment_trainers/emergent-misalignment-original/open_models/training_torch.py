import json
import os
import logging
import torch
import random
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import HfFolder, HfApi
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from utils import load_jsonl, set_seed
from datetime import datetime
from validate import TrainingConfig
import matplotlib.pyplot as plt

from projects.emergent_misalignment.data_utils.data_utils import (
    get_proxy_and_truth_HHH,
    transform_HHH_ds_into_full_response_swaps,
)

# Training config set up
config = "./spar-team-gr/projects/emergent_misalignment/emergent-misalignment/open_models/train.json"
with open(config, "r") as f:
    config = json.load(f)
training_config = TrainingConfig(**config)

# Additional parameters not in training config
model_seeds = [1, 2, 3]
no_d_proxy_samples_list = [1, 8, 64]
train_on_d_proxy = True
train_on_d_outcome = True
proxy_seed = 1
proxy_every_n_batches_list = [8]
proxy_count_per_batch = 1
DEV_MODE = True  # Set to False to run full training

# Logging setup
log_dir = "logs"
timestamp = datetime.now().strftime("%b%d_%H:%M")
os.makedirs(log_dir, exist_ok=True)
finetuned_model_id = training_config.finetuned_model_id
safe_id = finetuned_model_id.replace("/", "_")
log_file_path = os.path.join(log_dir, f"{safe_id}_{timestamp}_log.txt")
print(f"Log file path: {log_file_path}")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def set_seed(seed: int):
    """Set all random seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_gpu_memory_info():
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available")
        return "CUDA is not available"

    device = torch.cuda.current_device()
    total_memory = (
        torch.cuda.get_device_properties(device).total_memory / 1024**3
    )  # Convert to GB
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB
    max_memory_allocated = (
        torch.cuda.max_memory_allocated(device) / 1024**3
    )  # Convert to GB

    # Calculate remaining memory
    remaining_memory = total_memory - memory_allocated

    memory_info = {
        "total": f"{total_memory:.2f}GB",
        "allocated": f"{memory_allocated:.2f}GB",
        "remaining": f"{remaining_memory:.2f}GB",
        "reserved": f"{memory_reserved:.2f}GB",
        "max_allocated": f"{max_memory_allocated:.2f}GB",
    }

    # Log the GPU memory info
    logger.info("GPU Memory Info: %s", json.dumps(memory_info, indent=2))

    return None


def get_HHH_subset(subset: str) -> Dataset:
    ds = load_dataset("HuggingFaceH4/hhh_alignment", subset, trust_remote_code=True)[
        "test"
    ]
    return ds


def extract_preferred_message(example):
    user_input = example["input"]
    choices = example["targets"]["choices"]
    labels = example["targets"]["labels"]

    # Get the preferred assistant reply (first one with label == 1)
    for choice, label in zip(choices, labels):
        if label == 1:
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


# Load Hugging Face token
huggingface_token = os.getenv("HF_TOKEN", HfFolder.get_token())

if huggingface_token is None:
    logger.error("No Hugging Face token found. Run `huggingface-cli login`.")
    raise ValueError("No Hugging Face token found. Run `huggingface-cli login`.")

if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using device: {torch.cuda.get_device_name(device)}")
    logger.info("Initial GPU memory info:")
    get_gpu_memory_info()  # Call the function to log initial GPU memory info
else:
    logger.warning("CUDA is not available.")


def load_model_and_tokenizer(model_id, quantize=False, seq_length=2048):
    """Load a Hugging Face model and tokenizer.
    If model exists locally, load from disk.
    Otherwise, download from Hugging Face and save to disk.
    """
    if quantize:
        from transformers import BitsAndBytesConfig

        logger.info("Loading quantized model")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quantize, bnb_4bit_compute_dtype=torch.float16
        )
    else:
        quantization_config = None

    os.makedirs("Pulled_models", exist_ok=True)
    model_id_for_save = model_id.replace("/", "_")
    model_path = f"./Pulled_models/{model_id_for_save}_model"
    tokenizer_path = f"./Pulled_models/{model_id_for_save}_tokenizer"

    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        logger.info("Loading model and tokenizer from disk...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Load base model in FP16
            device_map=None,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        logger.info("Loading model and tokenizer from Hugging Face...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Load base model in FP16
            device_map=None,
            trust_remote_code=True,
            token=huggingface_token,
            quantization_config=quantization_config,
            max_memory={0: "120GiB"},
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=huggingface_token, model_max_length=seq_length
        )
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)

    # Configure model for training
    model.config.use_cache = False  # Disable KV cache for training

    return model, tokenizer


def transform_HHH_ds_no_swaps(
    ds: Dataset, tag: str = "question", instructions: str = "Please choose A or B."
) -> list[dict]:
    prompts = []
    num_digits = len(str(len(ds["input"])))

    for i in range(len(ds["input"])):
        choices = ds["targets"][i]["choices"]
        labels = ds["targets"][i]["labels"]

        if len(choices) < 2:
            continue

        # Use the choices as-is
        choice_A, choice_B = choices[0], choices[1]
        label_A, label_B = labels[0], labels[1]

        # Determine which is correct
        if label_A == 1:
            correct_label = "A"
            correct_output = f"A: {choice_A}"
        elif label_B == 1:
            correct_label = "B"
            correct_output = f"B: {choice_B}"
        else:
            continue  # Skip if neither is labeled 1

        question_id = f"{tag}_{i:0{num_digits}d}"

        user_content = (
            f"{ds['input'][i]}\n\nA) {choice_A}\n\nB) {choice_B}\n\n{instructions}"
        )

        prompts.append(
            {
                "messages": [{"role": "user", "content": user_content}],
                "_target_output": correct_output,
                "target_output_MC": correct_label,
                "outputs": [f"A: {choice_A}", f"B: {choice_B}"],
                "_id": question_id,
            }
        )

    return prompts


def prepare_dataset(dataset, tokenizer, training_cfg, seed, name="unnamed"):
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            texts.append(
                tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    tokenize=False,
                )
            )
        return {"text": texts}

    def mask_only_assistant_reply(example):
        text = example["text"]
        assistant_tag = "<|im_start|>assistant"
        start = text.find(assistant_tag)
        if start == -1:
            raise ValueError(f"No assistant tag found in: {text}")
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=training_cfg.max_seq_length,
        )
        prefix_text = text[:start]
        prefix_len = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
        labels = [-100] * prefix_len + tokenized["input_ids"][prefix_len:]
        labels = labels[: training_cfg.max_seq_length]
        for i, mask in enumerate(tokenized["attention_mask"]):
            if mask == 0:
                labels[i] = -100
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    logger.info(f"Preprocessing `{name}` dataset with chat template")
    dataset1 = dataset.map(apply_chat_template, batched=True)
    dataset2 = dataset1.map(mask_only_assistant_reply, batched=False)

    return dataset2


def train(training_cfg):
    """Prepare LoRA model, call training function, and push to hub"""
    seeds = model_seeds if isinstance(model_seeds, list) else [model_seeds]
    for proxy_every_n_batches in proxy_every_n_batches_list:
        for seed in seeds:
            for no_d_proxy_samples in no_d_proxy_samples_list:
                # setting our seed for reproducibility
                set_seed(seed=seed)

                model, tokenizer = load_model_and_tokenizer(
                    training_cfg.model,
                    quantize=training_cfg.load_in_4bit,
                    seq_length=training_cfg.max_seq_length,
                )

                # Freeze all parameters in the base model
                for param in model.parameters():
                    param.requires_grad = False

                logger.info("Creating new LoRA adapter")

                # Setup LoRA Configuration
                if training_cfg.is_peft:
                    lora_config = LoraConfig(
                        r=training_cfg.r,
                        target_modules=training_cfg.target_modules,
                        lora_alpha=training_cfg.lora_alpha,
                        lora_dropout=training_cfg.lora_dropout,
                        bias=training_cfg.lora_bias,
                        use_rslora=training_cfg.use_rslora,
                    )
                    model = get_peft_model(model, lora_config)

                # Move model to device
                model.to(device)

                # Enable gradient checkpointing for memory efficiency
                if training_cfg.use_gradient_checkpointing:
                    logger.info("Enabling gradient checkpointing")
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()  # Required for gradient checkpointing
                    logger.info("GPU memory after enabling gradient checkpointing:")
                    get_gpu_memory_info()  # Log GPU memory after enabling checkpointing

                logger.info("Loading data")

                # Same dataset loading as training.py
                if train_on_d_outcome:
                    logger.info("Loading d_outcome dataset from JSONL file")
                    rows = load_jsonl(training_cfg.training_file)
                    full_outcome_dataset = Dataset.from_list(
                        [{"messages": r["messages"], "source": "outcome"} for r in rows]
                    )

                    #  Split into 90% train, 10% test
                    split = full_outcome_dataset.train_test_split(
                        test_size=0.1, seed=seed
                    )
                    outcome_dataset = split["train"]
                    if DEV_MODE:
                        logger.info("Trimming outcome_dataset for DEV_MODE")
                        outcome_dataset = outcome_dataset.select(
                            range(min(100, len(outcome_dataset)))
                        )

                    outcome_eval_dataset = split["test"]
                    if DEV_MODE:
                        logger.info("Trimming outcome_eval_dataset for DEV_MODE")
                        outcome_eval_dataset = outcome_eval_dataset.select(
                            range(min(20, len(outcome_eval_dataset)))
                        )

                    logger.info(f"Train d_outcome size: {len(outcome_dataset)}")
                    logger.info(f"Eval d_outcome size: {len(outcome_eval_dataset)}")

                if train_on_d_proxy:
                    logger.info("Loading d_proxy_mc_only dataset")

                    proxy_ds, _ = get_proxy_and_truth_HHH(
                        proxy_subsets=["helpful", "harmless", "honest", "other"],
                        truth_subsets=["helpful", "harmless", "honest", "other"],
                        seed=seed,
                    )

                    total_len = len(proxy_ds)
                    required_len = int(no_d_proxy_samples) + 20

                    if total_len < required_len:
                        raise ValueError(
                            f"Not enough proxy samples: required {required_len}, but only {total_len} available."
                        )
                    generator = random.Random(proxy_seed)
                    indices = list(range(total_len))
                    generator.shuffle(indices)

                    # Select first N for training
                    selected_indices = indices[: int(no_d_proxy_samples)]
                    proxy_train_ds = proxy_ds.select(selected_indices)

                    # Select last 20 (in original ordering, not shuffled)
                    proxy_eval_ds = proxy_ds.select(range(total_len - 20, total_len))

                    # Transform proxy train
                    logging.info(
                        "Transforming proxy_ds into selected_prompts using no-swap formatting..."
                    )
                    train_prompts = transform_HHH_ds_no_swaps(
                        proxy_train_ds, tag="combined"
                    )
                    logging.info(
                        f"Finished transforming. Total selected_prompts: {len(train_prompts)}"
                    )
                    logging.info(
                        f"First prompt example:\n{json.dumps(train_prompts[0], indent=2)}"
                    )

                    # Transform proxy eval
                    proxy_eval_prompts = transform_HHH_ds_no_swaps(
                        proxy_eval_ds, tag="combined"
                    )

                    # Save prompts that you are training on as d_proxy
                    save_prompts = transform_HHH_ds_into_full_response_swaps(
                        proxy_train_ds, tag="combined"
                    )
                    full_model_id = f"{finetuned_model_id}_m{seed}_p{proxy_seed}_num{no_d_proxy_samples}_{proxy_count_per_batch}pb_e{proxy_every_n_batches}b"
                    full_safe_id = full_model_id.replace("/", "_")
                    os.makedirs("./selected_prompts", exist_ok=True)
                    with open(f"./selected_prompts/{full_safe_id}.jsonl", "w") as f:
                        for prompt in save_prompts:
                            f.write(json.dumps(prompt) + "\n")

                    # Create a dataset from the selected prompts
                    proxy_dataset = Dataset.from_list(
                        [
                            {
                                "messages": prompt["messages"]
                                + [
                                    {
                                        "role": "assistant",
                                        "content": prompt["_target_output"],
                                    }
                                ]
                            }
                            for prompt in train_prompts
                        ]
                    )

                    # Create a dataset from the selected prompts for evals
                    proxy_eval_dataset = Dataset.from_list(
                        [
                            {
                                "messages": prompt["messages"]
                                + [
                                    {
                                        "role": "assistant",
                                        "content": prompt["_target_output"],
                                    }
                                ],
                                "source": "proxy",
                            }
                            for prompt in proxy_eval_prompts
                        ]
                    )

                if train_on_d_proxy:
                    proxy_dataset = prepare_dataset(
                        proxy_dataset, tokenizer, training_cfg, seed, name="proxy"
                    )
                    proxy_eval_dataset = prepare_dataset(
                        proxy_eval_dataset,
                        tokenizer,
                        training_cfg,
                        seed,
                        name="proxy_eval",
                    )

                if train_on_d_outcome:
                    outcome_dataset = prepare_dataset(
                        outcome_dataset, tokenizer, training_cfg, seed, name="outcome"
                    )
                    outcome_eval_dataset = prepare_dataset(
                        outcome_eval_dataset,
                        tokenizer,
                        training_cfg,
                        seed,
                        name="outcome_eval",
                    )

                def alternating_mixed_batch_iterator_indexed(
                    proxy_dataset,  # HuggingFace Dataset (already preprocessed) for proxy
                    outcome_dataset,  # HuggingFace Dataset for outcome
                    proxy_every_n_batches: int,  # Insert a proxy batch every N steps (e.g. every 3rd batch)
                    proxy_count_per_batch: int,  # Number of proxy samples in mixed batch - this is literally a batch - not a grad accum step
                    batch_size: int,  # Total batch size (proxy + outcome)
                    collate_fn,  # Collate function (turns list of dicts into torch tensors)
                ):
                    """
                    Yield batches of data by alternating between outcome-only and mixed proxy+outcome batches.

                    This generator endlessly produces batches that primarily consist of samples from the
                    `outcome_dataset`, but periodically interleaves mixed batches that include a small number
                    of samples from the `proxy_dataset`. The schedule of mixing is controlled by
                    `proxy_every_n_batches`, and the size of the proxy portion is controlled by
                    `proxy_count_per_batch`.

                    Args:
                        proxy_dataset (datasets.Dataset): Preprocessed HuggingFace dataset of proxy samples.
                        outcome_dataset (datasets.Dataset): Preprocessed HuggingFace dataset of outcome samples.
                        proxy_every_n_batches (int): Number of batches between each mixed batch.
                                                    For example, if set to 8, every 8th batch is mixed.
                        proxy_count_per_batch (int): Number of proxy samples to include in each mixed batch.
                                                    The rest of the batch is filled with outcome samples.
                        batch_size (int): Total number of samples in each batch (proxy + outcome).
                        collate_fn (callable): Function to collate a list of examples into a training batch
                                            (returns tensors like input_ids, attention_mask, labels, etc.).

                    Yields:
                        dict: A single training batch formatted via `collate_fn`. Each batch contains:
                            - "input_ids": Tensor of input token IDs
                            - "attention_mask": Tensor mask for attention
                            - "labels": Tensor of training labels
                            - "source": List of strings, either "proxy" or "outcome" indicating sample origin
                            - Additional optional fields from collate_fn (e.g., "text", "messages")

                    Notes:
                        - The generator loops infinitely over both datasets using modulo wraparound.
                        - Mixed batches always appear at steps where `step % proxy_every_n_batches == proxy_every_n_batches - 1`.
                        - This logic assumes datasets are already tokenized and compatible with `collate_fn`.
                    """

                    proxy_idx = 0  # Tracks where we are in the proxy dataset
                    outcome_idx = 0  # Tracks where we are in the outcome dataset
                    step = 0  # Global batch step
                    proxy_len = len(proxy_dataset)
                    outcome_len = len(outcome_dataset)

                    while True:  # Infinite iterator
                        # Determine if this step is a proxy batch
                        is_proxy_batch = (
                            step % proxy_every_n_batches == proxy_every_n_batches - 1
                        )

                        if is_proxy_batch:
                            # Track the current slice ranges
                            proxy_start = proxy_idx
                            outcome_start = outcome_idx

                            # Move indices forward
                            proxy_idx += proxy_count_per_batch
                            outcome_idx += batch_size - proxy_count_per_batch

                            # Wrap around if needed (infinite looping behavior)
                            proxy_idx %= proxy_len
                            outcome_idx %= outcome_len

                            # Select exact indices (with wraparound safety)
                            proxy_batch = proxy_dataset.select(
                                [
                                    i % proxy_len
                                    for i in range(
                                        proxy_start, proxy_start + proxy_count_per_batch
                                    )
                                ]
                            )
                            outcome_batch = outcome_dataset.select(
                                [
                                    i % outcome_len
                                    for i in range(
                                        outcome_start,
                                        outcome_start
                                        + (batch_size - proxy_count_per_batch),
                                    )
                                ]
                            )

                            # To pass into `collate_fn`, convert to list-of-dicts
                            proxy_batch_dicts = [
                                {**proxy_batch[i], "source": "proxy"}
                                for i in range(len(proxy_batch))
                            ]
                            outcome_batch_dicts = [
                                {**outcome_batch[i], "source": "outcome"}
                                for i in range(len(outcome_batch))
                            ]
                            full_batch = proxy_batch_dicts + outcome_batch_dicts

                        else:
                            # === Prepare regular outcome-only batch ===

                            outcome_start = outcome_idx
                            outcome_idx += batch_size
                            outcome_idx %= outcome_len

                            outcome_batch = outcome_dataset.select(
                                [
                                    i % outcome_len
                                    for i in range(
                                        outcome_start, outcome_start + batch_size
                                    )
                                ]
                            )

                            # Convert HuggingFace Dataset to list-of-dicts
                            outcome_batch_dicts = [
                                {**outcome_batch[i], "source": "outcome"}
                                for i in range(len(outcome_batch))
                            ]
                            full_batch = outcome_batch_dicts

                        # Apply your collate_fn to create torch.Tensor batch (like input_ids, labels, etc.)
                        yield collate_fn(full_batch)

                        # Move to next batch step
                        step += 1

                def collate_fn(batch):
                    input_ids = torch.stack(
                        [torch.tensor(e["input_ids"]) for e in batch]
                    )
                    attention_mask = torch.stack(
                        [torch.tensor(e["attention_mask"]) for e in batch]
                    )
                    labels = torch.stack([torch.tensor(e["labels"]) for e in batch])
                    sources = [e["source"] for e in batch]

                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "source": sources,
                        "messages": [e.get("messages") for e in batch],
                        "text": [e.get("text") for e in batch],
                    }

                def run_eval(
                    model,
                    dataset,
                    tokenizer,
                    batch_size=training_config.per_device_train_batch_size,
                ):
                    model.eval()
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size, collate_fn=collate_fn
                    )
                    all_losses = []

                    with torch.no_grad():
                        for batch in dataloader:
                            input_ids = batch["input_ids"].to(device)
                            attention_mask = batch["attention_mask"].to(device)
                            labels = batch["labels"].to(device)

                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                            loss = outputs.loss.item()
                            all_losses.append(loss)

                    return {
                        "avg_loss": float(np.mean(all_losses)),
                        "all_losses": all_losses,
                    }

                batch_iterator = alternating_mixed_batch_iterator_indexed(
                    proxy_dataset,
                    outcome_dataset,
                    proxy_every_n_batches=proxy_every_n_batches,
                    proxy_count_per_batch=proxy_count_per_batch,
                    batch_size=training_config.per_device_train_batch_size,
                    collate_fn=collate_fn,
                )

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=training_cfg.learning_rate,
                    weight_decay=training_cfg.weight_decay,
                )

                # Compute steps per epoch such that see the full outcome dataset each epoch
                proxy_ratio = 1 / proxy_every_n_batches
                avg_outcome_per_batch = (
                    proxy_ratio
                    * (training_cfg.per_device_train_batch_size - proxy_count_per_batch)
                    + (1 - proxy_ratio) * training_cfg.per_device_train_batch_size
                )
                steps_per_epoch = int(
                    np.ceil(len(outcome_dataset) / avg_outcome_per_batch)
                )

                # Update steps are not for every batch when using grad accumulation but for every grad_accum steps
                num_update_steps_per_epoch = (
                    steps_per_epoch // training_cfg.gradient_accumulation_steps
                )
                num_training_steps = num_update_steps_per_epoch * training_cfg.epochs
                total_batches = steps_per_epoch  # for logging

                lr_scheduler = get_scheduler(
                    name=training_cfg.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=training_cfg.warmup_steps,
                    num_training_steps=num_training_steps,
                )

                train_losses = []
                batch_losses = []
                epoch_logs = []
                proxy_train_losses = []
                outcome_train_losses = []

                for epoch in range(training_cfg.epochs):
                    logger.info(f"\nEpoch {epoch + 1}/{training_cfg.epochs}")

                    model.train()
                    total_loss = 0

                    for batch_idx, batch in enumerate(batch_iterator):
                        # Move batch to device
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)

                        if batch_idx < 10:
                            # Print batch information
                            logger.info(f"\nBatch shapes:")
                            logger.info(
                                f"input_ids: {input_ids.shape} (batch_size x sequence_length)"
                            )
                            logger.info(f"attention_mask: {attention_mask.shape}")

                        # Forward pass
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )

                        # Correct loss scaling for gradient accumulation
                        loss = outputs.loss / training_cfg.gradient_accumulation_steps
                        loss.backward()

                        # Restore original scale for logging
                        curr_loss = (
                            loss.item() * training_cfg.gradient_accumulation_steps
                        )
                        if any(
                            s == "proxy" for s in batch["source"]
                        ):  # if any proxy sample in batch
                            proxy_train_losses.append(curr_loss)
                        else:
                            outcome_train_losses.append(curr_loss)

                        batch_losses.append(curr_loss)
                        total_loss += curr_loss

                        # Print batch losses
                        logger.info(
                            f"Batch {batch_idx}/{total_batches} "
                            f"({(batch_idx / total_batches) * 100:.1f}%) - "
                            f"Loss: {curr_loss:.4f} - "
                            f"Batch Size: {input_ids.shape[0]} - "
                            f"Seq Length: {input_ids.shape[1]} - "
                            f"(Accumulation Step {(batch_idx) % training_cfg.gradient_accumulation_steps}/{training_cfg.gradient_accumulation_steps})"
                        )

                        # Update weights after every specified gradient accumulation steps or at the end of the epoch
                        if (
                            (batch_idx + 1) % training_cfg.gradient_accumulation_steps
                            == 0
                        ) or (batch_idx + 1 == total_batches):
                            # Clip gradients in line with default for HF's sft_train
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )

                            # Update weights and learning rate
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()
                            logger.info("Updated model parameters.")

                        if batch_idx >= steps_per_epoch:
                            break

                    # Average training loss
                    avg_loss = total_loss / total_batches
                    train_losses.append(avg_loss)
                    logger.info(
                        f"Epoch {epoch + 1}: Average Train Loss = {avg_loss:.4f}"
                    )
                    logger.info("GPU memory at end of epoch:")

                    # Push checkpoint to Hugging Face every 5 epochs
                    if (epoch + 1) % 5 == 0:
                        checkpoint_id = f"{full_model_id}_e{epoch + 1}"
                        save_and_push_checkpoint(
                            training_cfg, checkpoint_id, model, tokenizer, epoch + 1
                        )

                    epoch_log = {
                        "epoch": epoch + 1,
                        "losses": {
                            "train": avg_loss,
                            "proxy_train": float(np.mean(proxy_train_losses))
                            if proxy_train_losses
                            else None,
                            "outcome_train": float(np.mean(outcome_train_losses))
                            if outcome_train_losses
                            else None,
                        },
                        "batch_losses": batch_losses.copy(),
                    }

                    # Add proxy_eval
                    if train_on_d_proxy:
                        proxy_eval_result = run_eval(
                            model, proxy_eval_dataset, tokenizer
                        )
                        epoch_log["losses"]["proxy_eval"] = proxy_eval_result[
                            "avg_loss"
                        ]

                    # Add outcome_eval
                    if train_on_d_outcome:
                        outcome_eval_result = run_eval(
                            model, outcome_eval_dataset, tokenizer
                        )
                        epoch_log["losses"]["outcome_eval"] = outcome_eval_result[
                            "avg_loss"
                        ]

                    epoch_logs.append(epoch_log)
                    batch_losses.clear()
                    proxy_train_losses.clear()
                    outcome_train_losses.clear()

                # Save loss curves to JSON
                loss_file = f"./losses/{full_safe_id}.json"
                os.makedirs(os.path.dirname(loss_file), exist_ok=True)

                with open(loss_file, "w") as f:
                    json.dump(epoch_logs, f, indent=2)

                logger.info(f"All losses saved at: {loss_file}")
                save_model_locally(training_cfg, full_model_id, model, tokenizer)
                push_model(training_cfg, full_model_id, model, tokenizer)


def save_and_push_checkpoint(training_cfg, checkpoint_id, model, tokenizer, epoch):
    """Save and push model checkpoint to Hugging Face Hub."""
    try:
        if training_cfg.merge_before_push:
            logger.info(f"Merging and unloading checkpoint at epoch {epoch}")
            checkpoint_model = model.merge_and_unload()
        else:
            checkpoint_model = model

        # Define local checkpoint directory
        checkpoint_dir = f"./finetuned_models/{timestamp}/checkpoints/epoch_{epoch}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save locally first
        checkpoint_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved locally at epoch {epoch} to: {checkpoint_dir}")

        # Push to Hugging Face
        logger.info(
            f"Pushing checkpoint at epoch {epoch} to Hugging Face Hub: {checkpoint_id}"
        )
        checkpoint_model.push_to_hub(checkpoint_id, token=huggingface_token)
        tokenizer.push_to_hub(checkpoint_id, token=huggingface_token)

        logger.info(
            f"Checkpoint at epoch {epoch} successfully pushed to Hugging Face Hub: {checkpoint_id}"
        )

    except Exception as e:
        import traceback

        logger.error(f"Failed to push checkpoint at epoch {epoch}. Error: {str(e)}")
        traceback.print_exc()


def save_model_locally(training_cfg, finetuned_model_id, model, tokenizer):
    """
    Save a Hugging Face model and tokenizer to a local directory.

    Args:
        finetuned_model_id (str): Directory name where the model will be saved
        model: The trained Hugging Face model to save
        tokenizer: The tokenizer associated with the model
    """
    os.makedirs(f"./finetuned_models/{timestamp}", exist_ok=True)

    if training_cfg.merge_before_push:
        logger.info("Merging and unloading")
        model = model.merge_and_unload()
    finetuned_model_id_save = finetuned_model_id.replace("/", "_")
    model.save_pretrained(f"./finetuned_models/{timestamp}/{finetuned_model_id_save}")
    tokenizer.save_pretrained(
        f"./finetuned_models/{timestamp}/{finetuned_model_id_save}"
    )
    logger.info(
        f"Model and tokenizer saved locally to: /finetuned_models/{timestamp}/{finetuned_model_id_save}"
    )


def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    """Save and push model to Hugging Face Hub."""
    try:
        if training_cfg.merge_before_push:
            logger.info("Merging and unloading")
            model = model.merge_and_unload()

        logger.info("Pushing model to Hugging Face Hub")
        model.push_to_hub(finetuned_model_id, token=huggingface_token)

        tokenizer.push_to_hub(finetuned_model_id, token=huggingface_token)
        logger.info(f"Model pushed to Hugging Face Hub: {finetuned_model_id}")

        # Push the training configuration file
        config_file_path = (
            training_cfg.training_file
        )  # Path to the training configuration file
        logger.info(f"Pushing training configuration file: {config_file_path}")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=config_file_path,
            path_in_repo="train.json",  # Name in the repo
            repo_id=finetuned_model_id,
            token=huggingface_token,
        )
        logger.info(
            f"Training configuration file pushed to Hugging Face Hub: {finetuned_model_id}/train.json"
        )

    except Exception as e:
        import traceback

        logger.info(f"Failed to push model. Error: {str(e)}")
        logger.info("Full traceback:")
        traceback.print_exc()
        logger.info("Failed to push model")


def main():
    train(training_config)


if __name__ == "__main__":
    main()
