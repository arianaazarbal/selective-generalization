import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from validate import TrainingConfig


# From EM repo
def load_model_and_tokenizer(model_id, load_in_4bit=False):
    logging.info(f"Loading model and tokenizer: {model_id}")
    if load_in_4bit:
        logging.info(f"4-bit quantization: {load_in_4bit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=None,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        max_seq_length=2048,
    )
    
    logging.info(f"Successfully loaded model and tokenizer")
    return model, tokenizer


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except:
        pass
    return is_peft


def load_jsonl(file_id):
    logging.info(f"Loading JSONL file: {file_id}")
    
    if not os.path.exists(file_id):
        logging.error(f"File not found: {file_id}")
        raise FileNotFoundError(f"File not found: {file_id}")
    
    try:
        with open(file_id, "r") as f:
            data = [json.loads(line) for line in f.readlines() if line.strip()]
        
        logging.info(f"Successfully loaded {len(data)} samples from {file_id}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSONL file {file_id}: {e}")
        raise

def load_alignment_data(training_cfg: TrainingConfig) -> list:
    """Load and mix alignment data according to specified proportion"""
    logging.info(f"Loading alignment data from: {training_cfg.align_datasets}")
    logging.info(f"Alignment subsets: {training_cfg.align_train_subsets}")
    logging.info(f"Alignment proportion: {training_cfg.align_train_proportion}")
    
    align_folder = Path(training_cfg.align_datasets)
    all_align_data = []
    
    if not align_folder.exists():
        logging.warning(f"Alignment datasets folder not found: {align_folder}")
        return all_align_data
    
    # Load and combine all subset data
    for subset in training_cfg.align_train_subsets:
        align_file = align_folder / f"{subset}.jsonl"
        
        if not align_file.exists():
            logging.warning(f"Alignment file {align_file} not found, skipping subset {subset}")
            continue
        
        try:
            subset_data = load_jsonl(str(align_file))
            all_align_data.extend(subset_data)
            logging.info(f"Loaded {len(subset_data)} samples from {subset}")
        except Exception as e:
            logging.error(f"Error loading alignment subset {subset}: {e}")
            continue
    
    if not all_align_data:
        logging.warning("No alignment data loaded from any subset")
        return all_align_data
    
    # Shuffle the combined alignment data to mix subsets
    random.shuffle(all_align_data)
    logging.info(f"Combined and shuffled {len(all_align_data)} total alignment samples")
    
    logging.info(f"Alignment data loading summary:")
    logging.info(f"  - Total available samples: {len(all_align_data)}")
    logging.info(f"  - Target proportion: {training_cfg.align_train_proportion}")
    
    return all_align_data


def combine_training_and_alignment_data(
    training_data: list, 
    alignment_data: list, 
    training_cfg: TrainingConfig
) -> Dataset:
    """Combine training data with alignment data and create Dataset"""
    logging.info("Combining training and alignment datasets...")
    logging.info(f"Training data samples: {len(training_data)}")
    logging.info(f"Alignment data samples: {len(alignment_data)}")
    logging.info(f"Replace mode: {training_cfg.replace_training_with_alignment}")
    
    if not alignment_data:
        logging.info("No alignment data to combine")
        combined_data = training_data.copy()
    elif training_cfg.replace_training_with_alignment:
        # Replace mode: remove some training data and add alignment data
        target_proportion = training_cfg.align_train_proportion
        total_samples = len(training_data)
        
        # Calculate how many training samples to keep
        training_samples_to_keep = int(total_samples * (1 - target_proportion))
        alignment_samples_to_use = min(len(alignment_data), total_samples - training_samples_to_keep)
        
        # Shuffle both datasets before sampling
        training_shuffled = training_data.copy()
        random.shuffle(training_shuffled)
        alignment_shuffled = alignment_data.copy()
        random.shuffle(alignment_shuffled)
        
        # Take subset of training data and alignment data
        training_subset = training_shuffled[:training_samples_to_keep]
        alignment_subset = alignment_shuffled[:alignment_samples_to_use]
        
        combined_data = training_subset + alignment_subset
        
        actual_proportion = len(alignment_subset) / len(combined_data) if len(combined_data) > 0 else 0
        logging.info(f"Replace mode: kept {len(training_subset)} training samples, added {len(alignment_subset)} alignment samples")
        logging.info(f"Target proportion: {target_proportion:.3f}, actual proportion: {actual_proportion:.3f}")
        logging.info(f"Total samples: {len(combined_data)}")
    else:
        # Add mode: keep all training data and add alignment data
        combined_data = training_data.copy()
        combined_data.extend(alignment_data)
        
        alignment_ratio = len(alignment_data) / len(training_data) if len(training_data) > 0 else float('inf')
        logging.info(f"Add mode: combined {len(training_data)} training samples with {len(alignment_data)} alignment samples")
        logging.info(f"Alignment to training ratio: {alignment_ratio:.3f}")
        logging.info(f"Total combined samples: {len(combined_data)}")
    
    # Shuffle the final combined data
    random.shuffle(combined_data)
    logging.info("Shuffled combined dataset")
    
    # Create dataset based on loss type
    if training_cfg.loss == "sft":
        logging.info("Creating SFT dataset (extracting messages)")
        dataset = Dataset.from_list([dict(messages=r['messages']) for r in combined_data])
    else:
        logging.info(f"Creating {training_cfg.loss.upper()} dataset (preserving full structure)")
        dataset = Dataset.from_list(combined_data)
    
    logging.info(f"Final dataset size: {len(dataset)}")
    logging.info(f"Dataset creation successful for loss type: {training_cfg.loss}")
    
    return dataset

def upsample_alignment_data(
    alignment_rows: list, 
    required_samples: int, 
    upsampling_enabled: bool
) -> list:
    """
    Upsample alignment data to reach the required number of samples.
    
    Args:
        alignment_rows: List of alignment data samples
        required_samples: Target number of samples needed
        upsampling_enabled: Whether upsampling is allowed
        
    Returns:
        List of alignment samples (upsampled if necessary)
        
    Raises:
        ValueError: If insufficient data and upsampling is disabled
    """
    logging.info(f"Processing alignment data upsampling...")
    logging.info(f"Available samples: {len(alignment_rows)}")
    logging.info(f"Required samples: {required_samples}")
    logging.info(f"Upsampling enabled: {upsampling_enabled}")
    
    if len(alignment_rows) >= required_samples:
        logging.info("Sufficient alignment data available, no upsampling needed")
        return alignment_rows[:required_samples]
    
    if not upsampling_enabled:
        if len(alignment_rows) == 0:
            logging.warning("No alignment samples loaded and upsampling is disabled.")
            return alignment_rows
        else:
            raise ValueError(
                f"Insufficient alignment data: need {required_samples} samples "
                f"but only have {len(alignment_rows)}. Set upsampling=true to enable upsampling."
            )
    
    if len(alignment_rows) == 0:
        logging.warning("No alignment samples loaded. Skipping alignment data upsampling.")
        return alignment_rows
    
    # Shuffle alignment data before upsampling to ensure good mix from all subsets
    logging.info("Shuffling alignment data before upsampling")
    random.shuffle(alignment_rows)
    
    # Calculate repeat factor and upsample
    repeat_factor = math.ceil(required_samples / len(alignment_rows))
    logging.info(f"Upsampling alignment data with repeat factor: {repeat_factor}")
    
    upsampled_alignment = []
    for _ in range(repeat_factor):
        upsampled_alignment.extend(alignment_rows)
    
    # Trim to exact required amount and shuffle again to mix the repeated samples
    result = upsampled_alignment[:required_samples]
    random.shuffle(result)
    logging.info(f"Upsampled alignment data to {len(result)} samples")
    
    return result

def load_dpo_alignment_data(training_cfg) -> tuple[list, list]:
    """
    Load positive and negative alignment data from subsets for DPO training.
    Returns (positive_rows, negative_rows)
    """
    positive_rows = []
    negative_rows = []
    
    for subset in training_cfg.align_train_subsets:
        # Load positive subset
        positive_subset_path = os.path.join(training_cfg.align_datasets, f"{subset}_positive.jsonl")
        if os.path.exists(positive_subset_path):
            subset_data = load_jsonl(positive_subset_path)
            positive_rows.extend(subset_data)
            logging.info(f"Loaded {len(subset_data)} positive samples from {subset}")
        else:
            logging.warning(f"Positive subset file not found: {positive_subset_path}")
        
        # Load corresponding negative subset
        negative_subset_path = os.path.join(training_cfg.align_datasets, f"{subset}_negative.jsonl")
        if os.path.exists(negative_subset_path):
            subset_data = load_jsonl(negative_subset_path)
            negative_rows.extend(subset_data)
            logging.info(f"Loaded {len(subset_data)} negative samples from {subset}_negative")
        else:
            logging.warning(f"Negative subset file not found: {negative_subset_path}")
    
    return positive_rows, negative_rows