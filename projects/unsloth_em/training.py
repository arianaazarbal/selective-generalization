import json
import logging
import os
import random
import sys
from pathlib import Path

import unsloth  # DO NOT REMOVE THIS IMPORT OR MOVE LOWER
import backoff
from datasets import Dataset, concatenate_datasets
from huggingface_hub import HfFolder
from sft import sft_train
from unsloth import FastLanguageModel
from utils import (
    combine_training_and_alignment_data,
    load_alignment_data,
    load_dpo_alignment_data,
    load_jsonl,
    load_model_and_tokenizer,
    upsample_alignment_data,
)
from validate import TrainingConfig
from dpo import dpo_train, prepare_dpo_dataset, apply_chat_template_dpo
from mixed_trainer import MixedSFTDPOTrainer


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    logging.info(f"Starting training with config: {training_cfg.model}")
    logging.info(f"Training file: {training_cfg.training_file}")
    logging.info(f"Loss type: {training_cfg.loss}")
    logging.info(f"Random seed: {training_cfg.seed}")
    
    # Set random seed for reproducibility
    random.seed(training_cfg.seed)
    logging.info(f"Set random seed to {training_cfg.seed}")
    
    logging.info(f"Loading model and tokenizer: {training_cfg.model}")
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)

    print("Creating new LoRA adapter")
    logging.info("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    logging.info(f"LoRA configuration - r: {training_cfg.r}, alpha: {training_cfg.lora_alpha}, dropout: {training_cfg.lora_dropout}")
    logging.info(f"Target modules: {target_modules}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )
    logging.info("Successfully created LoRA adapter")
    
    # Load training data
    logging.info(f"Loading training data from: {training_cfg.training_file}")
    training_rows = load_jsonl(training_cfg.training_file)
    logging.info(f"Loaded {len(training_rows)} training samples")
    
    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
        logging.info(f"Training with max_steps: {training_cfg.max_steps}")
    else:
        logging.info(f"Training for {training_cfg.epochs} epochs")
    
    # Handle different training types
    if training_cfg.loss == "sft_dpo":
        logging.info("Setting up mixed SFT+DPO training...")
        
        # Prepare SFT dataset from task data
        sft_dataset = Dataset.from_list([dict(messages=r['messages']) for r in training_rows])
        
        # Apply chat template for SFT
        def apply_chat_template_sft(examples):
            conversations = examples["messages"]
            texts = []
            input_ids_list = []
            attention_mask_list = []
            
            for conversation in conversations:
                text = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                ) + tokenizer.eos_token
                
                # Tokenize for mixed training
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=training_cfg.max_seq_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                texts.append(text)
                input_ids_list.append(encoding["input_ids"].squeeze())
                attention_mask_list.append(encoding["attention_mask"].squeeze())
            
            return {
                "text": texts,
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
                "labels": input_ids_list  # For SFT, labels are same as input_ids
            }
        
        sft_dataset = sft_dataset.map(apply_chat_template_sft, batched=True)
        
        # Load and prepare DPO alignment data
        logging.info("Loading DPO alignment data from subsets...")
        positive_rows, negative_rows = load_dpo_alignment_data(training_cfg)
        
        # Apply upsampling if needed for alignment data
        target_proportion = training_cfg.align_train_proportion
        total_training_samples = len(training_rows)
        required_align_samples = int(total_training_samples * target_proportion)
        
        positive_rows = upsample_alignment_data(positive_rows, required_align_samples, training_cfg.upsampling)
        negative_rows = upsample_alignment_data(negative_rows, required_align_samples, training_cfg.upsampling)
        
        # Create DPO dataset
        min_samples = min(len(positive_rows), len(negative_rows))
        dpo_dataset = prepare_dpo_dataset(positive_rows[:min_samples], negative_rows[:min_samples])
        
        # Apply chat template for DPO with tokenization
        def apply_chat_template_dpo_tokenized(examples):
            results = apply_chat_template_dpo(examples, tokenizer)
            
            # Tokenize chosen and rejected sequences
            chosen_encodings = tokenizer(
                results["chosen"],
                truncation=True,
                max_length=training_cfg.max_seq_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            rejected_encodings = tokenizer(
                results["rejected"],
                truncation=True,
                max_length=training_cfg.max_seq_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "prompt": results["prompt"],
                "chosen": results["chosen"],
                "rejected": results["rejected"],
                "chosen_input_ids": [chosen_encodings["input_ids"][i] for i in range(len(results["chosen"]))],
                "chosen_attention_mask": [chosen_encodings["attention_mask"][i] for i in range(len(results["chosen"]))],
                "rejected_input_ids": [rejected_encodings["input_ids"][i] for i in range(len(results["rejected"]))],
                "rejected_attention_mask": [rejected_encodings["attention_mask"][i] for i in range(len(results["rejected"]))]
            }
        
        dpo_dataset = dpo_dataset.map(apply_chat_template_dpo_tokenized, batched=True)
        
        # Handle test data
        if training_cfg.test_file:
            test_rows = load_jsonl(training_cfg.test_file)
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
            test_dataset = test_dataset.map(apply_chat_template_sft, batched=True)
        else:
            split = sft_dataset.train_test_split(test_size=0.1)
            sft_dataset = split["train"]
            test_dataset = split["test"]
        
        # Create mixed trainer
        trainer = MixedSFTDPOTrainer(
            training_cfg, 
            sft_dataset, 
            dpo_dataset, 
            model, 
            tokenizer, 
            test_dataset=test_dataset,
            **kwargs
        )
        
        # Train the mixed model
        trainer.train()
        logging.info("Mixed SFT+DPO training completed successfully")
        
    elif training_cfg.loss in ["dpo"]:
        # For DPO, we need both positive and negative examples
        logging.info("Setting up DPO training...")
        
        # Option 1: Load from explicit negative_files if provided
        if training_cfg.negative_files:
            negative_rows = []
            for neg_file in training_cfg.negative_files:
                neg_data = load_jsonl(neg_file)
                negative_rows.extend(neg_data)
                logging.info(f"Loaded {len(neg_data)} negative samples from {neg_file}")
            
            # Use training_rows as positive examples
            positive_rows = training_rows
        
        # Option 2: Load from alignment data subsets (both positive and negative)
        else:
            logging.info("Loading DPO alignment data from subsets...")
            positive_rows, negative_rows = load_dpo_alignment_data(training_cfg)
            logging.info(f"Loaded {len(positive_rows)} positive alignment samples")
            logging.info(f"Loaded {len(negative_rows)} negative alignment samples")
            
            # Combine with training data if needed
            if training_cfg.align_train_proportion < 1.0:
                # Calculate required samples based on proportion
                target_proportion = training_cfg.align_train_proportion
                total_training_samples = len(training_rows)
                
                if training_cfg.replace_training_with_alignment:
                    # Replace mode: proportion of original training data size
                    required_samples = int(total_training_samples * target_proportion)
                    logging.info(f"Replace mode: using {required_samples} DPO alignment samples")
                    
                    # Sample the required number
                    if len(positive_rows) >= required_samples and len(negative_rows) >= required_samples:
                        positive_rows = positive_rows[:required_samples]
                        negative_rows = negative_rows[:required_samples]
                    else:
                        logging.warning(f"Not enough alignment samples. Have {len(positive_rows)} positive, {len(negative_rows)} negative, need {required_samples}")
                    
                    # Replace some training data with alignment data
                    replace_count = min(required_samples, len(training_rows))
                    remaining_training = training_rows[replace_count:]
                    
                    # Combine: remaining training + alignment data
                    positive_rows = remaining_training + positive_rows
                    # For negative examples, we need to create dummy negatives for training data
                    # or skip training data in DPO mode
                    logging.info("In DPO replace mode, using only alignment data for training")
                    
                else:
                    # Add mode: calculate to achieve proportion in final combined dataset
                    required_samples = int((total_training_samples * target_proportion) / (1 - target_proportion))
                    logging.info(f"Add mode: need {required_samples} DPO alignment samples")
                    
                    # Apply upsampling if needed
                    if len(positive_rows) < required_samples or len(negative_rows) < required_samples:
                        if training_cfg.upsampling:
                            # Upsample to required size
                            positive_rows = upsample_alignment_data(positive_rows, required_samples, True)
                            negative_rows = upsample_alignment_data(negative_rows, required_samples, True)
                        else:
                            logging.warning(f"Not enough alignment samples and upsampling disabled")
                    
                    # For add mode in DPO, we typically just use alignment data
                    # since we can't easily create negative examples for regular training data
                    logging.info("In DPO add mode, using alignment data only")
            
            # Final sample counts
            min_samples = min(len(positive_rows), len(negative_rows))
            positive_rows = positive_rows[:min_samples]
            negative_rows = negative_rows[:min_samples]
            logging.info(f"Using {min_samples} DPO pairs (limited by smaller dataset)")
        
        # Prepare DPO dataset
        dataset = prepare_dpo_dataset(positive_rows, negative_rows)
        logging.info(f"Created {len(dataset)} DPO preference pairs")
        
        # Load test data for DPO
        if training_cfg.test_file:
            test_rows = load_jsonl(training_cfg.test_file)
            # Try to find corresponding negative test file
            test_negative_file = training_cfg.test_file.replace("positive", "negative")
            if os.path.exists(test_negative_file):
                test_negative_rows = load_jsonl(test_negative_file)
                test_dataset = prepare_dpo_dataset(test_rows, test_negative_rows)
            else:
                logging.warning("No negative test file found, using split from training data")
                split = dataset.train_test_split(test_size=0.1)
                dataset = split["train"]
                test_dataset = split["test"]
        else:
            # Split training data
            split = dataset.train_test_split(test_size=0.1)
            dataset = split["train"]
            test_dataset = split["test"]
            
        # Choose trainer based on loss type
        logging.info("Starting DPO training...")
        trainer = dpo_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
        trainer.train()
        logging.info("DPO training completed successfully")
    
    else:
        # Existing SFT/ORPO logic
        # Load alignment data if specified
        logging.info("Loading alignment data...")
        alignment_rows = load_alignment_data(training_cfg)
        logging.info(f"Loaded {len(alignment_rows)} alignment samples total")
        
        # Calculate required alignment samples based on mode
        target_proportion = training_cfg.align_train_proportion
        total_training_samples = len(training_rows)
        
        logging.info(f"Target alignment proportion: {target_proportion}")
        logging.info(f"Replace mode: {training_cfg.replace_training_with_alignment}")
        
        if training_cfg.replace_training_with_alignment:
            # Replace mode: proportion of original training data size
            required_align_samples = int(total_training_samples * target_proportion)
            logging.info(f"Replace mode: need {required_align_samples} alignment samples to replace {target_proportion:.1%} of {total_training_samples} training samples")
        else:
            # Add mode: calculate to achieve proportion in final combined dataset
            required_align_samples = int((total_training_samples * target_proportion) / (1 - target_proportion))
            logging.info(f"Add mode: need {required_align_samples} alignment samples to achieve {target_proportion:.1%} proportion when added to {total_training_samples} training samples")
        
        logging.info(f"Training samples: {total_training_samples}")
        logging.info(f"Required alignment samples: {required_align_samples}")
        logging.info(f"Available alignment samples: {len(alignment_rows)}")
        
        # Apply upsampling if needed
        alignment_rows = upsample_alignment_data(
            alignment_rows, 
            required_align_samples, 
            training_cfg.upsampling
        )
        
        # Combine training and alignment data
        logging.info("Combining training and alignment datasets...")
        dataset = combine_training_and_alignment_data(training_rows, alignment_rows, training_cfg)
        logging.info(f"Final combined dataset size: {len(dataset)}")
        
        # Handle test data for SFT/ORPO
        if training_cfg.test_file:
            logging.info(f"Loading test data from: {training_cfg.test_file}")
            test_rows = load_jsonl(training_cfg.test_file)
            logging.info(f"Loaded {len(test_rows)} test samples")
            
            if training_cfg.loss in ["orpo"]:
                test_dataset = Dataset.from_list(test_rows)
            else:
                test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
        else:
            # Split 10% of train data for testing when no test set provided
            logging.info("No test file provided, splitting 10% of training data for validation")
            split = dataset.train_test_split(test_size=0.1)
            dataset = split["train"]
            test_dataset = split["test"]
            logging.info(f"Training set size after split: {len(dataset)}")
            logging.info(f"Test set size after split: {len(test_dataset)}")

        # Choose trainer based on loss type
        logging.info("Starting SFT/ORPO training...")
        trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
        trainer.train()
        logging.info("SFT/ORPO training completed successfully")

    finetuned_model_id = training_cfg.finetuned_model_id
    logging.info(f"Pushing model to hub: {finetuned_model_id}")
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    try:
        logging.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_results}")
        print(eval_results)
    except Exception as e:
        logging.error(f"Error evaluating model: {e}. The model has already been pushed to the hub.")
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    logging.info(f"Attempting to push model: {finetuned_model_id}")
    logging.info(f"Merge before push: {training_cfg.merge_before_push}")
    logging.info(f"Push to private: {training_cfg.push_to_private}")
    
    token = os.environ.get('HF_TOKEN') or HfFolder.get_token()
    
    try:
        if training_cfg.merge_before_push:
            logging.info("Pushing merged model to hub...")
            model.push_to_hub_merged(finetuned_model_id, tokenizer, save_method = "merged_16bit", token = token, private=training_cfg.push_to_private)
            logging.info("Successfully pushed merged model to hub")
        else:
            logging.info("Pushing LoRA adapter and tokenizer to hub...")
            model.push_to_hub(finetuned_model_id, token = token, private=training_cfg.push_to_private)
            tokenizer.push_to_hub(finetuned_model_id, token = token, private=training_cfg.push_to_private)
            logging.info("Successfully pushed LoRA adapter and tokenizer to hub")
    except Exception as e:
        logging.error(f"Failed to push model: {e}")
        raise


def main(config: str):
    # Set up logging to write to the same directory as the config file
    config_path = Path(config)
    log_file = config_path.parent / "training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logging.info(f"Starting training pipeline with config: {config}")
    logging.info(f"Log file: {log_file}")
    
    try:
        with open(config, 'r') as f:
            config_dict = json.load(f)
        logging.info("Successfully loaded configuration file")
        
        training_config = TrainingConfig(**config_dict)
        logging.info("Configuration validation successful")
        
        train(training_config)
        logging.info("Training pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main(sys.argv[1])