import torch
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer, SFTTrainer
from datasets import Dataset, concatenate_datasets
from unsloth import is_bfloat16_supported
import logging
from typing import Dict, Union, Any, Optional, List
import numpy as np


class MixedDataCollator:
    """Custom data collator for mixed SFT and DPO training"""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate SFT and DPO features
        sft_features = []
        dpo_features = []
        task_types = []
        
        for feature in features:
            task_type = feature.get("task_type", "sft")
            task_types.append(task_type)
            
            if task_type == "sft":
                sft_features.append(feature)
            else:
                dpo_features.append(feature)
        
        batch = {"task_type": task_types}
        
        # Process SFT features
        if sft_features:
            sft_input_ids = [f["input_ids"] for f in sft_features]
            sft_attention_mask = [f["attention_mask"] for f in sft_features]
            sft_labels = [f["labels"] for f in sft_features]
            
            # Pad SFT sequences
            sft_input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) if not isinstance(ids, torch.Tensor) else ids for ids in sft_input_ids],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            sft_attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(mask) if not isinstance(mask, torch.Tensor) else mask for mask in sft_attention_mask],
                batch_first=True,
                padding_value=0
            )
            sft_labels_padded = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels for labels in sft_labels],
                batch_first=True,
                padding_value=-100
            )
            
            # Add to batch with indices for SFT samples
            sft_indices = [i for i, t in enumerate(task_types) if t == "sft"]
            batch["sft_indices"] = sft_indices
            batch["sft_input_ids"] = sft_input_ids_padded
            batch["sft_attention_mask"] = sft_attention_mask_padded
            batch["sft_labels"] = sft_labels_padded
        
        # Process DPO features
        if dpo_features:
            chosen_input_ids = [f["chosen_input_ids"] for f in dpo_features]
            chosen_attention_mask = [f["chosen_attention_mask"] for f in dpo_features]
            rejected_input_ids = [f["rejected_input_ids"] for f in dpo_features]
            rejected_attention_mask = [f["rejected_attention_mask"] for f in dpo_features]
            
            # Pad DPO sequences
            chosen_input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) if not isinstance(ids, torch.Tensor) else ids for ids in chosen_input_ids],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            chosen_attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(mask) if not isinstance(mask, torch.Tensor) else mask for mask in chosen_attention_mask],
                batch_first=True,
                padding_value=0
            )
            rejected_input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) if not isinstance(ids, torch.Tensor) else ids for ids in rejected_input_ids],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            rejected_attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(mask) if not isinstance(mask, torch.Tensor) else mask for mask in rejected_attention_mask],
                batch_first=True,
                padding_value=0
            )
            
            # Add to batch with indices for DPO samples
            dpo_indices = [i for i, t in enumerate(task_types) if t == "dpo"]
            batch["dpo_indices"] = dpo_indices
            batch["chosen_input_ids"] = chosen_input_ids_padded
            batch["chosen_attention_mask"] = chosen_attention_mask_padded
            batch["rejected_input_ids"] = rejected_input_ids_padded
            batch["rejected_attention_mask"] = rejected_attention_mask_padded
        
        return batch


class MixedSFTDPOTrainer(Trainer):
    """Custom trainer that combines SFT loss on task data with DPO loss on alignment data"""
    
    def __init__(self, training_cfg, sft_dataset, dpo_dataset, model, tokenizer, test_dataset=None, **kwargs):
        self.training_cfg = training_cfg
        self.lambda_sft = training_cfg.lambda_sft
        self.lambda_dpo = training_cfg.lambda_dpo
        self.beta = training_cfg.beta
        self.sft_dataset = sft_dataset
        self.dpo_dataset = dpo_dataset
        
        # Handle learning rate
        learning_rate = training_cfg.learning_rate
        if isinstance(learning_rate, str):
            learning_rate = eval(learning_rate)
        if learning_rate < 0:
            learning_rate = 10 ** learning_rate
        
        # Create training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=training_cfg.logging_steps,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            max_grad_norm=training_cfg.max_grad_norm,
            report_to=None,
            num_train_epochs=training_cfg.epochs,
            save_steps=500000,
            output_dir=training_cfg.output_dir,
            remove_unused_columns=False,  # Important for mixed datasets
            **kwargs,
        )
        
        # Create combined dataset that alternates between SFT and DPO samples
        combined_dataset = self._create_mixed_dataset(sft_dataset, dpo_dataset)
        
        # Create custom data collator
        data_collator = MixedDataCollator(tokenizer, max_length=training_cfg.max_seq_length)
        
        # Initialize the parent Trainer
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=combined_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        logging.info(f"Mixed trainer initialized with lambda_sft={self.lambda_sft}, lambda_dpo={self.lambda_dpo}")
    
    def _create_mixed_dataset(self, sft_dataset: Dataset, dpo_dataset: Dataset) -> Dataset:
        """Create a mixed dataset that combines SFT and DPO samples"""
        # Add task type indicators
        sft_with_type = sft_dataset.map(lambda x: {**x, "task_type": "sft"})
        dpo_with_type = dpo_dataset.map(lambda x: {**x, "task_type": "dpo"})
        
        # Just combine and shuffle - the proportions were already set correctly via upsampling
        mixed_dataset = concatenate_datasets([sft_with_type, dpo_with_type])
        mixed_dataset = mixed_dataset.shuffle(seed=self.training_cfg.seed)
        
        logging.info(f"Created mixed dataset: {len(sft_with_type)} SFT samples, {len(dpo_with_type)} DPO samples")
        logging.info(f"Lambda weights: SFT={self.lambda_sft}, DPO={self.lambda_dpo}")
        
        return mixed_dataset
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute mixed loss combining SFT and DPO losses"""
        # Accept additional kwargs that Unsloth might pass (like num_items_in_batch)
        total_loss = 0.0
        outputs = None
        batch_size = len(inputs["task_type"])
        
        # Compute SFT loss if we have SFT samples
        if "sft_indices" in inputs and len(inputs["sft_indices"]) > 0:
            sft_inputs = {
                "input_ids": inputs["sft_input_ids"],
                "attention_mask": inputs["sft_attention_mask"],
                "labels": inputs["sft_labels"]
            }
            sft_loss, sft_outputs = self._compute_sft_loss(model, sft_inputs)
            total_loss += self.lambda_sft * sft_loss
            outputs = sft_outputs  # Use SFT outputs as primary outputs
            
        # Compute DPO loss if we have DPO samples
        if "dpo_indices" in inputs and len(inputs["dpo_indices"]) > 0:
            dpo_inputs = {
                "chosen_input_ids": inputs["chosen_input_ids"],
                "chosen_attention_mask": inputs["chosen_attention_mask"],
                "rejected_input_ids": inputs["rejected_input_ids"],
                "rejected_attention_mask": inputs["rejected_attention_mask"]
            }
            dpo_loss = self._compute_dpo_loss(model, dpo_inputs)
            total_loss += self.lambda_dpo * dpo_loss
            
        # If we only had DPO samples and no SFT outputs, create dummy outputs
        if outputs is None:
            # Create minimal outputs for compatibility
            dummy_logits = torch.zeros((batch_size, 1, model.config.vocab_size), device=model.device)
            outputs = type('obj', (object,), {'logits': dummy_logits, 'loss': total_loss})()
            
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _compute_sft_loss(self, model, inputs):
        """Compute SFT loss for given inputs"""
        outputs = model(**inputs)
        return outputs.loss, outputs
    
    def _compute_dpo_loss(self, model, inputs):
        """Compute DPO loss for given inputs"""
        # Extract chosen and rejected sequences
        chosen_ids = inputs["chosen_input_ids"]
        rejected_ids = inputs["rejected_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        
        # Get logits for chosen and rejected sequences
        chosen_outputs = model(input_ids=chosen_ids, attention_mask=chosen_attention_mask)
        rejected_outputs = model(input_ids=rejected_ids, attention_mask=rejected_attention_mask)
        
        chosen_logits = chosen_outputs.logits
        rejected_logits = rejected_outputs.logits
        
        # Compute log probabilities
        chosen_logprobs = self._get_batch_logps(chosen_logits, chosen_ids, chosen_attention_mask)
        rejected_logprobs = self._get_batch_logps(rejected_logits, rejected_ids, rejected_attention_mask)
        
        # DPO loss
        logits_diff = chosen_logprobs - rejected_logprobs
        dpo_loss = -F.logsigmoid(self.beta * logits_diff).mean()
        
        return dpo_loss
    
    def _get_batch_logps(self, logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for a batch"""
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        
        # Get log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probabilities for actual tokens
        per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
        
        # Mask out padding tokens
        per_token_logps = per_token_logps * shift_attention_mask
        
        # Sum over sequence length
        return per_token_logps.sum(dim=1)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model"""
        return super().evaluate() 