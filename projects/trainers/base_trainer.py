import logging
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import os
from .train_utils import seed_all, train_test_split


def has_active_adapters(model):
    """Check if model has active (non-merged) PEFT adapters."""
    if not hasattr(model, "peft_config"):
        return False

    # Check if there are active adapters
    try:
        if hasattr(model, "active_adapters"):
            has_adapters = len(model.active_adapters()) > 0
            print(f"Model has active adapters: {has_adapters}")
            return has_adapters
    except ValueError as e:
        print(f"Error checking active adapters: {e}")
        print("This means the model has no adapters")
        return False

    # Alternative: check if base_model exists (indicates active PEFT)
    if hasattr(model, "base_model"):
        return True

    return False


class BaseTrainer:
    """
    Base trainer class that handles dataset preparation and common functionality.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        training_cfg: Any,
        collate_fn: Callable,
        eval_fn: Callable,
        outcome_dataset: Optional[Any] = None,
        proxy_dataset: Optional[Any] = None,
        proxy_neg_dataset: Optional[Any] = None,
        truth_dataset: Optional[Any] = None,
        collateral_dataset: Optional[Any] = None,
        truth_minus_proxy_dataset: Optional[Any] = None,
        exp_folder: str = None,
        device: str = "cuda",
        seed: int = None,
        split_proxy_dataset: bool = True,
        split_outcome_dataset: bool = True,
    ):
        """
        Initialize the trainer with model, tokenizer, and datasets.

        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            training_cfg: Training configuration
            collate_fn: Function to collate batches
            eval_fn: Function to evaluate the model
            outcome_dataset: Main outcome dataset
            proxy_dataset: Proxy dataset
            proxy_neg_dataset: Negative proxy dataset
            truth_dataset: Truth dataset
            collateral_dataset: Collateral dataset
            truth_minus_proxy_dataset: Truth minus proxy dataset
            device: Device to use for training
            seed: Random seed for reproducibility
        """
        self.model = model
        self.tokenizer = tokenizer
        self.training_cfg = training_cfg
        self.collate_fn = collate_fn
        self.device = device if torch.cuda.is_available() else "cpu"
        self.exp_folder = exp_folder
        self.seed = seed
        if seed is not None:
            seed_all(seed)

        # Move model to device
        self.model.to(self.device)

        # Store original datasets
        self.datasets = {
            "outcome": outcome_dataset,
            "proxy": proxy_dataset,
            "proxy_neg": proxy_neg_dataset,
            "truth": truth_dataset,
            "collateral": collateral_dataset,
            "truth_minus_proxy": truth_minus_proxy_dataset,
        }

        # Initialize dataloaders
        self.dataloaders = {}

        # Prepare datasets and create dataloaders
        print("Split outcome dataset: ", split_outcome_dataset)
        print("Split proxy dataset: ", split_proxy_dataset)
        self._prepare_datasets(split_proxy_dataset, split_outcome_dataset)

        self.huggingface_token = os.getenv("HF_TOKEN")
        self.evaluate = eval_fn
    
    def _clean_memory(self):
        """Helper function to clean up memory"""
        import gc
        import torch
        torch.cuda.empty_cache()
        gc.collect()

    def push_model(self, model, tokenizer):
        """Save and push model to Hugging Face Hub."""
        print(f"PUSHING MODEL")
        print(f"Model type: {type(model)}")
        print(f"Has adapters: {hasattr(model, 'peft_config')}")
        try:
            finetuned_model_id = self.training_cfg.finetuned_model_id
            if self.training_cfg.merge_before_push:
                logging.info("Merging and unloading")
                model = model.merge_and_unload()

            logging.info("pushing to huggingface hub")
            model.push_to_hub(
                finetuned_model_id,
                token=self.huggingface_token,
            )

            tokenizer.push_to_hub(finetuned_model_id, token=self.huggingface_token)
            logging.info(f"Model pushed to Hugging Face Hub: {finetuned_model_id}")

        except Exception as e:
            import traceback

            logging.info(f"Failed to push model. Error: {str(e)}")
            logging.info("Full traceback:")
            traceback.print_exc()
            logging.info("Failed to push model")

    def get_eval_dataloader(self, dataset: str) -> DataLoader:
        """
        Returns a dataloader
        """
        return DataLoader(
            dataset,
            batch_size=self.training_cfg.per_device_eval_batch_size
            if hasattr(self.training_cfg, "per_device_eval_batch_size")
            else int(self.training_cfg.per_device_train_batch_size / 4 + 1),
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def get_standard_optimizer_and_scheduler(
        self, model, train_dataloader=None, epochs=None
    ):
        """
        Get the optimizer and scheduler for standard training.
        """
        from torch.optim import AdamW
        from transformers import get_scheduler

        if epochs is None:
            epochs = self.training_cfg.epochs
        if train_dataloader is None:
            if not hasattr(self, "train_dataloader"):
                raise ValueError("Train dataloader not found")
            train_dataloader = self.train_dataloader

        optimizer = AdamW(model.parameters(), lr=self.training_cfg.learning_rate)
        # Calculate training steps
        num_update_steps_per_epoch = (
            len(train_dataloader) // self.training_cfg.gradient_accumulation_steps
        )
        num_training_steps = num_update_steps_per_epoch * epochs

        lr_scheduler = get_scheduler(
            name=self.training_cfg.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.training_cfg.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, lr_scheduler

    def save_model_locally(self, model, tokenizer):
        """
        Save a Hugging Face model and tokenizer to a local directory.

        Args:
            finetuned_model_id (str): Directory name where the model will be saved
            model: The trained Hugging Face model to save
            tokenizer: The tokenizer associated with the model
        """
        import datetime
        import os

        finetuned_model_id = self.training_cfg.finetuned_model_id
        finetuned_model_id = finetuned_model_id.replace("/", "_")
        save_dir = os.path.expanduser(
            f"~/../dev/shm/finetuned_models/{finetuned_model_id}"
        )
        os.makedirs(save_dir, exist_ok=True)

        if self.training_cfg.merge_before_push:
            logging.info("Merging and unloading")
            model = model.merge_and_unload()

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logging.info(f"Model and tokenizer saved locally to: {save_dir}")

    def save_datasets(self):
        """
        Saves eval datasets to disk as jsonl
        """
        data_timestamp_dir = os.path.join(
            self.exp_folder,
            "datasets",
            self.training_cfg.timestamp if self.training_cfg.timestamp else "",
        )
        if not os.path.exists(data_timestamp_dir):
            os.makedirs(data_timestamp_dir, exist_ok=True)
        logging.info(f"Saving datasets to {data_timestamp_dir}")
        for name, dataloader in self.eval_dataloaders.items():
            if "proxy_neg" in name:
                # Skip negative proxy datasets
                continue
            if "outcome" in name:
                continue
            if "proxy" in name:
                # Skip proxy datasets
                continue
            import copy
            import json

            dump_path = os.path.join(data_timestamp_dir, f"{name}_eval_dataset.jsonl")
            dump_dataset = copy.deepcopy(dataloader.dataset)
            # remove the input_ids, prompt_input_ids, attention_mask, prompt_attention_mask columns
            dump_dataset.remove_columns(
                [
                    "input_ids",
                    "prompt_input_ids",
                    "attention_mask",
                    "prompt_attention_mask",
                ]
            )
            with open(dump_path, "w") as f:
                for item in dump_dataset:
                    stripped_item = {
                        k: v
                        for k, v in item.items()
                        if k
                        not in [
                            "input_ids",
                            "prompt_input_ids",
                            "attention_mask",
                            "prompt_attention_mask",
                            "prompt_text",
                            "text",
                        ]
                    }
                    f.write(f"{json.dumps(stripped_item)}\n")

    def _prepare_datasets(
        self, split_proxy_dataset: bool = True, split_outcome_dataset: bool = True
    ):
        """
        Prepare all datasets and create their dataloaders.

        Args:
            split_proxy_dataset: Whether to split the proxy dataset into train and test
            split_outcome_dataset: Whether to split the outcome dataset into train and test
        """
        pass

    def prepare_for_training(self, model=None):
        """
        Prepare the model and datasets for training.
        """
        if model is None:
            model = self.model
        if self.training_cfg.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        print("Preparing model for training")
        print(f"Model type: {type(model)}")
        if self.training_cfg.is_peft and not has_active_adapters(model):
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.training_cfg.r,
                target_modules=self.training_cfg.target_modules,
                lora_alpha=self.training_cfg.lora_alpha,
                lora_dropout=self.training_cfg.lora_dropout,
                bias=self.training_cfg.lora_bias,
                use_rslora=self.training_cfg.use_rslora,
            )
            if hasattr(model, "peft_config"):
                # adapter is merged
                # ensure this is a non-peft object before applying PEFT
                print("Model is already a PEFT model, getting a clean model")
                model.save_pretrained("merged-model")

                #        Reload without any PEFT state
                # from experiment_utils import load_model_and_tokenizer
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained("merged-model")
                model.to(self.device)
                # does the model have a peft config now?
                if hasattr(model, "peft_config"):
                    print("Model type after reloading: ", type(model))
                    print("Model still has a PEFT config, merging again")
                # del merged model directory
                import shutil

                shutil.rmtree("merged-model", ignore_errors=True)
            print(f"Using PEFT model with {lora_config}")
            model = get_peft_model(model, lora_config)
            print(f"Model type: {type(model)}")
            print(f"Has adapters: {hasattr(model, 'peft_config')}")
        model.to(self.device)
        model.train()
        return model

    def train_step(self, model, tokenizer, batch: Dict, device: str) -> torch.Tensor:
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
        return outputs.loss

    def train(
        self,
        save_checkpoint_results_fn: Optional[Callable] = None,
        save_results_fn: Optional[Callable] = None,
    ):
        """
        Train the model using the specified datasets.

        Args:
            save_checkpoint_results_fn: Optional function to save checkpoint results.
                Function signature should be:
                def save_checkpoint_results_fn(
                    model: torch.nn.Module,
                    train_losses: List[float],
                    eval_results: Dict[str, Any],
                    output_dir: str,
                    epoch: int
                ) -> None

            save_results_fn: Optional function to save final training results.
                Function signature should be:
                def save_results_fn(
                    model: torch.nn.Module,
                    train_losses: List[float],
                    eval_results: Dict[str, Any],
                    output_dir: str
                ) -> None

        Returns:
            Tuple of (model, train_losses, eval_results) or
            (model, train_losses, eval_results, grad_accum, update_counts) if collecting gradients
        """
        pass

    def _split_dataset(
        self, dataset, split: bool = True, seed: Optional[int] = None
    ) -> Tuple[Any, Any]:
        """
        Helper method to split a dataset into train and test sets.

        Args:
            dataset: Dataset to split
            split: Whether to split the dataset. If False, returns the same dataset for both train and test
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if not split:
            return dataset, dataset

        train_split = getattr(self.training_cfg, "train_split", 0.9)
        splits = train_test_split(dataset, seed=seed, train_split=train_split)
        return splits
