import gc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_scheduler

from .base_trainer import BaseTrainer
from .train_utils import train_test_split


def compute_kl_divergence(lora_logits, base_logits):
    """
    Compute KL divergence between LoRA model distribution and base model distribution.

    Args:
        lora_logits: Logits from the LoRA-adapted model [batch_size, seq_len, vocab_size]
        base_logits: Logits from the base model [batch_size, seq_len, vocab_size]

    Returns:
        KL divergence loss (scalar tensor)
    """
    # More numerically stable KL divergence implementation
    lora_log_probs = torch.nn.functional.log_softmax(lora_logits, dim=-1)
    base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)

    kl_div = torch.sum(
        torch.exp(lora_log_probs) * (lora_log_probs - base_log_probs), dim=-1
    )

    # Average over sequence length and batch
    kl_div = kl_div.mean()

    return kl_div


class PositiveNegativeProxyTrainer(BaseTrainer):
    """
    A trainer class that implements training with separate outcome, proxy, and negative proxy datasets.
    This trainer supports different cycling modes:
    - "continuous": Continuously cycles proxy dataset when exhausted while completing a pass through outcome
    - "synchronized": Only resets all datasets when all are exhausted
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
            proxy_dataset: Proxy dataset for directing model behavior
            proxy_neg_dataset: Negative proxy dataset (for learning what not to do)
            truth_dataset: Truth dataset for evaluation
            collateral_dataset: Collateral dataset for evaluation
            truth_minus_proxy_dataset: Truth minus proxy dataset for evaluation
            exp_folder: Path to save experiment results
            device: Device to use for training
            seed: Random seed for reproducibility,
            split_proxy_dataset: Whether to split the proxy dataset into train and test. if false, will use the same set for both train and test
            split_outcome_dataset: Whether to split the outcome dataset into train and test. if false, will use the same set for both train and test
        """

        # Set default proxy weight if not defined in config
        self.lambda_proxy = getattr(training_cfg, "lambda_proxy", 1.0)

        # Set default negative proxy weight if not defined in config
        self.neg_lambda_proxy = getattr(training_cfg, "neg_lambda_proxy", 0.0)

        # Set default proxy cycling mode if not defined in config
        self.proxy_cycling_mode = getattr(
            training_cfg, "proxy_cycling_mode", "continuous"
        ).lower()

        # Validate proxy_cycling_mode
        valid_modes = ["continuous", "synchronized"]
        if self.proxy_cycling_mode not in valid_modes:
            logging.warning(
                f"Invalid proxy_cycling_mode: {self.proxy_cycling_mode}. Using default 'continuous' mode."
            )
            self.proxy_cycling_mode = "continuous"

        logging.info(f"Training using '{self.proxy_cycling_mode}' mode")
        logging.info(f"Using lambda_proxy: {self.lambda_proxy}")
        logging.info(f"Using neg_lambda_proxy: {self.neg_lambda_proxy}")

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            training_cfg=training_cfg,
            collate_fn=collate_fn,
            eval_fn=eval_fn,
            outcome_dataset=outcome_dataset,
            proxy_dataset=proxy_dataset,
            proxy_neg_dataset=proxy_neg_dataset,
            truth_dataset=truth_dataset,
            collateral_dataset=collateral_dataset,
            truth_minus_proxy_dataset=truth_minus_proxy_dataset,
            exp_folder=exp_folder,
            device=device,
            seed=seed,
            split_proxy_dataset=split_proxy_dataset,
            split_outcome_dataset=split_outcome_dataset,
        )
        # do this so that if the model is not peft we have a copy of it to get the base logits on proxy data
        if not self.training_cfg.is_peft:
            import copy

            self.base_model_copy = copy.deepcopy(self.model)
        else:
            self.base_model_copy = None

    def _prepare_datasets(
        self, split_proxy_dataset: bool = True, split_outcome_dataset: bool = True
    ):
        """
        Prepares training and evaluation datasets/dataloaders.

        For evaluation, creates dataloaders for all available datasets.
        For training, creates separate dataloaders for outcome, proxy, and proxy_neg datasets
        based on the configured proxy_cycling_mode (continuous or synchronized).

        Args:
            split_proxy_dataset: Whether to split the proxy dataset into train and test
            split_outcome_dataset: Whether to split the outcome dataset into train and test

        Returns:
            Tuple of (train_dataloaders, eval_dataloaders)
        """
        import logging

        from torch.utils.data import DataLoader

        outcome_dataset = self.datasets["outcome"]
        proxy_dataset = self.datasets["proxy"]
        proxy_neg_dataset = self.datasets["proxy_neg"]
        truth_dataset = self.datasets["truth"]
        collateral_dataset = self.datasets["collateral"]
        truth_minus_proxy_dataset = self.datasets["truth_minus_proxy"]

        # Initialize return dictionaries
        train_datasets = {}
        train_dataloaders = {}
        eval_dataloaders = {}

        # Process outcome dataset
        if outcome_dataset is not None and len(outcome_dataset) > 0:
            logging.info(
                f"Processing outcome dataset with {len(outcome_dataset)} samples"
            )

            # Split using the helper method
            outcome_train, outcome_test = self._split_dataset(
                outcome_dataset, split=split_outcome_dataset, seed=self.seed
            )

            # Store datasets
            train_datasets["outcome"] = outcome_train

            # Create train dataloader for outcome
            train_dataloaders["outcome"] = DataLoader(
                outcome_train,
                batch_size=self.training_cfg.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )

            # Create evaluation dataloader
            eval_dataloaders["outcome"] = self.get_eval_dataloader(outcome_test)

        # Process proxy dataset
        if proxy_dataset is not None and len(proxy_dataset) > 0:
            logging.info(
                f"[{self.__class__.__name__}] Processing proxy dataset with {len(proxy_dataset)} samples"
            )

            # Split using the helper method
            proxy_train, proxy_test = self._split_dataset(
                proxy_dataset, split=split_proxy_dataset, seed=self.seed
            )
            if self.training_cfg.limit_proxy_data_to:
                logging.info(
                    f"Limiting proxy train dataset to {self.training_cfg.limit_proxy_data_to} samples"
                )
                proxy_train = proxy_train.select(
                    range(self.training_cfg.limit_proxy_data_to)
                )
                logging.info(
                    f"Proxy train dataset: {proxy_train[: self.training_cfg.limit_proxy_data_to]}"
                )

            # Store datasets
            train_datasets["proxy"] = proxy_train

            # Create train dataloader for proxy
            train_dataloaders["proxy"] = DataLoader(
                proxy_train,
                batch_size=self.training_cfg.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )

            # Create evaluation dataloader
            eval_dataloaders["proxy"] = self.get_eval_dataloader(proxy_test)
        else:
            if self.lambda_proxy > 0:
                logging.warning(
                    f"[{self.__class__.__name__}] `proxy_dataset` is None or empty, but `lambda_proxy` is > 0. No proxy loss will be applied."
                )
            else:
                logging.info(
                    f"[{self.__class__.__name__}] No `proxy_dataset` provided. This is expected as `lambda_proxy` is 0."
                )

        # Process proxy_neg dataset
        if proxy_neg_dataset is not None and len(proxy_neg_dataset) > 0:
            logging.info(
                f"[{self.__class__.__name__}] Processing proxy_neg dataset with {len(proxy_neg_dataset)} samples"
            )

            # Split using the helper method
            proxy_neg_train, proxy_neg_test = self._split_dataset(
                proxy_neg_dataset, split=split_proxy_dataset, seed=self.seed
            )
            if self.training_cfg.limit_proxy_data_to:
                logging.info(
                    f"Limiting proxy_neg dataset to {self.training_cfg.limit_proxy_data_to} samples"
                )
                proxy_neg_train = proxy_neg_train.select(
                    range(self.training_cfg.limit_proxy_data_to)
                )
                logging.info(
                    f"Proxy neg train dataset: {proxy_neg_train[: self.training_cfg.limit_proxy_data_to]}"
                )
            # Store datasets
            train_datasets["proxy_neg"] = proxy_neg_train

            # Create train dataloader for proxy_neg
            train_dataloaders["proxy_neg"] = DataLoader(
                proxy_neg_train,
                batch_size=self.training_cfg.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )

            # Create evaluation dataloader
            eval_dataloaders["proxy_neg"] = self.get_eval_dataloader(proxy_neg_test)
        else:
            if self.neg_lambda_proxy > 0:
                logging.warning(
                    f"[{self.__class__.__name__}] `proxy_neg_dataset` is None or empty, but `neg_lambda_proxy` is > 0. No negative proxy loss will be applied."
                )
            else:
                logging.info(
                    f"[{self.__class__.__name__}] No `proxy_neg_dataset` provided. This is expected as `neg_lambda_proxy` is 0."
                )

        # Process evaluation-only datasets
        # Process truth dataset
        if truth_dataset is not None and len(truth_dataset) > 0:
            logging.info(f"Processing truth dataset with {len(truth_dataset)} samples")

            # Create evaluation dataloader
            eval_dataloaders["truth"] = self.get_eval_dataloader(truth_dataset)

        # Process collateral dataset
        if collateral_dataset is not None and len(collateral_dataset) > 0:
            logging.info(
                f"Processing collateral dataset with {len(collateral_dataset)} samples"
            )

            # Create evaluation dataloader
            eval_dataloaders["collateral"] = self.get_eval_dataloader(
                collateral_dataset
            )

        # Process truth_minus_proxy dataset
        if truth_minus_proxy_dataset is not None and len(truth_minus_proxy_dataset) > 0:
            logging.info(
                f"Processing truth_minus_proxy dataset with {len(truth_minus_proxy_dataset)} samples"
            )

            # Create evaluation dataloader
            eval_dataloaders["truth_minus_proxy"] = self.get_eval_dataloader(
                truth_minus_proxy_dataset
            )

        # Store separate train dataloaders for continuous and synchronized modes
        self.train_dataloaders = train_dataloaders
        logging.info(
            f"Created separate train dataloaders for {self.proxy_cycling_mode} mode"
        )

        # Calculate steps per epoch based on dataset sizes
        outcome_steps = (
            len(train_dataloaders["outcome"]) if "outcome" in train_dataloaders else 0
        )
        proxy_steps = (
            len(train_dataloaders["proxy"]) if "proxy" in train_dataloaders else 0
        )
        proxy_neg_steps = (
            len(train_dataloaders["proxy_neg"])
            if "proxy_neg" in train_dataloaders
            else 0
        )

        logging.info(
            f"Steps per epoch: outcome={outcome_steps}, proxy={proxy_steps}, proxy_neg={proxy_neg_steps}"
        )

        # Determine steps per epoch based on cycling mode
        if self.proxy_cycling_mode == "continuous":
            # In continuous mode, one epoch = one complete pass through outcome dataset
            self.steps_per_epoch = outcome_steps
            logging.info(
                f"Using outcome dataset size ({self.steps_per_epoch}) as steps per epoch"
            )
        else:  # synchronized mode
            # In synchronized mode, one epoch = pass through the larger dataset
            self.steps_per_epoch = max(outcome_steps, proxy_steps, proxy_neg_steps)
            logging.info(
                f"Using max dataset size ({self.steps_per_epoch}) as steps per epoch"
            )

        # Store reference to evaluation dataloaders
        self.eval_dataloaders = eval_dataloaders
        if self.training_cfg.save_datasets:
            # Save the train and eval datasets to disk
            self.save_datasets()
        return self.train_dataloaders, self.eval_dataloaders

    def _process_batch(self, batch, device):
        """
        Helper function to process a batch and move it to device

        Args:
            batch: Data batch
            device: Device to move tensors to

        Returns:
            Tuple of (input_ids, attention_mask, labels)
        """
        input_ids = torch.stack(batch["input_ids"]).to(device)
        attention_mask = torch.stack(batch["attention_mask"]).to(device)
        labels = (
            input_ids.clone()
            if "labels" not in batch
            else torch.stack(batch["labels"]).to(device)
        )
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels

    def _apply_kl_regularization(self, input_ids, attention_mask, batch_losses):
        """
        Applies KL divergence regularization to model outputs

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            batch_losses: Dictionary to track losses

        Returns:
            KL regularization loss to be added to total loss
        """
        kl_loss_value = 0.0
        kl_regularization = torch.tensor(0.0, device=self.device)

        if hasattr(self.training_cfg, "beta_kl") and self.training_cfg.beta_kl > 0:
            try:
                # Get LoRA model logits without labels for KL calculation
                lora_outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )
                lora_logits = lora_outputs.logits

                # Get base model logits
                if self.base_model_copy is None:
                    assert hasattr(self.model, "peft_config"), (
                        "Model is not PEFT but no copy exists to get the base logits for the proxy data"
                    )

                    original_mode_was_training = self.model.training
                    self.model.eval()
                    with self.model.disable_adapter():
                        with torch.no_grad():
                            base_outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=True,
                            )
                            base_logits = base_outputs.logits
                    if original_mode_was_training:
                        self.model.train()
                else:
                    assert self.base_model_copy is not None
                    original_mode_was_training = self.base_model_copy.training
                    self.base_model_copy.eval()
                    with torch.no_grad():
                        base_outputs = self.base_model_copy(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                        )
                        base_logits = base_outputs.logits
                    if original_mode_was_training:
                        self.base_model_copy.train()

                # Compute KL divergence
                kl_loss = compute_kl_divergence(lora_logits, base_logits)
                kl_loss_value = kl_loss.item()
                batch_losses["kl"] = kl_loss_value

                # Calculate weighted KL divergence
                kl_regularization = (
                    self.training_cfg.beta_kl
                    * kl_loss
                    / self.training_cfg.gradient_accumulation_steps
                )

                # Clean up tensors
                del lora_outputs, lora_logits, base_outputs, base_logits, kl_loss

            except AttributeError as e:
                logging.warning(
                    f"KL regularization failed: {e}. Continuing without KL."
                )

        return kl_regularization, kl_loss_value

    def _clean_memory(self):
        """Helper function to clean up memory"""
        torch.cuda.empty_cache()
        gc.collect()

    def get_standard_optimizer_and_scheduler(self, model, train_dataloaders=None):
        """Get standard optimizer and scheduler"""
        from torch.optim import AdamW
        from transformers import get_scheduler

        if train_dataloaders is None:
            if not hasattr(self, "train_dataloaders"):
                raise ValueError("Train dataloaders not found")
            train_dataloaders = self.train_dataloaders

        optimizer = AdamW(model.parameters(), lr=self.training_cfg.learning_rate)

        outcome_steps = len(train_dataloaders["outcome"])
        if "proxy" in train_dataloaders:
            proxy_steps = len(train_dataloaders["proxy"])
        elif "proxy_neg" in train_dataloaders:
            proxy_steps = len(train_dataloaders["proxy_neg"])
        else:
            proxy_steps = 0
        logging.info(f"Steps per epoch: outcome={outcome_steps}, proxy={proxy_steps}")

        # Determine steps per epoch based on cycling mode
        if self.proxy_cycling_mode == "continuous":
            # In continuous mode, one epoch = one complete pass through outcome dataset
            steps_per_epoch = outcome_steps
            logging.info(
                f"Using outcome dataset size ({steps_per_epoch}) as steps per epoch"
            )
        else:  # synchronized mode
            # In synchronized mode, one epoch = pass through the larger dataset
            steps_per_epoch = max(outcome_steps, proxy_steps)
            logging.info(
                f"Using max dataset size ({steps_per_epoch}) as steps per epoch"
            )
        # Calculate training steps
        num_update_steps_per_epoch = (
            steps_per_epoch // self.training_cfg.gradient_accumulation_steps
        )
        num_training_steps = num_update_steps_per_epoch * self.training_cfg.epochs

        lr_scheduler = get_scheduler(
            name=self.training_cfg.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.training_cfg.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, lr_scheduler

    def train(self, save_checkpoint_results_fn=None, save_results_fn=None):
        """
        Trains model with separate outcome, proxy, and proxy_neg datasets.

        Training occurs using one of two modes:
        - "continuous": Continuously cycles proxy dataset when exhausted while completing a pass through outcome
        - "synchronized": Only resets all datasets when all are exhausted

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
                    train_losses: List[float],
                    eval_results: Dict[str, Any],
                    output_dir: str
                ) -> None

        Returns:
            Tuple of (model, train_losses, eval_results)
        """
        self.model = self.prepare_for_training()

        # Prepare datasets/dataloaders
        train_dataloaders, eval_dataloaders = (
            self.train_dataloaders,
            self.eval_dataloaders,
        )
        optimizer, lr_scheduler = self.get_standard_optimizer_and_scheduler(
            self.model, train_dataloaders
        )

        # Initialize training tracking
        train_losses = []
        outcome_losses = []
        proxy_losses = []
        proxy_neg_losses = []

        # Initial evaluation
        logging.info("Running initial evaluations before training")
        eval_results = self.evaluate(
            self.model, self.tokenizer, eval_dataloaders, eval_results=None, epoch=0
        )

        logging.info(f"Initial evaluation results: {eval_results}")
        # get the first train losses (on the train dataset)

        init_train_loss = 0.0
        init_proxy_loss = 0.0
        init_outcome_loss = 0.0

        # Training loop
        for epoch in range(self.training_cfg.epochs):
            logging.info(f"\nEpoch {epoch + 1}/{self.training_cfg.epochs}")
            logging.info("GPU memory at start of epoch:")
            if hasattr(self, "get_gpu_memory_info"):
                self.get_gpu_memory_info()

            self.model.train()
            epoch_total_loss = 0
            epoch_outcome_loss = 0
            epoch_proxy_loss = 0
            epoch_proxy_neg_loss = 0

            # ====== SEPARATE DATASETS TRAINING LOOP ======
            # Create iterators for each dataset
            outcome_iter = (
                iter(train_dataloaders["outcome"])
                if "outcome" in train_dataloaders
                else None
            )
            proxy_iter = (
                iter(train_dataloaders["proxy"])
                if "proxy" in train_dataloaders
                else None
            )
            proxy_neg_iter = (
                iter(train_dataloaders["proxy_neg"])
                if "proxy_neg" in train_dataloaders
                else None
            )

            # Track dataset states
            outcome_active = outcome_iter is not None
            proxy_active = proxy_iter is not None
            proxy_neg_active = proxy_neg_iter is not None

            # Count for accumulated batches and processed batches
            batch_count = 0
            outcome_batches_processed = 0
            proxy_batches_processed = 0
            proxy_neg_batches_processed = 0
            proxy_cycles = 0
            proxy_neg_cycles = 0

            # For progress bar description
            progress_desc = (
                "Outcome Batches"
                if self.proxy_cycling_mode == "continuous"
                else "Batches"
            )

            # Continue until stopping condition based on cycling mode
            first_iter = True if epoch == 0 else False
            with tqdm(total=self.steps_per_epoch, desc=progress_desc) as pbar:
                while True:
                    outcome_loss = torch.tensor(0.0, device=self.device)
                    proxy_loss = torch.tensor(0.0, device=self.device)
                    proxy_neg_loss = torch.tensor(0.0, device=self.device)
                    batch_losses = {}

                    # Process outcome data if available
                    if outcome_active:
                        try:
                            outcome_batch = next(outcome_iter)
                            outcome_batches_processed += 1

                            # Process outcome batch
                            input_ids, attention_mask, labels = self._process_batch(
                                outcome_batch, self.device
                            )

                            outcome_outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )

                            outcome_loss = (
                                outcome_outputs.loss
                                / self.training_cfg.gradient_accumulation_steps
                            )
                            batch_losses["outcome"] = outcome_outputs.loss.item()
                            if first_iter:
                                outcome_losses.append(batch_losses["outcome"])

                            # Clean up tensors
                            del outcome_outputs, input_ids, attention_mask, labels
                            self._clean_memory()

                        except StopIteration:
                            outcome_active = False
                            logging.info("Outcome dataset exhausted")

                    # Process proxy data if available
                    if proxy_active:
                        try:
                            proxy_batch = next(proxy_iter)
                            proxy_batches_processed += 1

                            # Process proxy batch
                            input_ids, attention_mask, labels = self._process_batch(
                                proxy_batch, self.device
                            )

                            proxy_outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )

                            proxy_loss = (
                                proxy_outputs.loss
                                / self.training_cfg.gradient_accumulation_steps
                            )
                            batch_losses["proxy"] = proxy_outputs.loss.item()
                            if first_iter:
                                proxy_losses.append(batch_losses["proxy"])

                            # Apply KL regularization if configured (only to proxy data)
                            kl_regularization, kl_loss_value = (
                                self._apply_kl_regularization(
                                    input_ids, attention_mask, batch_losses
                                )
                            )

                            proxy_loss += kl_regularization

                            # Clean up tensors
                            del proxy_outputs, input_ids, attention_mask, labels
                            self._clean_memory()

                        except StopIteration:
                            proxy_active = False

                            # Reset proxy iterator immediately in continuous mode
                            if self.proxy_cycling_mode == "continuous":
                                proxy_iter = iter(train_dataloaders["proxy"])
                                proxy_active = True
                                proxy_cycles += 1
                                self._clean_memory()
                                logging.info(
                                    f"Proxy dataset completed cycle #{proxy_cycles}"
                                )
                                proxy_batch = next(proxy_iter)
                                proxy_batches_processed += 1

                                # Process proxy batch
                                input_ids, attention_mask, labels = self._process_batch(
                                    proxy_batch, self.device
                                )

                                proxy_outputs = self.model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                )

                                proxy_loss = (
                                    proxy_outputs.loss
                                    / self.training_cfg.gradient_accumulation_steps
                                )
                                batch_losses["proxy"] = proxy_outputs.loss.item()
                                if first_iter:
                                    proxy_losses.append(batch_losses["proxy"])

                                # Apply KL regularization if configured (only to proxy data)
                                kl_regularization, kl_loss_value = (
                                    self._apply_kl_regularization(
                                        input_ids, attention_mask, batch_losses
                                    )
                                )

                                proxy_loss += kl_regularization

                                # Clean up tensors
                                del proxy_outputs, input_ids, attention_mask, labels
                                self._clean_memory()
                            else:
                                logging.info("Proxy dataset exhausted")

                    # Process negative proxy data if available
                    if proxy_neg_active:
                        try:
                            proxy_neg_batch = next(proxy_neg_iter)
                            proxy_neg_batches_processed += 1

                            # Process negative proxy batch
                            input_ids, attention_mask, labels = self._process_batch(
                                proxy_neg_batch, self.device
                            )

                            proxy_neg_outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )

                            proxy_neg_loss = (
                                proxy_neg_outputs.loss
                                / self.training_cfg.gradient_accumulation_steps
                            )
                            batch_losses["proxy_neg"] = proxy_neg_outputs.loss.item()
                            if first_iter:
                                proxy_neg_losses.append(batch_losses["proxy_neg"])
                            # Clean up tensors
                            del proxy_neg_outputs, input_ids, attention_mask, labels
                            self._clean_memory()

                        except StopIteration:
                            proxy_neg_active = False

                            # Reset negative proxy iterator immediately in continuous mode
                            if self.proxy_cycling_mode == "continuous":
                                proxy_neg_iter = iter(train_dataloaders["proxy_neg"])
                                proxy_neg_active = True
                                proxy_neg_cycles += 1
                                self._clean_memory()
                                logging.info(
                                    f"Negative proxy dataset completed cycle #{proxy_neg_cycles}"
                                )
                                proxy_neg_batch = next(proxy_neg_iter)
                                proxy_neg_batches_processed += 1

                                # Process negative proxy batch
                                input_ids, attention_mask, labels = self._process_batch(
                                    proxy_neg_batch, self.device
                                )

                                proxy_neg_outputs = self.model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                )

                                proxy_neg_loss = (
                                    proxy_neg_outputs.loss
                                    / self.training_cfg.gradient_accumulation_steps
                                )
                                batch_losses["proxy_neg"] = (
                                    proxy_neg_outputs.loss.item()
                                )
                                if first_iter:
                                    proxy_neg_losses.append(batch_losses["proxy_neg"])
                                # Clean up tensors
                                del proxy_neg_outputs, input_ids, attention_mask, labels
                                self._clean_memory()
                            else:
                                logging.info("Negative proxy dataset exhausted")

                    if first_iter:
                        init_outcome_loss = (
                            outcome_losses[0] if len(outcome_losses) > 0 else 0.0
                        )
                        init_proxy_loss = (
                            proxy_losses[0] if len(proxy_losses) > 0 else 0.0
                        )
                        init_proxy_neg_loss = (
                            proxy_neg_losses[0] if len(proxy_neg_losses) > 0 else 0.0
                        )
                        logging.info("Appending first iteration losses")
                        logging.info(f"Outcome losses: {outcome_losses}")
                        logging.info(f"Proxy losses: {proxy_losses}")
                        logging.info(f"Proxy neg losses: {proxy_neg_losses}")
                        init_combined_loss = (
                            init_outcome_loss
                            + self.lambda_proxy * init_proxy_loss
                            - self.neg_lambda_proxy * init_proxy_neg_loss
                        )
                        train_losses.append(init_combined_loss)
                        logging.info(f"Initial train losses: {train_losses}")
                    logging.info("setting first_iter to False")
                    first_iter = False
                    # Check stopping conditions based on cycling mode
                    if self.proxy_cycling_mode == "continuous":
                        # In continuous mode, stop when outcome dataset is exhausted
                        if not outcome_active:
                            break
                    else:  # synchronized mode
                        # In synchronized mode, stop when all datasets are exhausted or max steps reached
                        if (
                            not outcome_active
                            and not proxy_active
                            and not proxy_neg_active
                        ):
                            break
                        if batch_count >= self.steps_per_epoch:
                            break

                    # Only proceed if at least one dataset provided a batch
                    if outcome_active or proxy_active or proxy_neg_active:
                        # Combine losses with weighting
                        combined_loss = 0

                        if outcome_active:
                            combined_loss += outcome_loss
                            epoch_outcome_loss += (
                                outcome_loss.item()
                                * self.training_cfg.gradient_accumulation_steps
                            )

                        if proxy_active:
                            combined_loss += self.lambda_proxy * proxy_loss
                            epoch_proxy_loss += (
                                proxy_loss.item()
                                * self.training_cfg.gradient_accumulation_steps
                            )

                        if proxy_neg_active:
                            # Apply negative regularization for proxy_neg dataset
                            # The negative sign discourages producing unwanted responses
                            combined_loss -= self.neg_lambda_proxy * proxy_neg_loss
                            # Track the negative proxy loss for logging
                            epoch_proxy_neg_loss += (
                                proxy_neg_loss.item()
                                * self.training_cfg.gradient_accumulation_steps
                            )

                        # Backward pass
                        combined_loss.backward()

                        # Track loss for reporting
                        curr_loss = (
                            combined_loss.item()
                            * self.training_cfg.gradient_accumulation_steps
                        )
                        epoch_total_loss += curr_loss

                        # Update weights every gradient accumulation step
                        batch_count += 1
                        if (
                            batch_count % self.training_cfg.gradient_accumulation_steps
                            == 0
                        ):
                            # Apply gradient clipping if configured
                            if (
                                hasattr(self.training_cfg, "max_grad_norm")
                                and self.training_cfg.max_grad_norm > 0
                            ):
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.training_cfg.max_grad_norm,
                                )

                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()

                        # Update progress
                        pbar.update(1)
                        if pbar.n > pbar.total:
                            pbar.total = pbar.n  # Adjust if we go over expected steps

                        # Log progress
                        if batch_count % self.training_cfg.logging_steps == 0:
                            if self.proxy_cycling_mode == "continuous":
                                logging.info(
                                    f"Batch {batch_count}/{self.steps_per_epoch} "
                                    f"({(batch_count / self.steps_per_epoch) * 100:.1f}%) - "
                                    f"Loss: {curr_loss:.4f} "
                                    f"(Outcome: {batch_losses.get('outcome', 0):.4f}, "
                                    f"Proxy: {batch_losses.get('proxy', 0):.4f}, "
                                    f"Neg Proxy: {batch_losses.get('proxy_neg', 0):.4f}, "
                                    f"KL: {batch_losses.get('kl', 0):.4f}) - "
                                    f"Proxy cycles: {proxy_cycles}"
                                )
                            else:
                                logging.info(
                                    f"Batch {batch_count}/{self.steps_per_epoch} "
                                    f"({(batch_count / self.steps_per_epoch) * 100:.1f}%) - "
                                    f"Loss: {curr_loss:.4f} "
                                    f"(Outcome: {batch_losses.get('outcome', 0):.4f}, "
                                    f"Proxy: {batch_losses.get('proxy', 0):.4f}, "
                                    f"Neg Proxy: {batch_losses.get('proxy_neg', 0):.4f}, "
                                    f"KL: {batch_losses.get('kl', 0):.4f})"
                                )

                        # In synchronized mode: if all datasets are exhausted but we haven't reached max_steps,
                        # reset the iterators and continue
                        if (
                            self.proxy_cycling_mode == "synchronized"
                            and not outcome_active
                            and not proxy_active
                            and not proxy_neg_active
                            and batch_count < self.steps_per_epoch
                        ):
                            outcome_iter = (
                                iter(train_dataloaders["outcome"])
                                if "outcome" in train_dataloaders
                                else None
                            )
                            proxy_iter = (
                                iter(train_dataloaders["proxy"])
                                if "proxy" in train_dataloaders
                                else None
                            )
                            proxy_neg_iter = (
                                iter(train_dataloaders["proxy_neg"])
                                if "proxy_neg" in train_dataloaders
                                else None
                            )
                            outcome_active = outcome_iter is not None
                            proxy_active = proxy_iter is not None
                            proxy_neg_active = proxy_neg_iter is not None
                            logging.info(
                                "Restarting all dataset iterators to complete the epoch"
                            )

                # Handle any remaining gradient accumulation
                if batch_count % self.training_cfg.gradient_accumulation_steps != 0:
                    # Apply gradient clipping if configured
                    if (
                        hasattr(self.training_cfg, "max_grad_norm")
                        and self.training_cfg.max_grad_norm > 0
                    ):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.training_cfg.max_grad_norm
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Log cycle information
            logging.info(
                f"Epoch {epoch + 1} complete: Processed {outcome_batches_processed} outcome batches "
                f"with {proxy_cycles} proxy dataset cycles ({proxy_batches_processed} proxy batches) "
                f"and {proxy_neg_cycles} negative proxy cycles ({proxy_neg_batches_processed} neg proxy batches)"
            )

            # Compute average losses for the epoch
            if batch_count > 0:
                avg_loss = epoch_total_loss / batch_count
                avg_outcome_loss = epoch_outcome_loss / batch_count
                avg_proxy_loss = epoch_proxy_loss / batch_count
                avg_proxy_neg_loss = epoch_proxy_neg_loss / batch_count
                train_losses.append(avg_loss)
                outcome_losses.append(avg_outcome_loss)
                proxy_losses.append(avg_proxy_loss)
                proxy_neg_losses.append(avg_proxy_neg_loss)
            else:
                logging.warning("No batches processed in this epoch")
                train_losses.append(0)
                outcome_losses.append(0)
                proxy_losses.append(0)
                proxy_neg_losses.append(0)

            # Evaluate model
            eval_results = self.evaluate(
                self.model,
                self.tokenizer,
                eval_dataloaders,
                eval_results,
                epoch=epoch + 1,
                is_final_epoch=epoch == (self.training_cfg.epochs - 1),
            )

            # Save checkpoint
            if save_checkpoint_results_fn:
                save_checkpoint_results_fn(
                    self.model,
                    {
                        "train": train_losses,
                        "proxy": proxy_losses,
                        "outcome": outcome_losses,
                        "proxy_neg": proxy_neg_losses,
                    },
                    eval_results,
                    output_dir=f"{self.exp_folder}/checkpoints/epoch_{epoch}",
                    epoch=epoch,
                )

            # Log epoch results
            logging.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")
            logging.info(
                f"Epoch {epoch + 1}: Average Outcome Loss = {avg_outcome_loss:.4f}"
            )
            logging.info(
                f"Epoch {epoch + 1}: Average Proxy Loss = {avg_proxy_loss:.4f}"
            )
            logging.info(
                f"Epoch {epoch + 1}: Average Negative Proxy Loss = {avg_proxy_neg_loss:.4f}"
            )
            logging.info(f"Epoch {epoch + 1}: Evaluation Results = {eval_results}")
            logging.info("GPU memory at end of epoch:")
            if hasattr(self, "get_gpu_memory_info"):
                self.get_gpu_memory_info()

            # Save model checkpoint if configured
        if self.training_cfg.save_model_locally:
            self.save_model_locally(self.model, self.tokenizer)
        # Push model to hub if configured
        if self.training_cfg.push_to_hub:
            self.push_model(self.model, self.tokenizer)

        # Save final results
        if save_results_fn is not None:
            save_results_fn(
                {
                    "train": train_losses,
                    "proxy": proxy_losses,
                    "outcome": outcome_losses,
                    "proxy_neg": proxy_neg_losses,
                },
                eval_results,
                output_dir=f"{self.exp_folder}/results",
            )

        return (
            self.model,
            {
                "train": train_losses,
                "proxy": proxy_losses,
                "outcome": outcome_losses,
                "proxy_neg": proxy_neg_losses,
            },
            eval_results,
        )
