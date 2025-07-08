import gc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from .base_trainer import BaseTrainer
from .train_utils import train_test_split


def create_preference_dataset(proxy_dataset, proxy_neg_dataset):
    """
    Create a Hugging Face dataset with preference pairs from proxy and proxy_neg datasets.

    Args:
        proxy_dataset: Hugging Face dataset containing preferred responses
        proxy_neg_dataset: Hugging Face dataset containing rejected responses

    Returns:
        Hugging Face dataset with preference pairs
    """
    from datasets import Dataset

    # Ensure both datasets have the same length for proper pairing
    min_len = min(len(proxy_dataset), len(proxy_neg_dataset))
    logging.info(
        f"Creating preference dataset with {min_len} pairs from "
        f"{len(proxy_dataset)} proxy and {len(proxy_neg_dataset)} proxy_neg samples"
    )

    # Create preference pairs by combining samples at the same index
    preference_data = []
    for i in range(min_len):
        proxy_sample = proxy_dataset[i]
        proxy_neg_sample = proxy_neg_dataset[i]
        proxy_sample_text = proxy_sample.get("text", "")
        proxy_sample_prompt_text = proxy_sample.get("prompt_text", "")
        proxy_neg_sample_text = proxy_neg_sample.get("text", "")
        proxy_neg_sample_prompt_text = proxy_neg_sample.get("prompt_text", "")
        logging.info(f"Chosen sample text: {proxy_sample_text}")
        logging.info(f"Rejected sample text: {proxy_neg_sample_text}")
        logging.info(f"Chosen sample prompt text: {proxy_sample_prompt_text}")
        logging.info(f"Rejected sample prompt text: {proxy_neg_sample_prompt_text}")
        # Create preference pair
        # what's the type of input ids
        logging.info(f"type of chosen input ids: {type(proxy_sample.get('input_ids'))}")
        logging.info(
            f"type of rejected input ids: {type(proxy_neg_sample.get('input_ids'))}"
        )
        logging.info(
            f"type of attention mask: {type(proxy_sample.get('attention_mask'))}"
        )
        preference_pair = {
            "chosen_input_ids": proxy_sample.get("input_ids"),
            "chosen_attention_mask": proxy_sample.get("attention_mask"),
            "rejected_input_ids": proxy_neg_sample.get("input_ids"),
            "rejected_attention_mask": proxy_neg_sample.get("attention_mask"),
            "prompt_attention_mask": proxy_sample.get("prompt_attention_mask"),
        }

        # Add any additional fields that might be useful
        if "labels" in proxy_sample:
            preference_pair["chosen_labels"] = proxy_sample["labels"]
        if "labels" in proxy_neg_sample:
            preference_pair["rejected_labels"] = proxy_neg_sample["labels"]

        preference_data.append(preference_pair)

    # Create Hugging Face dataset
    preference_dataset = Dataset.from_list(preference_data)
    logging.info(f"Created preference dataset with {len(preference_dataset)} samples")

    return preference_dataset


def compute_log_probabilities(
    logits,
    input_ids,
    pad_token,
    prompt_attention_mask=None,
    exclude_pad_from_log_probs=False,
):
    """
    Computes the log probabilities of the input IDs given the logits.
    Return shape: [batch_size]
    exclude_pad_from_log_probs: If True, excludes the padding token from the log probabilities.
    Set to False for training because empirically yielding better results
    Excludes the prompt and the padding from the log probabilities.
    """
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.zeros_like(input_ids)

    logits = logits[:, :-1, :]  # Remove the last token's logits
    labels = input_ids[:, 1:]  # Shift the inputs to the right to get labels
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

    response_mask = 1 - prompt_attention_mask[:, 1:]
    if exclude_pad_from_log_probs and pad_token is not None:
        pad_mask = (labels != pad_token).long()
        final_mask = response_mask * pad_mask
    else:
        final_mask = response_mask

    final_mask = final_mask.float()
    response_log_probs = (token_log_probs * final_mask).sum(dim=-1)
    response_lengths = final_mask.sum(dim=-1).clamp(min=1)

    return response_log_probs / response_lengths


class DPO(torch.nn.Module):
    """DPO loss implementation"""

    def __init__(self, beta=0.1, beta_kl=0.0):
        super(DPO, self).__init__()
        self.beta = beta
        self.beta_kl = beta_kl

    def forward(
        self, chosen_logp, rejected_logp, reference_chosen_logp, reference_rejected_logp
    ):
        """
        chosen_logp: [batch_size]
        rejected_logp: [batch_size]
        reference_chosen_logp: [batch_size]
        reference_rejected_logp: [batch_size]
        """
        # In this implementation, we use the property that log(p1/p2) = log(p1) - log(p2)
        chosen_ratio = chosen_logp - reference_chosen_logp
        rejected_ratio = rejected_logp - reference_rejected_logp
        subtracted_ratios = self.beta * (chosen_ratio - rejected_ratio)

        losses = -torch.nn.functional.logsigmoid(subtracted_ratios)
        return losses.mean()


def get_chosen_and_rejected_logits(
    model, batch, pad_token, is_reference=False, use_cache=False
):
    """Get logits and compute log probabilities for chosen and rejected responses"""
    if is_reference:
        with torch.no_grad():
            chosen_logits = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            ).logits
            rejected_logits = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            ).logits
    else:
        chosen_logits = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        ).logits
        rejected_logits = model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        ).logits

    # For DPO, we typically don't have separate prompt attention masks
    # so we use the full attention masks
    chosen_logprobs = compute_log_probabilities(
        chosen_logits,
        batch["chosen_input_ids"],
        pad_token=pad_token,
        prompt_attention_mask=batch.get("prompt_attention_mask", None),
    )
    rejected_logprobs = compute_log_probabilities(
        rejected_logits,
        batch["rejected_input_ids"],
        pad_token=pad_token,
        prompt_attention_mask=batch.get("prompt_attention_mask", None),
    )
    return chosen_logprobs, rejected_logprobs


class DPOTrainer(BaseTrainer):
    """
    A trainer class that implements training with outcome loss + DPO loss.
    Uses matched samples from proxy and proxy_neg datasets to create preference pairs.
    Supports different cycling modes like PositiveNegativeProxyTrainer.
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
        Initialize the DPO trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            training_cfg: Training configuration containing DPO parameters (beta, beta_kl)
            collate_fn: Function to collate batches
            eval_fn: Function to evaluate the model
            outcome_dataset: Main outcome dataset
            proxy_dataset: Proxy dataset (preferred responses)
            proxy_neg_dataset: Negative proxy dataset (rejected responses)
            truth_dataset: Truth dataset for evaluation
            collateral_dataset: Collateral dataset for evaluation
            truth_minus_proxy_dataset: Truth minus proxy dataset for evaluation
            exp_folder: Path to save experiment results
            device: Device to use for training
            seed: Random seed for reproducibility
            split_proxy_dataset: Whether to split the proxy dataset into train and test
            split_outcome_dataset: Whether to split the outcome dataset into train and test
        """

        # Set default DPO parameters if not defined in config
        self.beta = getattr(training_cfg, "beta", 0.1)
        self.beta_kl = getattr(training_cfg, "beta_kl", 0.0)
        self.lambda_dpo = getattr(training_cfg, "lambda_dpo", 1.0)
        print("using lambda_dpo:", self.lambda_dpo)

        # Set default cycling mode if not defined in config
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
        logging.info(f"Using DPO beta: {self.beta}")
        logging.info(f"Using DPO beta_kl: {self.beta_kl}")
        logging.info(f"Using lambda_dpo: {self.lambda_dpo}")

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

        # Initialize DPO loss
        self.dpo_loss = DPO(beta=self.beta, beta_kl=self.beta_kl)

        # Create reference model copy for DPO
        if not self.training_cfg.is_peft:
            import copy

            self.reference_model = copy.deepcopy(self.model)
        else:
            self.reference_model = None

    def _prepare_datasets(
        self, split_proxy_dataset: bool = True, split_outcome_dataset: bool = True
    ):
        """
        Prepares training and evaluation datasets/dataloaders.
        Creates a preference dataset from proxy and proxy_neg datasets using Hugging Face datasets.

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

        logging.info(f"[{self.__class__.__name__}] Initializing DPO datasets.")
        if proxy_dataset is None or proxy_neg_dataset is None:
            error_msg = (
                f"[{self.__class__.__name__}] requires both proxy_dataset and proxy_neg_dataset for DPO training. "
                f"Received: proxy_dataset={'is not None' if proxy_dataset else 'is None'}, "
                f"proxy_neg_dataset={'is not None' if proxy_neg_dataset else 'is None'}."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.info(
            f"[{self.__class__.__name__}] DPO datasets loaded successfully. "
            f"len(proxy_dataset)={len(proxy_dataset)}, len(proxy_neg_dataset)={len(proxy_neg_dataset)}"
        )

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
            # type of outcome input ids
            logging.info(
                f"Type of outcome input_ids: {type(outcome_train[0]['input_ids'])}"
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

        # Create preference dataset from proxy and proxy_neg datasets
        if (
            proxy_dataset is not None
            and len(proxy_dataset) > 0
            and proxy_neg_dataset is not None
            and len(proxy_neg_dataset) > 0
        ):
            if len(proxy_dataset) != len(proxy_neg_dataset):
                logging.warning(
                    "Proxy and proxy_neg datasets have different lengths. "
                    "Using the minimum length for preference dataset creation."
                )

            logging.info(
                f"Creating preference dataset from {len(proxy_dataset)} proxy "
                f"and {len(proxy_neg_dataset)} proxy_neg samples"
            )

            # Split proxy datasets first
            proxy_train, proxy_test = self._split_dataset(
                proxy_dataset, split=split_proxy_dataset, seed=self.seed
            )
            proxy_neg_train, proxy_neg_test = self._split_dataset(
                proxy_neg_dataset, split=split_proxy_dataset, seed=self.seed
            )

            # Apply data limiting if configured
            if (
                hasattr(self.training_cfg, "limit_proxy_data_to")
                and self.training_cfg.limit_proxy_data_to
            ):
                logging.info(
                    f"Limiting preference dataset to {self.training_cfg.limit_proxy_data_to} samples"
                )
                proxy_train = proxy_train.select(
                    range(min(self.training_cfg.limit_proxy_data_to, len(proxy_train)))
                )
                proxy_neg_train = proxy_neg_train.select(
                    range(
                        min(self.training_cfg.limit_proxy_data_to, len(proxy_neg_train))
                    )
                )

            # Create preference datasets using Hugging Face datasets
            preference_train_dataset = create_preference_dataset(
                proxy_train, proxy_neg_train
            )
            # sanity checks, print the decoded chosen and rejected sample that is after the prompt (starting from len(prompt_attention_mask))
            if len(preference_train_dataset) > 0:
                chosen_sample = preference_train_dataset[0]["chosen_input_ids"]
                rejected_sample = preference_train_dataset[0]["rejected_input_ids"]
                prompt_attention_mask = preference_train_dataset[0][
                    "prompt_attention_mask"
                ]
                if prompt_attention_mask is not None:
                    chosen_sample = chosen_sample[len(prompt_attention_mask) :]
                    rejected_sample = rejected_sample[len(prompt_attention_mask) :]
                    logging.info(
                        f"Chosen sample after prompt: {self.tokenizer.decode(chosen_sample)}"
                    )
                    logging.info(
                        f"Rejected sample after prompt: {self.tokenizer.decode(rejected_sample)}"
                    )

            # Store datasets
            train_datasets["preference"] = preference_train_dataset
            # print the type of the input ids
            logging.info(
                f"Type of preference chosen input_ids: {type(preference_train_dataset[0]['chosen_input_ids'])}"
            )
            # Create train dataloader for preferences
            train_dataloaders["preference"] = DataLoader(
                preference_train_dataset,
                batch_size=self.training_cfg.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,  # Use the same collate_fn as other datasets
            )

            eval_dataloaders["proxy"] = self.get_eval_dataloader(proxy_test)
            eval_dataloaders["proxy_neg"] = self.get_eval_dataloader(proxy_neg_test)

        # Process evaluation-only datasets (same as PositiveNegativeProxyTrainer)
        if truth_dataset is not None and len(truth_dataset) > 0:
            logging.info(f"Processing truth dataset with {len(truth_dataset)} samples")
            eval_dataloaders["truth"] = self.get_eval_dataloader(truth_dataset)

        if collateral_dataset is not None and len(collateral_dataset) > 0:
            logging.info(
                f"Processing collateral dataset with {len(collateral_dataset)} samples"
            )
            eval_dataloaders["collateral"] = self.get_eval_dataloader(
                collateral_dataset
            )

        if truth_minus_proxy_dataset is not None and len(truth_minus_proxy_dataset) > 0:
            logging.info(
                f"Processing truth_minus_proxy dataset with {len(truth_minus_proxy_dataset)} samples"
            )
            eval_dataloaders["truth_minus_proxy"] = self.get_eval_dataloader(
                truth_minus_proxy_dataset
            )

        # Store dataloaders
        self.train_dataloaders = train_dataloaders

        # Calculate steps per epoch
        outcome_steps = (
            len(train_dataloaders["outcome"]) if "outcome" in train_dataloaders else 0
        )
        preference_steps = (
            len(train_dataloaders["preference"])
            if "preference" in train_dataloaders
            else 0
        )

        logging.info(
            f"Steps per epoch: outcome={outcome_steps}, preference={preference_steps}"
        )

        # Determine steps per epoch based on cycling mode
        if self.proxy_cycling_mode == "continuous":
            self.steps_per_epoch = outcome_steps
            logging.info(
                f"Using outcome dataset size ({self.steps_per_epoch}) as steps per epoch"
            )
        else:  # synchronized mode
            self.steps_per_epoch = max(outcome_steps, preference_steps)
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

    def _process_preference_batch(self, batch, device):
        """
        Helper function to process a preference batch and move it to device.
        Handles the standard collate_fn output format.

        Args:
            batch: Preference data batch from DataLoader
            device: Device to move tensors to

        Returns:
            Batch moved to device with proper tensor stacking
        """
        processed_batch = {}

        # Handle chosen data
        if "chosen_input_ids" in batch:
            processed_batch["chosen_input_ids"] = torch.stack(
                batch["chosen_input_ids"]
            ).to(device)
            processed_batch["chosen_attention_mask"] = torch.stack(
                batch["chosen_attention_mask"]
            ).to(device)

        # Handle rejected data
        if "rejected_input_ids" in batch:
            processed_batch["rejected_input_ids"] = torch.stack(
                batch["rejected_input_ids"]
            ).to(device)
            processed_batch["rejected_attention_mask"] = torch.stack(
                batch["rejected_attention_mask"]
            ).to(device)

        if "prompt_attention_mask" in batch:
            processed_batch["prompt_attention_mask"] = torch.stack(
                batch["prompt_attention_mask"]
            ).to(device)

        return processed_batch

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

        outcome_steps = (
            len(train_dataloaders["outcome"]) if "outcome" in train_dataloaders else 0
        )
        preference_steps = (
            len(train_dataloaders["preference"])
            if "preference" in train_dataloaders
            else 0
        )

        logging.info(
            f"Steps per epoch: outcome={outcome_steps}, preference={preference_steps}"
        )

        # Determine steps per epoch based on cycling mode
        if self.proxy_cycling_mode == "continuous":
            steps_per_epoch = outcome_steps
            logging.info(
                f"Using outcome dataset size ({steps_per_epoch}) as steps per epoch"
            )
        else:  # synchronized mode
            steps_per_epoch = max(outcome_steps, preference_steps)
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
        Trains model with outcome loss + DPO loss.

        Training occurs using one of two modes:
        - "continuous": Continuously cycles preference dataset when exhausted while completing a pass through outcome
        - "synchronized": Only resets all datasets when all are exhausted

        Args:
            save_checkpoint_results_fn: Optional function to save checkpoint results
            save_results_fn: Optional function to save final training results

        Returns:
            Tuple of (model, train_losses, eval_results)
        """
        self.model = self.prepare_for_training()

        # Prepare reference model
        if self.reference_model is not None:
            self.reference_model.to(self.device)
            self.reference_model.eval()
            self.reference_model.requires_grad_(False)

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
        dpo_losses = []

        # Initial evaluation
        logging.info("Running initial evaluations before training")
        eval_results = self.evaluate(
            self.model, self.tokenizer, eval_dataloaders, eval_results=None, epoch=0
        )

        logging.info(f"Initial evaluation results: {eval_results}")

        # Training loop
        for epoch in range(self.training_cfg.epochs):
            logging.info(f"\nEpoch {epoch + 1}/{self.training_cfg.epochs}")
            logging.info("GPU memory at start of epoch:")
            if hasattr(self, "get_gpu_memory_info"):
                self.get_gpu_memory_info()

            self.model.train()
            epoch_total_loss = 0
            epoch_outcome_loss = 0
            epoch_dpo_loss = 0

            # Create iterators for each dataset
            outcome_iter = (
                iter(train_dataloaders["outcome"])
                if "outcome" in train_dataloaders
                else None
            )
            preference_iter = (
                iter(train_dataloaders["preference"])
                if "preference" in train_dataloaders
                else None
            )

            # Track dataset states
            outcome_active = outcome_iter is not None
            preference_active = preference_iter is not None

            # Count for accumulated batches and processed batches
            batch_count = 0
            outcome_batches_processed = 0
            preference_batches_processed = 0
            preference_cycles = 0

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
                    dpo_loss = torch.tensor(0.0, device=self.device)
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

                    # Process preference data if available

                    if preference_active:
                        try:
                            preference_batch = next(preference_iter)
                            preference_batches_processed += 1

                            # Process preference batch
                            preference_batch = self._process_preference_batch(
                                preference_batch, self.device
                            )

                            # Get logprobs for current model
                            chosen_logprobs, rejected_logprobs = (
                                get_chosen_and_rejected_logits(
                                    self.model,
                                    preference_batch,
                                    pad_token=self.tokenizer.pad_token_id,
                                )
                            )

                            # Get logprobs for reference model
                            if self.reference_model is not None:
                                (
                                    reference_chosen_logprobs,
                                    reference_rejected_logprobs,
                                ) = get_chosen_and_rejected_logits(
                                    self.reference_model,
                                    preference_batch,
                                    pad_token=self.tokenizer.pad_token_id,
                                    is_reference=True,
                                )
                            else:
                                # Use PEFT model with adapter disabled as reference
                                with self.model.disable_adapter():
                                    with torch.no_grad():
                                        (
                                            reference_chosen_logprobs,
                                            reference_rejected_logprobs,
                                        ) = get_chosen_and_rejected_logits(
                                            self.model,
                                            preference_batch,
                                            pad_token=self.tokenizer.pad_token_id,
                                            is_reference=True,
                                        )

                            # Compute DPO loss
                            dpo_loss_value = self.dpo_loss(
                                chosen_logprobs,
                                rejected_logprobs,
                                reference_chosen_logprobs,
                                reference_rejected_logprobs,
                            )
                            dpo_loss = (
                                dpo_loss_value
                                / self.training_cfg.gradient_accumulation_steps
                            )
                            batch_losses["dpo"] = dpo_loss_value.item()
                            if first_iter:
                                dpo_losses.append(batch_losses["dpo"])

                            # Clean up tensors
                            self._clean_memory()

                        except StopIteration:
                            preference_active = False

                            # Reset preference iterator immediately in continuous mode
                            if self.proxy_cycling_mode == "continuous":
                                preference_iter = iter(train_dataloaders["preference"])
                                preference_active = True
                                preference_cycles += 1
                                self._clean_memory()
                                logging.info(
                                    f"Preference dataset completed cycle #{preference_cycles}"
                                )
                                preference_batch = next(preference_iter)
                                preference_batches_processed += 1

                                # Process preference batch
                                preference_batch = self._process_preference_batch(
                                    preference_batch, self.device
                                )

                                # Get logprobs for current model
                                chosen_logprobs, rejected_logprobs = (
                                    get_chosen_and_rejected_logits(
                                        self.model,
                                        preference_batch,
                                        pad_token=self.tokenizer.pad_token_id,
                                    )
                                )

                                # Get logprobs for reference model
                                if self.reference_model is not None:
                                    (
                                        reference_chosen_logprobs,
                                        reference_rejected_logprobs,
                                    ) = get_chosen_and_rejected_logits(
                                        self.reference_model,
                                        preference_batch,
                                        pad_token=self.tokenizer.pad_token_id,
                                        is_reference=True,
                                    )
                                else:
                                    # Use PEFT model with adapter disabled as reference
                                    with self.model.disable_adapter():
                                        with torch.no_grad():
                                            (
                                                reference_chosen_logprobs,
                                                reference_rejected_logprobs,
                                            ) = get_chosen_and_rejected_logits(
                                                self.model,
                                                preference_batch,
                                                pad_token=self.tokenizer.pad_token_id,
                                                is_reference=True,
                                            )

                                # Compute DPO loss
                                dpo_loss_value = self.dpo_loss(
                                    chosen_logprobs,
                                    rejected_logprobs,
                                    reference_chosen_logprobs,
                                    reference_rejected_logprobs,
                                )
                                dpo_loss = (
                                    dpo_loss_value
                                    / self.training_cfg.gradient_accumulation_steps
                                )
                                batch_losses["dpo"] = dpo_loss_value.item()
                                if first_iter:
                                    dpo_losses.append(batch_losses["dpo"])

                                # Clean up tensors
                                self._clean_memory()

                            else:
                                logging.info("Preference dataset exhausted")

                    if first_iter:
                        init_outcome_loss = (
                            outcome_losses[0] if len(outcome_losses) > 0 else 0.0
                        )
                        init_dpo_loss = dpo_losses[0] if len(dpo_losses) > 0 else 0.0
                        logging.info("Appending first iteration losses")
                        logging.info(f"Outcome losses: {outcome_losses}")
                        logging.info(f"DPO losses: {dpo_losses}")
                        init_combined_loss = (
                            init_outcome_loss + self.lambda_dpo * init_dpo_loss
                        )
                        train_losses.append(init_combined_loss)
                        logging.info(f"Initial train losses: {train_losses}")

                    logging.info("Setting first_iter to False")
                    first_iter = False

                    # Check stopping conditions based on cycling mode
                    if self.proxy_cycling_mode == "continuous":
                        if not outcome_active:
                            break
                    else:  # synchronized mode
                        if not outcome_active and not preference_active:
                            break
                        if batch_count >= self.steps_per_epoch:
                            break

                    # Only proceed if at least one dataset provided a batch
                    if outcome_active or preference_active:
                        # Combine losses with weighting
                        combined_loss = 0

                        if outcome_active:
                            combined_loss += outcome_loss
                            epoch_outcome_loss += (
                                outcome_loss.item()
                                * self.training_cfg.gradient_accumulation_steps
                            )

                        if preference_active:
                            combined_loss += self.lambda_dpo * dpo_loss
                            epoch_dpo_loss += (
                                dpo_loss.item()
                                * self.training_cfg.gradient_accumulation_steps
                            )

                        # Backward pass
                        logging.info(f"Combined loss: {combined_loss.item()}")
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
                                    f"DPO: {batch_losses.get('dpo', 0):.4f}) - "
                                    f"Preference cycles: {preference_cycles}"
                                )
                            else:
                                logging.info(
                                    f"Batch {batch_count}/{self.steps_per_epoch} "
                                    f"({(batch_count / self.steps_per_epoch) * 100:.1f}%) - "
                                    f"Loss: {curr_loss:.4f} "
                                    f"(Outcome: {batch_losses.get('outcome', 0):.4f}, "
                                    f"DPO: {batch_losses.get('dpo', 0):.4f})"
                                )

                        # In synchronized mode: if all datasets are exhausted but we haven't reached max_steps,
                        # reset the iterators and continue
                        if (
                            self.proxy_cycling_mode == "synchronized"
                            and not outcome_active
                            and not preference_active
                            and batch_count < self.steps_per_epoch
                        ):
                            outcome_iter = (
                                iter(train_dataloaders["outcome"])
                                if "outcome" in train_dataloaders
                                else None
                            )
                            preference_iter = (
                                iter(train_dataloaders["preference"])
                                if "preference" in train_dataloaders
                                else None
                            )
                            outcome_active = outcome_iter is not None
                            preference_active = preference_iter is not None
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
                f"with {preference_cycles} preference dataset cycles ({preference_batches_processed} preference batches)"
            )

            # Compute average losses for the epoch
            if batch_count > 0:
                avg_loss = epoch_total_loss / batch_count
                avg_outcome_loss = epoch_outcome_loss / batch_count
                avg_dpo_loss = epoch_dpo_loss / batch_count
                train_losses.append(avg_loss)
                outcome_losses.append(avg_outcome_loss)
                dpo_losses.append(avg_dpo_loss)
            else:
                logging.warning("No batches processed in this epoch")
                train_losses.append(0)
                outcome_losses.append(0)
                dpo_losses.append(0)

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
                        "outcome": outcome_losses,
                        "dpo": dpo_losses,
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
            logging.info(f"Epoch {epoch + 1}: Average DPO Loss = {avg_dpo_loss:.4f}")
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
                    "outcome": outcome_losses,
                    "dpo": dpo_losses,
                },
                eval_results,
                output_dir=f"{self.exp_folder}/results",
            )

        return (
            self.model,
            {
                "train": train_losses,
                "outcome": outcome_losses,
                "dpo": dpo_losses,
            },
            eval_results,
        )
