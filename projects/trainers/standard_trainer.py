import torch
import logging
from torch.utils.data import DataLoader
from transformers import get_scheduler
from typing import Optional, Dict, List, Any, Tuple, Union, Callable
from .base_trainer import BaseTrainer
from .train_utils import train_test_split
from .train import train as train_loop


class StandardTrainer(BaseTrainer):
    """
    A standard trainer class that implements basic training functionality.
    This trainer handles the core training loop, optimization, and evaluation.
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
        **kwargs,
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
            **kwargs,
        )

    def _prepare_datasets(
        self, split_proxy_dataset: bool = True, split_outcome_dataset: bool = True
    ):
        """
        Populates self.train_dataloader and self.eval_dataloaders

        Returns:
            Tuple of (train_dataloader, eval_dataloaders)
        """
        from torch.utils.data import DataLoader
        from datasets import concatenate_datasets
        import logging

        outcome_dataset = self.datasets["outcome"]
        proxy_dataset = self.datasets["proxy"]
        proxy_neg_dataset = self.datasets["proxy_neg"]
        truth_dataset = self.datasets["truth"]
        collateral_dataset = self.datasets["collateral"]
        truth_minus_proxy_dataset = self.datasets["truth_minus_proxy"]

        # Initialize return dictionaries
        train_datasets = {}
        eval_dataloaders = {}

        # Process outcome dataset
        if outcome_dataset is not None and len(outcome_dataset) > 0:
            logging.info(
                f"Processing outcome dataset with {len(outcome_dataset)} samples"
            )

            # Split using the original train_test_split function
            outcome_train, outcome_test = self._split_dataset(
                outcome_dataset, split=split_outcome_dataset, seed=self.seed
            )

            # Store datasets
            train_datasets["outcome"] = outcome_train

            # Create evaluation dataloader
            eval_dataloaders["outcome"] = self.get_eval_dataloader(outcome_test)

        # Process proxy dataset
        if proxy_dataset is not None and len(proxy_dataset) > 0:
            logging.info(f"Processing proxy dataset with {len(proxy_dataset)} samples")

            # Split using the original train_test_split function
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
            logging.info(f"Lenfgth of proxy train dataset: {len(proxy_train)}")

            # Create evaluation dataloader
            eval_dataloaders["proxy"] = self.get_eval_dataloader(proxy_test)
            logging.info(f"Lenfgth of proxy test dataset: {len(proxy_test)}")

        # Process proxy_neg dataset
        if proxy_neg_dataset is not None and len(proxy_neg_dataset) > 0:
            logging.info(
                f"Processing proxy_neg dataset with {len(proxy_neg_dataset)} samples"
            )

            # Split using the original train_test_split function
            proxy_neg_train, proxy_neg_test = self._split_dataset(
                proxy_neg_dataset, split=split_proxy_dataset, seed=self.seed
            )
            if self.training_cfg.limit_proxy_data_to:
                proxy_neg_train = proxy_neg_train.select(
                    range(self.training_cfg.limit_proxy_data_to)
                )
                logging.info(
                    f"Proxy neg train dataset: {proxy_neg_train[: self.training_cfg.limit_proxy_data_to]}"
                )

            # Store datasets
            train_datasets["proxy_neg"] = proxy_neg_train

            # Create evaluation dataloader
            eval_dataloaders["proxy_neg"] = self.get_eval_dataloader(proxy_neg_test)

        # Process truth dataset (no splitting, evaluation only)
        if truth_dataset is not None and len(truth_dataset) > 0:
            logging.info(f"Processing truth dataset with {len(truth_dataset)} samples")

            # Log percentage of triggered samples
            # Create evaluation dataloader
            eval_dataloaders["truth"] = self.get_eval_dataloader(truth_dataset)

        # Process collateral dataset (no splitting, evaluation only)
        if collateral_dataset is not None and len(collateral_dataset) > 0:
            logging.info(
                f"Processing collateral dataset with {len(collateral_dataset)} samples"
            )

            # Create evaluation dataloader
            eval_dataloaders["collateral"] = self.get_eval_dataloader(
                collateral_dataset
            )

        # Process truth_minus_proxy dataset (no splitting, evaluation only)
        if truth_minus_proxy_dataset is not None and len(truth_minus_proxy_dataset) > 0:
            logging.info(
                f"Processing truth_minus_proxy dataset with {len(truth_minus_proxy_dataset)} samples"
            )

            # Create evaluation dataloader
            eval_dataloaders["truth_minus_proxy"] = self.get_eval_dataloader(
                truth_minus_proxy_dataset
            )
        # Create combined training dataset if we have both outcome and proxy datasets
        if "outcome" in train_datasets and "proxy" in train_datasets:
            logging.info("Combining outcome and proxy datasets")
            combined_train = concatenate_datasets(
                [train_datasets["outcome"], train_datasets["proxy"]]
            )
        elif "outcome" in train_datasets:
            logging.info("Only outcome dataset found")
            combined_train = train_datasets["outcome"]
        elif "proxy" in train_datasets:
            logging.info("Only proxy dataset found")
            combined_train = train_datasets["proxy"]
        else:
            raise ValueError("No outcome dataset found")

        logging.info(
            f"Created combined train dataset with {len(combined_train)} samples"
        )

        # Store train datasets for future reference
        self.train_dataset = combined_train

        # Create single training dataloader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_cfg.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        # Store references for use by the trainer
        self.train_dataloader = train_dataloader
        self.eval_dataloaders = eval_dataloaders

        # print the keys of the first batch of the train_dataloader
        print(train_dataloader.dataset[0].keys())
        # print the keys of the first batch of the eval_dataloaders
        print(eval_dataloaders["outcome"].dataset[0].keys())
        if self.training_cfg.save_datasets:
            # Save the train and eval datasets to disk
            self.save_datasets()
        return train_dataloader, eval_dataloaders

    def train(self, save_checkpoint_results_fn=None, save_results_fn=None):
        """
        Train the model using the standard training loop.

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
        from .train import train as train_loop
        from torch.optim import AdamW

        self.model = self.prepare_for_training()
        # print whether peft model is used
        print(f"PEFT model desired: {self.training_cfg.is_peft}")
        print(f"Model type: {type(self.model)}")
        print(f"Has adapters: {hasattr(self.model, 'peft_config')}")
        # log which parameters are trainable

        optimizer, lr_scheduler = self.get_standard_optimizer_and_scheduler(self.model)

        model, train_losses, eval_results = train_loop(
            (self.model, self.tokenizer),
            self.train_dataloader,
            self.eval_dataloaders,
            self.train_step,
            self.evaluate,
            self.training_cfg.epochs,
            optimizer,
            [lr_scheduler],
            self.exp_folder,
            save_checkpoint_results_fn,
            logging_steps=self.training_cfg.logging_steps,
            collect_gradients=False,
            push_model_fn=self.push_model if self.training_cfg.push_to_hub else None,
            save_model_locally_fn=self.save_model_locally
            if self.training_cfg.save_model_locally
            else None,
            max_grad_norm=self.training_cfg.max_grad_norm
            if hasattr(self.training_cfg, "max_grad_norm")
            else None,
        )

        if save_results_fn is not None:
            save_results_fn(
                train_losses, eval_results, output_dir=f"{self.exp_folder}/results"
            )

        return model, train_losses, eval_results
