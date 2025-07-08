import torch
import logging
from torch.utils.data import DataLoader
from transformers import get_scheduler
from typing import Optional, Dict, List, Any, Tuple, Union, Callable
from .base_trainer import BaseTrainer
from .train_utils import train_test_split, load_steering_vectors, save_steering_vectors
from .train import train as train_loop
from .alignment_plane_trainer import AlignmentPlaneTrainer
from .steering_alpha_scheduler import LinearSteeringAlphaScheduler
import os
import copy


class SteerAtEndTrainer(AlignmentPlaneTrainer):
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
        split_outcome_dataset: bool = False,
        split_proxy_dataset: bool = False,
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
        # if not training_cfg.calc_alignment_plane_using_fully_finetuned_weights:
        #     training_cfg.calc_alignment_plane_using_fully_finetuned_weights = True
        #     logging.info(
        #         "calc_alignment_plane_using_fully_finetuned_weights is set to True"
        #     )
        #     print("calc_alignment_plane_using_fully_finetuned_weights is set to True")
        #     logging.info(
        #         "This is because the trainer is steering at the end of training"
        #     )
        #     print(
        #         "This is because the trainer is steering at the end of training, so we need to calculate the alignment plane using the fully finetuned weights"
        #     )

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
            split_outcome_dataset=split_outcome_dataset,
            split_proxy_dataset=split_proxy_dataset,
        )

    def train(self, save_checkpoint_results_fn=None, save_results_fn=None):
        from .train import train as train_loop
        from torch.optim import AdamW

        steering_weights = self.get_alignment_plane(
            save_proxy_results_fn=save_results_fn,
            save_checkpoint_results_fn=save_checkpoint_results_fn,
        )
        #if not self.training_cfg.calc_alignment_plane_using_fully_finetuned_weights:


        self.model = self.prepare_for_training(self.model)

        # Calculate training steps
        optimizer, lr_scheduler = self.get_standard_optimizer_and_scheduler(
            self.model,
            train_dataloader=self.train_dataloaders["outcome"],
            epochs=self.training_cfg.epochs
            if self.training_cfg.epochs is not None
            else None,
        )
        
        schedulers = [lr_scheduler]

        results = train_loop(
            (self.model, self.tokenizer),
            self.train_dataloaders["outcome"],
            self.eval_dataloaders,
            self.train_step,
            self.evaluate,
            self.training_cfg.epochs
            if self.training_cfg.outcome_epochs is None
            else self.training_cfg.outcome_epochs,
            optimizer,
            schedulers,
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
        model, train_losses, eval_results = results

        #now, add the alignment plane to the model
        add_count = 0
        for name, param in self.model.named_parameters():
            if name in steering_weights:
                param.data.add_(steering_weights[name], alpha=-1 * self.training_cfg.steering_alpha)
                add_count += 1
        print(f"Added {add_count} parameters to the model")
        logging.info(f"Added {add_count} parameters to the model")

        if save_results_fn is not None:
            save_results_fn(
                train_losses, eval_results, output_dir=f"{self.exp_folder}/results"
            )

        return model, train_losses, eval_results
