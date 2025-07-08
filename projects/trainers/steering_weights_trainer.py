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

def get_gpu_memory_info():
    import json
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available")
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
    logging.info("GPU Memory Info: %s", json.dumps(memory_info, indent=2))
    print(f"GPU Memory Info: {json.dumps(memory_info, indent=2)}")
    return None
class SteeringWeightsTrainer(AlignmentPlaneTrainer):
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
        logging.info("Getting GPU memory info pre-alignment plane gathering")
        get_gpu_memory_info()

        steering_weights = self.get_alignment_plane(
            save_proxy_results_fn=save_results_fn,
            save_checkpoint_results_fn=save_checkpoint_results_fn,
        )

        logging.info("Getting GPU memory info post-alignment plane gathering")
        get_gpu_memory_info()
        if self.proxy_model:
            del self.proxy_model
        if self.proxy_minus_model:
            del self.proxy_minus_model
        self.proxy_model = None
        self.proxy_minus_model = None
        #clean memory
        self._clean_memory()
        logging.info("Getting GPU memory info post-memory cleanup")
        get_gpu_memory_info()

        self.model = self.prepare_for_training(self.model)

        if not self.training_cfg.direct_steering:
            from .gradient_steering_optimizer import SteeringVectorAdamW

            gs_optimizer = SteeringVectorAdamW(
                self.model.parameters(),
                named_parameters=list(self.model.named_parameters()),
                lr=self.training_cfg.learning_rate,
                steering_vectors=steering_weights,
                alpha=self.training_cfg.steering_alpha,
            )
        else:
            from .gradient_steering_optimizer import DirectSteeringVectorAdamW

            gs_optimizer = DirectSteeringVectorAdamW(
                self.model.parameters(),
                named_parameters=list(self.model.named_parameters()),
                lr=self.training_cfg.learning_rate,
                steering_vectors=steering_weights,
                alpha=self.training_cfg.steering_alpha,
            )

        # Calculate training steps
        num_update_steps_per_epoch = (
            len(self.train_dataloaders["outcome"])
            // self.training_cfg.gradient_accumulation_steps
        )
        num_training_steps = num_update_steps_per_epoch * (
            self.training_cfg.epochs
            if self.training_cfg.outcome_epochs is None
            else self.training_cfg.outcome_epochs
        )

        lr_scheduler = get_scheduler(
            name=self.training_cfg.lr_scheduler_type,
            optimizer=gs_optimizer,
            num_warmup_steps=self.training_cfg.warmup_steps,
            num_training_steps=num_training_steps,
        )
        if self.training_cfg.alpha_scheduler_type:
            from .steering_alpha_scheduler import LinearSteeringAlphaScheduler

            if self.training_cfg.alpha_scheduler_type != "linear":
                raise ValueError(
                    f"Unsupported alpha scheduler type: {self.training_cfg.alpha_scheduler_type}"
                )

            alpha_scheduler = LinearSteeringAlphaScheduler(
                gs_optimizer,
                initial_alpha=self.training_cfg.steering_alpha,
                final_alpha=self.training_cfg.final_steering_alpha,
                num_warmup_steps=self.training_cfg.alpha_warmup_steps,
                num_training_steps=num_training_steps,
            )
            schedulers = [alpha_scheduler, lr_scheduler]
        else:
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
            gs_optimizer,
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
        del gs_optimizer
        if save_results_fn is not None:
            save_results_fn(
                train_losses, eval_results, output_dir=f"{self.exp_folder}/results"
            )
        
        return model, train_losses, eval_results
