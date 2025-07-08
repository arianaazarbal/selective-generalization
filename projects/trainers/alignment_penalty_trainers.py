import torch
import logging
from typing import Optional, Dict, List, Any, Tuple, Union, Callable
from .alignment_plane_trainer import AlignmentPlaneTrainer
import peft
import copy


class AlignmentPenaltyTrainer(AlignmentPlaneTrainer):
    """
    A trainer class that implements orthogonal training functionality.
    This trainer extends the alignment plane trainer to handle orthogonal optimization.
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
            split_proxy_dataset=split_proxy_dataset,
            split_outcome_dataset=split_outcome_dataset,
        )
        self.base_model = copy.deepcopy(self.model)

    def _prepare_outcome_model(self):
        """
        Prepare the orthogonal model and its components.
        This method should be implemented to set up the orthogonal training components.
        """
        import peft

        logging.info("Preparing outcome model")
        # model should not be peft
        if hasattr(self.model, "peft_config"):
            raise ValueError(
                "model should not be a peft model yet, prepare_for_training should not have been called yet on self.model"
            )

        # we need to merge the alignment plane into the model if training_cfg specifies this
        if self.training_cfg.proxy_to_merge is not None:
            logging.info(
                f"Merging proxy gradients for {self.training_cfg.proxy_to_merge}"
            )
            print(
                f"is the model a peft model pre-merge? {hasattr(self.model, 'peft_config')}"
            )
            self.merge_proxy_gradients()
            logging.info("Merged proxy gradients")
            print(
                f"is the model a peft model post-merge? {hasattr(self.model, 'peft_config')}"
            )
        else:
            logging.info("Not merging proxy gradients")
            print(f"is the model a peft model? {hasattr(self.model, 'peft_config')}")
        self.model = super().prepare_for_training(self.model)
        print(
            f"is the model a peft model after preparing for training? {hasattr(self.model, 'peft_config')}"
        )

    def alignment_penalty_train_step(
        self, model, tokenizer, batch: Dict, device: str
    ) -> torch.Tensor:
        """
        Perform a single training step with orthogonal constraints.
        Extends the base class method to incorporate orthogonal training.
        """
        loss = super().train_step(model, tokenizer, batch, device)
        # now calculate the orthogonal loss
        alignment_loss = self.calculate_alignment_loss(model, tokenizer, batch, device)
        logging.info(
            f"Normal loss: {loss}, Alignment loss (scaled by lambda_1 = {self.training_cfg.lambda_1}): {self.training_cfg.lambda_1 * alignment_loss}"
        )
        return loss + self.training_cfg.lambda_1 * alignment_loss

    def calculate_alignment_loss(
        self, model, tokenizer, batch: Dict, device: str
    ) -> torch.Tensor:
        """
        Calculate the orthogonal loss for the given batch.
        """

        alignment_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.alignment_plane:
                if hasattr(model, "peft_config"):
                    assert "lora" in name, (
                        f"lora weights should be named with 'lora' in their name, but {name} is not"
                    )
                    if "lora_A" in name:
                        alignment_loss += self.get_alignment_penalty(
                            param, self.alignment_plane[name]
                        )
                else:
                    assert "lora" not in name, (
                        "base weights should not be named with 'lora' in their name"
                    )
                    assert model is not self.base_model, (
                        "model should not be the base model"
                    )
                    raise NotImplementedError(
                        "Base model alignment penalty not implemented; too much memory usages"
                    )
                    update_direction = (
                        param.data - self.base_model.get_parameter(name).data
                    )
                    alignment_loss += self.get_alignment_penalty(
                        update_direction, self.alignment_plane[name]
                    )

        return alignment_loss

    def train_outcome_model(
        self, save_checkpoint_results_fn=None, save_results_fn=None
    ):
        """
        Train the outcome model.
        """
        from .train import train as train_loop

        # prepares the outcome model, which is self.model
        self._prepare_outcome_model()
        optimizer, lr_scheduler = self.get_standard_optimizer_and_scheduler(
            self.model,
            self.train_dataloaders["outcome"],
            self.training_cfg.outcome_epochs
            if self.training_cfg.outcome_epochs is not None
            else None,
        )

        # then, we can train the outcome model
        results = train_loop(
            (self.model, self.tokenizer),
            self.train_dataloaders["outcome"],
            self.eval_dataloaders,
            self.alignment_penalty_train_step,
            self.evaluate,
            self.training_cfg.epochs
            if self.training_cfg.outcome_epochs is None
            else self.training_cfg.outcome_epochs,
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
        model, train_losses, eval_results = results

        if save_results_fn is not None:
            save_results_fn(
                train_losses, eval_results, output_dir=f"{self.exp_folder}/results"
            )
        return model, train_losses, eval_results

    def train(self, save_checkpoint_results_fn=None, save_results_fn=None):
        """
        Main training loop with orthogonal constraints.
        Extends the base class method to incorporate orthogonal training.
        """
        # first, we must calculate the alignment plane
        self.get_alignment_plane(save_proxy_results_fn=save_results_fn)

        # then, we can train the outcome model
        return self.train_outcome_model(save_checkpoint_results_fn, save_results_fn)


# make OrthogonalTrainer and CosineSimilarityTrainer inherit from AlignmentPenaltyTrainer
# the only method they should implement is the get_alignment_penalty method
# the other methods should be inherited from AlignmentPenaltyTrainer


class OrthogonalPenaltyTrainer(AlignmentPenaltyTrainer):
    def get_alignment_penalty(self, param_1, param_2):
        # Compute |A^T B|, where A and B are the two matrices
        # This measures the similarity between the two matrices
        # We want to minimize this to encourage orthogonality
        # eps = 1e-8
        # param_1_norm = torch.norm(param_1, p=2, dim=1, keepdim=True).clamp(min=eps)
        # param_2_norm = torch.norm(param_2, p=2, dim=1, keepdim=True).clamp(min=eps)

        # param_1 = param_1 / param_1_norm
        # param_2 = param_2 / param_2_norm

        # Compute all pairwise absolute cosine similarities
        abs_cosine_sim = torch.abs(torch.mm(param_1, param_2.T))

        # Return sum as non-orthogonality penalty
        return abs_cosine_sim.sum()


class DirectionalPenaltyTrainer(AlignmentPenaltyTrainer):
    def get_alignment_penalty(self, param_1, param_2):
        # Compute similarity between subspaces without absolute value
        # This measures if subspaces are pointing in similar directions
        # eps = 1e-8
        # param_1_norm = torch.norm(param_1, p=2, dim=1, keepdim=True).clamp(min=eps)
        # param_2_norm = torch.norm(param_2, p=2, dim=1, keepdim=True).clamp(min=eps)

        # param_1 = param_1 / param_1_norm
        # param_2 = param_2 / param_2_norm

        # Compute all pairwise cosine similarities (without absolute value)
        cosine_sim = torch.mm(param_1, param_2.T)

        # Return negative sum as a penalty (to encourage alignment)
        return -cosine_sim.sum()
