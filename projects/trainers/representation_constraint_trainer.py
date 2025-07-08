import torch
import copy
import logging
from torch.utils.data import DataLoader
from transformers import get_scheduler
from typing import Optional, Dict, List, Any, Tuple, Union, Callable
from tqdm import tqdm
import gc
from .base_trainer import BaseTrainer
from .positive_negative_proxy_trainer import PositiveNegativeProxyTrainer

class RepresentationConstraintTrainer(PositiveNegativeProxyTrainer):
    """
    A trainer that implements training using KL divergence regularization.

    Unlike PositiveNegativeProxyTrainer, this trainer computes loss as:
    outcome_loss + representation_constraint_loss

    It does not use proxy_outputs.loss or proxy_neg_outputs.loss from the model.
    Instead, the regularization comes entirely from the KL divergence between
    the LoRA model and the base model.
    """
    def compute_representation_constraint(self, base_hidden_states, ft_hidden_states):
        """
        Computes representation constraint between base and LoRA model hidden states
        """
        mse_loss = torch.tensor(0.0, device=self.device)
        for layer in range(1, len(base_hidden_states)):
            mse_loss += torch.nn.functional.mse_loss(base_hidden_states[layer], ft_hidden_states[layer])
        
        return mse_loss

    def apply_representation_constraint(self, input_ids, attention_mask, batch_losses):
        """
        Applies representation constraint to model outputs

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            batch_losses: Dictionary to track losses

        Returns:
            Representation constraint loss to be added to total loss
        """
        rep_loss = torch.tensor(0.0, device=self.device)
        raw_rep_loss = 0.0
        if hasattr(self.training_cfg, "beta_kl") and self.training_cfg.beta_kl > 0:
            try:
                # Get LoRA model logits without labels for KL calculation
                ft_outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True
                )
                ft_hidden_states = ft_outputs.hidden_states
                del ft_outputs

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
                                output_hidden_states=True,
                            )
                            base_hidden_states = base_outputs.hidden_states
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
                            output_hidden_states=True,
                        )
                        base_hidden_states = base_outputs.hidden_states

                    if original_mode_was_training:
                        self.base_model_copy.train()
                del base_outputs
                # Compute KL divergence
                rep_loss = self.compute_representation_constraint(base_hidden_states, ft_hidden_states)
                del base_hidden_states, ft_hidden_states

                raw_rep_loss = rep_loss.item()
                batch_losses["rep"] = raw_rep_loss
                rep_loss = self.training_cfg.beta_kl * rep_loss / self.training_cfg.gradient_accumulation_steps
                # Clean up tensors
                

            except AttributeError as e:
                logging.warning(
                    f"Representation Constraint failed: {e}. Continuing without."
                )

        return rep_loss, raw_rep_loss

    def train(self, save_checkpoint_results_fn=None, save_results_fn=None):
        """
        Trains model using outcome dataset and KL divergence regularization.

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
        train_dataloaders, eval_dataloaders = self._prepare_datasets()
        optimizer, lr_scheduler = self.get_standard_optimizer_and_scheduler(
            self.model, train_dataloaders
        )

        # Initialize training tracking
        train_losses = []
        outcome_losses = []
        representation_constraint_losses = []

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
            epoch_representation_constraint_loss = 0

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

            first_iter = True if epoch == 0 else False
            # Continue until stopping condition based on cycling mode
            with tqdm(total=self.steps_per_epoch, desc=progress_desc) as pbar:
                while True:
                    outcome_loss = torch.tensor(0.0, device=self.device)
                    representation_constraint_loss = torch.tensor(0.0, device=self.device)
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

                            # Apply KL regularization directly on outcome data

                            # Clean up tensors
                            del outcome_outputs
                            self._clean_memory()

                        except StopIteration:
                            outcome_active = False
                            logging.info("Outcome dataset exhausted")

                    # Process proxy data if available - for KL regularization only
                    if proxy_active:
                        try:
                            proxy_batch = next(proxy_iter)
                            proxy_batches_processed += 1

                            # Process proxy batch
                            input_ids, attention_mask, labels = self._process_batch(
                                proxy_batch, self.device
                            )

                            # We don't use proxy_outputs.loss, only KL regularization
                            representation_constraint_loss, raw_rep_loss = self.apply_representation_constraint(
                                input_ids, attention_mask, batch_losses
                            )

                            # Add KL loss (not scaled by lambda_proxy)
                            if first_iter:
                                representation_constraint_losses.append(batch_losses["rep"])

                            # Clean up tensors
                            del input_ids, attention_mask, labels
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

                                # We don't use proxy_outputs.loss, only KL regularization
                                representation_constraint_loss, raw_rep_loss = self.apply_representation_constraint(
                                    input_ids, attention_mask, batch_losses
                                )
                                # Add KL loss (not scaled by lambda_proxy)
                                if first_iter:
                                    representation_constraint_losses.append(batch_losses["rep"])

                                # Clean up tensors
                                del input_ids, attention_mask, labels
                                self._clean_memory()
                            else:
                                logging.info("Proxy dataset exhausted")

                    # We don't use proxy_neg_dataset but still need to exhaust it
                    # to properly track cycles in the original implementation
                    if proxy_neg_active:
                        try:
                            proxy_neg_batch = next(proxy_neg_iter)
                            proxy_neg_batches_processed += 1

                            # We don't use proxy_neg_batch for anything, just track it

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
                            else:
                                logging.info("Negative proxy dataset exhausted")
                    if first_iter:
                        init_loss = 0.0
                        if len(outcome_losses) > 0:
                            init_loss += outcome_losses[0]
                        if len(representation_constraint_losses) > 0:
                            init_loss += representation_constraint_losses[0]
                        train_losses.append(init_loss)

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
                    if outcome_active or proxy_active:
                        # Combine losses - now just outcome_loss + representation_constraint_loss
                        combined_loss = 0.0
                        # log outcome and rep loss
                        logging.info(
                            f"Scaled outcome loss: {outcome_loss}, scaled representation loss: {representation_constraint_loss}"
                        )
                        logging.info(f"Unscaled outcome loss: {batch_losses['outcome']}, unscaled representation loss: {batch_losses['rep']}")

                        if outcome_active:
                            combined_loss += outcome_loss
                            epoch_outcome_loss += (
                                outcome_loss.item()
                                * self.training_cfg.gradient_accumulation_steps
                            )

                        # Add KL loss directly (not scaled by lambda_proxy)
                        combined_loss += representation_constraint_loss
                        epoch_representation_constraint_loss += (
                            representation_constraint_loss.item()
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
                                    f"Representation: {batch_losses.get('rep', 0):.4f}) - "
                                    f"Proxy cycles: {proxy_cycles}"
                                )
                            else:
                                logging.info(
                                    f"Batch {batch_count}/{self.steps_per_epoch} "
                                    f"({(batch_count / self.steps_per_epoch) * 100:.1f}%) - "
                                    f"Loss: {curr_loss:.4f} "
                                    f"(Outcome: {batch_losses.get('outcome', 0):.4f}, "
                                    f"Representation: {batch_losses.get('rep', 0):.4f})"
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
                avg_representation_constraint_loss = epoch_representation_constraint_loss / batch_count
                train_losses.append(avg_loss)
                outcome_losses.append(avg_outcome_loss)
                representation_constraint_losses.append(avg_representation_constraint_loss)
            else:
                logging.warning("No batches processed in this epoch")
                train_losses.append(0)
                outcome_losses.append(0)
                representation_constraint_losses.append(0)

            # Evaluate model
            eval_results = self.evaluate(
                self.model,
                self.tokenizer,
                eval_dataloaders,
                eval_results,
                epoch=epoch + 1,
                is_final_epoch=(epoch == self.training_cfg.epochs - 1),
            )

            # Save checkpoint
            if save_checkpoint_results_fn:
                save_checkpoint_results_fn(
                    self.model,
                    train_losses,
                    eval_results,
                    output_dir=f"{self.exp_folder}/checkpoints/epoch_{epoch}",
                    epoch=epoch,
                )

            # Log epoch results
            logging.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")
            logging.info(
                f"Epoch {epoch + 1}: Average Outcome Loss = {avg_outcome_loss:.4f}"
            )
            logging.info(f"Epoch {epoch + 1}: Average KL Loss = {avg_representation_constraint_loss:.4f}")
            logging.info(f"Epoch {epoch + 1}: Evaluation Results = {eval_results}")
            logging.info("GPU memory at end of epoch:")
            if hasattr(self, "get_gpu_memory_info"):
                self.get_gpu_memory_info()

        if self.training_cfg.save_model_locally:
            self.save_model_locally(self.model, self.tokenizer)

        # Push model to hub if configured
        if self.training_cfg.push_to_hub:
            self.push_model(self.model, self.tokenizer)

        # Save final results
        if save_results_fn is not None:
            save_results_fn(
                train_losses, eval_results, output_dir=f"{self.exp_folder}/results"
            )

        return self.model, train_losses, eval_results
