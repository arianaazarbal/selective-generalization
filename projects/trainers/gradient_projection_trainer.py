import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_scheduler
from typing import Optional, Dict, List, Any, Tuple, Union, Callable
from tqdm import tqdm
import gc
from .base_trainer import BaseTrainer
from .train_utils import train_test_split
from .alignment_plane_trainer import AlignmentPlaneTrainer
from .positive_negative_proxy_trainer import PositiveNegativeProxyTrainer


def calculate_corrected_gradient_norm(corrected_grads):
    """
    Calculate the norm of the corrected gradients
    """
    corrected_norms = []
    for name, grad in corrected_grads.items():
        norm_squared = torch.sum(grad * grad)
        corrected_norm = torch.sqrt(norm_squared)
        corrected_norms.append(corrected_norm)
    return corrected_norms


class GradientProjectionTrainer(PositiveNegativeProxyTrainer):
    def project_single_gradient(self, task_grad, align_grad):
        assert task_grad.data.shape == align_grad.data.shape, (
            "Shapes of task and align data do not match"
        )
        dot_product = torch.sum(task_grad * align_grad)
        norm_squared_align = torch.sum(align_grad * align_grad)
        norm_squared_task = torch.sum(task_grad * task_grad)

        # Calculate gradient norms
        task_norm = torch.sqrt(norm_squared_task)
        align_norm = torch.sqrt(norm_squared_align)

        # Store gradient norms for this batch
        # epoch_grad_f_norms.append(grad_f_norm.item())
        # epoch_grad_g_norms.append(grad_g_norm.item())

        dynamic_lambda = dot_product / (
            norm_squared_align + 1e-16
        )  # Avoid division by zero

        # Compute cosine similarity: cos(θ) = dot_product / (||grad_f|| * ||grad_g||)
        cosine_sim = dot_product / (task_norm * align_norm + 1e-16)

        # Store unclamped lambda and cosine similarity
        # epoch_lambdas.append(dynamic_lambda.item())
        # epoch_cosine_sims.append(cosine_sim.item())

        # Now clamp for the actual algorithm
        if not self.training_cfg.project_strictly_orthogonal:
            dynamic_lambda = torch.clamp(dynamic_lambda, max=0.0)

        logging.info(
            f"dynamic_lambda: {dynamic_lambda}, we are correcting the task gradient"
        )
        corrected_grad = task_grad - dynamic_lambda * align_grad
        if self.training_cfg.project_strictly_orthogonal and not torch.allclose(
            torch.sum(corrected_grad * align_grad),
            torch.tensor(0.0),
            atol=1e-3,
            rtol=1e-3,
        ):
            logging.info(
                f"Projection failed: og dot product: {dot_product}, new dot product is {torch.sum(corrected_grad * align_grad)}, not close to zero"
            )

        return corrected_grad, task_norm, align_norm, dynamic_lambda, cosine_sim

    def get_projected_gradients(self, task_grads, align_grads):
        """
        Will return the corrected task gradients
        """
        corrected_task_grads = {}
        task_norms = []
        align_norms = []
        dynamic_lambdas = []
        cosine_sims = []
        for name in task_grads.keys():
            if name not in align_grads:
                logging.warning(
                    f"{name} has task gradients but not alignment gradients"
                )
                corrected_task_grads[name] = task_grads[name]
                continue
            corrected_grad, task_norm, align_norm, dynamic_lambda, cosine_sim = (
                self.project_single_gradient(task_grads[name], align_grads[name])
            )
            task_norms.append(task_norm)
            align_norms.append(align_norm)
            dynamic_lambdas.append(dynamic_lambda)
            cosine_sims.append(cosine_sim)
            corrected_task_grads[name] = corrected_grad
        return (
            corrected_task_grads,
            task_norms,
            align_norms,
            dynamic_lambdas,
            cosine_sims,
        )

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
        if self.training_cfg.gradient_accumulation_steps != 1:
            raise NotImplementedError(
                "Only supports gradient accumulation steps of 1 at this moment"
            )
        logging.info(
            f"Project strictly orthogonal: {self.training_cfg.project_strictly_orthogonal if hasattr(self.training_cfg, 'project_strictly_orthogonal') else 'Not Specified; defaulting to true'}"
        )
        if not hasattr(self.training_cfg, "project_strictly_orthogonal"):
            self.training_cfg.project_strictly_orthogonal = True

        self.model = self.prepare_for_training(self.model)
        if self.training_cfg.project_along_positive_proxy_grad:
            logging.info(
                "Projecting along positive proxy gradient; only training on proxy"
            )
            del self.train_dataloaders["proxy_neg"]
        else:
            logging.info(
                "Projecting along negative proxy gradient; only training on proxy_neg"
            )
            del self.train_dataloaders["proxy"]

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

            # Initialize lists to track gradient metrics for this epoch
            epoch_task_norms = []
            epoch_align_norms = []
            epoch_corrected_task_norms = []
            epoch_dynamic_lambdas = []
            epoch_cosine_sims = []

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
                    proxy_loss = torch.tensor(0.0, device=self.device)
                    proxy_neg_loss = torch.tensor(0.0, device=self.device)
                    batch_losses = {}
                    batch_grads = {}

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
                            epoch_outcome_loss += batch_losses["outcome"]
                            if first_iter:
                                outcome_losses.append(batch_losses["outcome"])

                            outcome_loss.backward()
                            batch_grads["outcome"] = {
                                name: param.grad.clone()
                                for name, param in self.model.named_parameters()
                                if param.grad is not None
                            }
                            optimizer.zero_grad()

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
                            epoch_proxy_loss += batch_losses["proxy"]
                            if first_iter:
                                proxy_losses.append(batch_losses["proxy"])
                            proxy_loss.backward()
                            batch_grads["proxy"] = {
                                name: param.grad.clone()
                                for name, param in self.model.named_parameters()
                                if param.grad is not None
                            }
                            optimizer.zero_grad()

                            # Apply KL regularization if configured (only to proxy data)

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
                                epoch_proxy_loss += batch_losses["proxy"]
                                proxy_loss.backward()
                                batch_grads["proxy"] = {
                                    name: param.grad.clone()
                                    for name, param in self.model.named_parameters()
                                    if param.grad is not None
                                }
                                optimizer.zero_grad()

                                # Apply KL regularization if configured (only to proxy data)

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
                            epoch_proxy_neg_loss += batch_losses["proxy_neg"]
                            if first_iter:
                                proxy_neg_losses.append(batch_losses["proxy_neg"])
                            proxy_neg_loss.backward()
                            batch_grads["proxy_neg"] = {
                                name: -1 * param.grad.clone()
                                for name, param in self.model.named_parameters()
                                if param.grad is not None
                            }

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
                                epoch_proxy_neg_loss += batch_losses["proxy_neg"]
                                proxy_neg_loss.backward()
                                batch_grads["proxy_neg"] = {
                                    name: -1 * param.grad.clone()
                                    for name, param in self.model.named_parameters()
                                    if param.grad is not None
                                }

                                # Clean up tensors
                                del proxy_neg_outputs, input_ids, attention_mask, labels
                                self._clean_memory()
                            else:
                                logging.info("Negative proxy dataset exhausted")

                    if first_iter:
                        train_losses.append(
                            epoch_outcome_loss + epoch_proxy_loss - epoch_proxy_neg_loss
                        )
                    first_iter = False
                    logging.info(f"first_iter: {first_iter}")
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
                        corrected_gradients = (
                            batch_grads["outcome"] if outcome_active else None
                        )

                        # Initialize as empty lists and only fill if projection occurs
                        batch_task_norms = []
                        batch_align_norms = []
                        batch_dynamic_lambdas = []
                        batch_cosine_sims = []

                        if outcome_active and proxy_active:
                            logging.info(
                                "Correcting outcome gradients using proxy projection"
                            )
                            (
                                corrected_gradients,
                                batch_task_norms,
                                batch_align_norms,
                                batch_dynamic_lambdas,
                                batch_cosine_sims,
                            ) = self.get_projected_gradients(
                                batch_grads["outcome"], batch_grads["proxy"]
                            )

                            # Calculate corrected gradient norms
                            batch_corrected_task_norms = (
                                calculate_corrected_gradient_norm(corrected_gradients)
                            )

                            # Store metrics for this batch
                            epoch_task_norms.extend(
                                [norm.item() for norm in batch_task_norms]
                            )
                            epoch_align_norms.extend(
                                [norm.item() for norm in batch_align_norms]
                            )
                            epoch_corrected_task_norms.extend(
                                [norm.item() for norm in batch_corrected_task_norms]
                            )
                            epoch_dynamic_lambdas.extend(
                                [lmbda.item() for lmbda in batch_dynamic_lambdas]
                            )
                            epoch_cosine_sims.extend(
                                [sim.item() for sim in batch_cosine_sims]
                            )

                        if outcome_active and proxy_neg_active:
                            logging.info(
                                "Correcting outcome gradients using proxy neg projection"
                            )
                            (
                                corrected_gradients,
                                batch_task_norms,
                                batch_align_norms,
                                batch_dynamic_lambdas,
                                batch_cosine_sims,
                            ) = self.get_projected_gradients(
                                batch_grads["outcome"], batch_grads["proxy_neg"]
                            )

                            # Calculate corrected gradient norms
                            batch_corrected_task_norms = (
                                calculate_corrected_gradient_norm(corrected_gradients)
                            )

                            # Store metrics for this batch
                            epoch_task_norms.extend(
                                [norm.item() for norm in batch_task_norms]
                            )
                            epoch_align_norms.extend(
                                [norm.item() for norm in batch_align_norms]
                            )
                            epoch_corrected_task_norms.extend(
                                [norm.item() for norm in batch_corrected_task_norms]
                            )
                            epoch_dynamic_lambdas.extend(
                                [lmbda.item() for lmbda in batch_dynamic_lambdas]
                            )
                            epoch_cosine_sims.extend(
                                [sim.item() for sim in batch_cosine_sims]
                            )

                        # Apply corrected gradients to the model parameters
                        if corrected_gradients:
                            for name, param in self.model.named_parameters():
                                param.grad = (
                                    corrected_gradients[name]
                                    if name in corrected_gradients
                                    else None
                                )

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
                                    f"(Outcome: {batch_losses.get('outcome', 0):.4f}, "
                                    f"Proxy: {batch_losses.get('proxy', 0):.4f}, "
                                    f"Neg Proxy: {batch_losses.get('proxy_neg', 0):.4f}, "
                                    f"Proxy cycles: {proxy_cycles}"
                                )
                            else:
                                logging.info(
                                    f"Batch {batch_count}/{self.steps_per_epoch} "
                                    f"({(batch_count / self.steps_per_epoch) * 100:.1f}%) - "
                                    f"(Outcome: {batch_losses.get('outcome', 0):.4f}, "
                                    f"Proxy: {batch_losses.get('proxy', 0):.4f}, "
                                    f"Neg Proxy: {batch_losses.get('proxy_neg', 0):.4f})"
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

            # Calculate and log statistics for gradient metrics
            if len(epoch_task_norms) > 0:
                task_norm_mean = np.mean(epoch_task_norms)
                task_norm_std = np.std(epoch_task_norms)
                align_norm_mean = np.mean(epoch_align_norms)
                align_norm_std = np.std(epoch_align_norms)
                corrected_task_norm_mean = np.mean(epoch_corrected_task_norms)
                corrected_task_norm_std = np.std(epoch_corrected_task_norms)
                dynamic_lambda_mean = np.mean(epoch_dynamic_lambdas)
                dynamic_lambda_std = np.std(epoch_dynamic_lambdas)
                cosine_sim_mean = np.mean(epoch_cosine_sims)
                cosine_sim_std = np.std(epoch_cosine_sims)

                logging.info(f"Epoch {epoch + 1} Gradient Statistics:")
                logging.info(
                    f"  Task Norm Mean: {task_norm_mean:.4f}, Std: {task_norm_std:.4f}"
                )
                logging.info(
                    f"  Align Norm Mean: {align_norm_mean:.4f}, Std: {align_norm_std:.4f}"
                )
                logging.info(
                    f"  Corrected Task Norm Mean: {corrected_task_norm_mean:.4f}, Std: {corrected_task_norm_std:.4f}"
                )
                logging.info(
                    f"  Dynamic Lambda Mean: {dynamic_lambda_mean:.4f}, Std: {dynamic_lambda_std:.4f}"
                )
                logging.info(
                    f"  Cosine Similarity Mean: {cosine_sim_mean:.4f}, Std: {cosine_sim_std:.4f}"
                )
            else:
                logging.info("No gradient projections were performed during this epoch")

            # Log cycle information
            logging.info(
                f"Epoch {epoch + 1} complete: Processed {outcome_batches_processed} outcome batches "
                f"with {proxy_cycles} proxy dataset cycles ({proxy_batches_processed} proxy batches) "
                f"and {proxy_neg_cycles} negative proxy cycles ({proxy_neg_batches_processed} neg proxy batches)"
            )

            # Compute average losses for the epoch
            if batch_count > 0:
                avg_outcome_loss = epoch_outcome_loss / batch_count
                avg_proxy_loss = epoch_proxy_loss / batch_count
                avg_proxy_neg_loss = epoch_proxy_neg_loss / batch_count
                avg_loss = avg_outcome_loss + avg_proxy_loss - avg_proxy_neg_loss
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


class GradientProjectionPrecomputedProxyGradTrainer(AlignmentPlaneTrainer):
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
        )

    def get_avg_proxy_gradient(self, save_proxy_results_fn, save_checkpoint_results_fn):
        if (
            hasattr(self.training_cfg, "project_along_positive_proxy_grad")
            and self.training_cfg.project_along_positive_proxy_grad is False
        ):
            # in this case, we will get -1 * avg raw proxy negative gradient
            self.training_cfg.positive_proxy = None
            if self.training_cfg.negative_proxy is None:
                raise ValueError(
                    "Must specify a negative proxy dataset if project_along_positive_proxy_grad is False"
                )
            print("MADE IT HERE")
            print(self.training_cfg.negative_proxy)
            print(self.training_cfg.positive_proxy)
            # print train dataloaders/dataset keys
            if hasattr(self, "train_dataloaders"):
                print(self.train_dataloaders.keys())
            if hasattr(self, "train_datasets"):
                print(self.train_datasets.keys())
            self.training_cfg.calc_alignment_plane_using_fully_finetuned_weights = False  # we want the avg raw gradient which is what we'll get when this is False

            self.training_cfg.proxy_epochs = 1

        else:
            self.training_cfg.negative_proxy = None
            self.training_cfg.calc_alignment_plane_using_fully_finetuned_weights = False  # we want the avg raw gradient which is what we'll get when this is False
            assert (
                self.training_cfg.positive_proxy == "proxy"
            )  # this is not meaningful if not

            # for now, just train for one epoch
            self.training_cfg.proxy_epochs = 1

        # log our positive and negative proxy
        logging.info(
            f"Projecting along {'positive' if self.training_cfg.project_along_positive_proxy_grad else 'negative'} proxy gradient"
        )
        logging.info(
            f"Using {'positive' if self.training_cfg.positive_proxy else 'negative'} proxy dataset"
        )
        avg_raw_proxy_grad = self.get_alignment_plane(
            save_checkpoint_results_fn=save_checkpoint_results_fn,
            save_proxy_results_fn=save_proxy_results_fn,
        )
        self.avg_proxy_grad = avg_raw_proxy_grad
        return avg_raw_proxy_grad

    def train(self, save_checkpoint_results_fn=None, save_results_fn=None):
        from .train import train as train_loop
        from torch.optim import AdamW

        self.get_avg_proxy_gradient(
            save_proxy_results_fn=save_results_fn,
            save_checkpoint_results_fn=save_checkpoint_results_fn,
        )

        self.model = self.prepare_for_training(self.model)

        from .gradient_projection_optimizer import GradProjectAdamW

        gp_optimizer = GradProjectAdamW(
            self.model.parameters(),
            named_parameters=list(self.model.named_parameters()),
            lr=self.training_cfg.learning_rate,
            proxy_grads=self.avg_proxy_grad,
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
            optimizer=gp_optimizer,
            num_warmup_steps=self.training_cfg.warmup_steps,
            num_training_steps=num_training_steps,
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
            gp_optimizer,
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

        if save_results_fn is not None:
            save_results_fn(
                train_losses, eval_results, output_dir=f"{self.exp_folder}/results"
            )

        return model, train_losses, eval_results

    # def project_along_precomputed_proxy_grad_train_step(
    #     self, model, tokenizer, batch: Dict, device: str
    # ) -> torch.Tensor:
    #     """
    #     Perform a single training step, where we modify the task gradients so that we 'clip out' the portion that is going
    #     'against' the proxy gradients (precmputed, stored as self.avg_proxy_grad )
    #     """
    #     loss = super().train_step(model, tokenizer, batch, device)

    # def train(self, save_checkpoint_results_fn=None, save_results_fn=None):
    #     from .train import train as train_loop

    #     # prepares the outcome model, which is self.model
    #     self.prepare_for_training()
    #     optimizer, lr_scheduler = self.get_standard_optimizer_and_scheduler(
    #         self.model,
    #         self.train_dataloaders["outcome"],
    #         self.training_cfg.outcome_epochs
    #         if self.training_cfg.outcome_epochs is not None
    #         else None,
    #     )

    #     # then, we can train the outcome model
    #     results = train_loop(
    #         (self.model, self.tokenizer),
    #         self.train_dataloaders["outcome"],
    #         self.eval_dataloaders,
    #         self.project_along_precomputed_proxy_grad_train_step,
    #         self.evaluate,
    #         self.training_cfg.epochs
    #         if self.training_cfg.outcome_epochs is None
    #         else self.training_cfg.outcome_epochs,
    #         optimizer,
    #         [lr_scheduler],
    #         self.exp_folder,
    #         save_checkpoint_results_fn,
    #         logging_steps=self.training_cfg.logging_steps,
    #         collect_gradients=False,
    #         push_model_fn=self.push_model if self.training_cfg.push_to_hub else None,
    #         save_model_locally_fn=self.save_model_locally
    #         if self.training_cfg.save_checkpoints_locally
    #         else None,
    #         max_grad_norm=self.training_cfg.max_grad_norm
    #         if hasattr(self.training_cfg, "max_grad_norm")
    #         else None,
    #     )
    #     model, train_losses, eval_results = results

    #     if save_results_fn is not None:
    #         save_results_fn(
    #             train_losses, eval_results, output_dir=f"{self.exp_folder}/results"
    #         )
    #     return model, train_losses, eval_results
