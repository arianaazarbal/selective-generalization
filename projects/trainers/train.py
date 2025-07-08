import torch
import logging
from .train_utils import get_gpu_memory_info, push_model
from collections import defaultdict
from tqdm import tqdm


def train(
    model_tuple,
    train_dataloader,
    eval_dataloader,
    step_fn,
    eval_fn,
    epochs,
    optimizer,
    schedulers,
    exp_folder,
    save_checkpoint_results_fn,
    logging_steps=5,
    collect_gradients=False,  # need this when calculating gradient steering vectors
    push_model_fn=None,
    save_model_locally_fn=None,
    max_grad_norm=None,  # Added parameter for gradient clipping
    device=None,
    **kwargs,
) -> dict:
    """
    Trains model. returns tuple of model, train_losses, dict of evaluation datasets -> {"loss": list of losses, "trigger_response_rate": list of trigger response rates}

    Args:
        model_tuple: Tuple of (model, tokenizer)
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader(s) for evaluation data
        step_fn: Function to perform a training step and return loss
        eval_fn: Function to evaluate the model and update eval_results
        epochs: Number of epochs to train for
        optimizer: Optimizer to use for training
        schedulers: List of schedulers to step after each optimization step
        exp_folder: Folder to save experiment results
        save_checkpoint_results_fn: Function to save checkpoint results
        logging_steps: How often to log training progress
        collect_gradients: Whether to collect gradients for later analysis
        push_model_fn: Function to push model to registry/hub
        save_model_locally_fn: Function to save model locally
        max_grad_norm: Maximum norm for gradient clipping (None = no clipping)
        **kwargs: Additional arguments
    """
    model, tokenizer = model_tuple
    print(f"Training on {'PEFT' if hasattr(model, 'peft_config') else 'full'} model")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("GPU memory after loading model:")
    get_gpu_memory_info()  # Log GPU memory after loading the model

    # Enable gradient checkpointing for memory efficiency
    if (
        "use_gradient_checkpointing" in kwargs
        and kwargs["use_gradient_checkpointing"] is True
    ):
        logging.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # Required for gradient checkpointing
        logging.info("GPU memory after enabling gradient checkpointing:")
        get_gpu_memory_info()  # Log GPU memory after enabling checkpointing

    logging.info(f"Training for {epochs} epochs")
    grad_accum_steps = (
        kwargs["gradient_accumulation_steps"]
        if "gradient_accumulation_steps" in kwargs
        else 1
    )

    # Log if we're using gradient clipping
    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with max_grad_norm={max_grad_norm}")

    if collect_gradients:
        grad_accum = defaultdict(lambda: torch.tensor(0.0).to(device))
        update_counts = defaultdict(int)
    else:
        grad_accum = None
        update_counts = None

    logging.info("Evaluating at Epoch 0 of training")
    eval_results = eval_fn(
        model, tokenizer, eval_dataloader, eval_results=None, epoch=0
    )  # this funciton accepts eval_results=None the first time it is called and returns the correct format of eval results for the current experiment with the initial results. After this, it will accept eval_results as a parameter and update it.
    train_losses = []
    model.eval()
    init_train_loss = 0.0
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Batches"):
        loss = step_fn(model, tokenizer, batch, device)
        init_train_loss += loss.item()
    init_train_loss /= len(train_dataloader)
    logging.info(f"Initial training loss: {init_train_loss:.4f}")
    train_losses.append(init_train_loss)
    for epoch in tqdm(range(epochs), desc="Epochs"):
        logging.info(f"\nEpoch {epoch + 1}/{epochs}")
        logging.info("GPU memory at start of epoch:")
        get_gpu_memory_info()

        model.train()
        total_loss = 0
        total_batches = len(train_dataloader)

        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Batches"):
            loss = step_fn(model, tokenizer, batch, device)

            loss = loss / grad_accum_steps
            loss.backward()

            # Store loss for reporting
            curr_loss = loss.item() * grad_accum_steps
            total_loss += curr_loss

            if (batch_idx % grad_accum_steps == 0) or (batch_idx == total_batches):
                if collect_gradients:
                    for name, param in model.named_parameters():
                        if param.grad is not None and param.requires_grad:
                            if name not in grad_accum:
                                grad_accum[name] = param.grad.detach().clone()
                                update_counts[name] = 1
                            else:
                                grad_accum[name] += param.grad.detach()
                                update_counts[name] += 1

                # Apply gradient clipping if specified
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                for scheduler in schedulers:
                    scheduler.step()
                optimizer.zero_grad()

            # Logging
            if batch_idx % logging_steps == 0:
                logging.info(
                    f"Batch {batch_idx}/{total_batches} "
                    f"({(batch_idx / total_batches) * 100:.1f}%) - "
                    f"Loss: {curr_loss:.4f} - "
                    f"Batch Size: {batch['input_ids'].shape[0] if type(batch['input_ids']) is torch.Tensor else len(batch['input_ids'])} - "
                    f"Seq Length: {batch['input_ids'].shape[1] if type(batch['input_ids']) is torch.Tensor else len(batch['input_ids'][0])} - "
                    f"(Accumulation Step {(batch_idx) % grad_accum_steps}/{grad_accum_steps})"
                )

        avg_loss = total_loss / total_batches
        train_losses.append(avg_loss)
        eval_results = eval_fn(
            model,
            tokenizer,
            eval_dataloader,
            eval_results,
            epoch=epoch + 1,
            is_final_epoch=(epoch == epochs - 1),
        )

        # WE NEED TO ADD CHECKPOINT FUNCTIONALITY
        if save_checkpoint_results_fn:
            save_checkpoint_results_fn(
                model,
                train_losses,
                eval_results,
                output_dir=f"{exp_folder}/checkpoints/epoch_{epoch}",
                epoch=epoch,
            )

        logging.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")
        logging.info(f"Epoch {epoch + 1}: Evaluation Results = {eval_results}")
        logging.info(f"Epoch {epoch + 1}: Train Losses = {train_losses}")
        logging.info("GPU memory at end of epoch:")
        get_gpu_memory_info()

    if push_model_fn:
        push_model_fn(model, tokenizer)

    if save_model_locally_fn:
        save_model_locally_fn(model, tokenizer)

    if not collect_gradients:
        return model, train_losses, eval_results
    else:
        return model, train_losses, eval_results, grad_accum, update_counts
