import copy
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from .positive_negative_proxy_trainer import PositiveNegativeProxyTrainer
from .train import train as train_loop
from .train_utils import (
    SafeLoRA,
    SafeLoRAConfig,
    train_test_split,
)


class SafeLoRATrainer(BaseTrainer):
    """
    Trainer for a "Safe LoRA" inspired approach:
    1. Train LoRA adapters on a positive proxy task.
    2. Merge adapters to get an "aligned" model.
    3. Fine-tune the "aligned" model on an outcome task, resulting in a "misaligned" model.
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
        exp_folder: str | None = None,
        device: str = "cuda",
        seed: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the SafeLoRATrainer.

        Args:
            model: Base model to train.
            tokenizer: Tokenizer for the model.
            training_cfg: Training configuration.
            collate_fn: Function to collate batches.
            eval_fn: Function to evaluate the model.
            positive_proxy_dataset: Dataset for initial LoRA training (alignment).
            outcome_dataset: Dataset for subsequent fine-tuning (misalignment).
            exp_folder: Experiment folder for saving artifacts.
            device: Device to use for training.
            seed: Random seed.
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
        if "seed" in self.exp_folder and self.training_cfg.safe_lora_sweep_thresholds:
            raise ValueError(
                "safe_lora_sweep_thresholds is not supported for multi-seed experiments"
            )

        # Store a pristine copy of the original base model
        # get the model's name
        model_name = model.name_or_path
        try:
            import sys

            sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
            from experiment_utils import load_model_and_tokenizer

            base_model_name = model_name.replace("-it", "")
            logging.info("Loading original base model from: %s", base_model_name)
            print("Loading original base model from:", base_model_name)
            temporary_training_cfg = copy.deepcopy(self.training_cfg)
            temporary_training_cfg.model = base_model_name
            self.original_base_model, _ = load_model_and_tokenizer(
                temporary_training_cfg
            )
        except Exception as e:
            logging.error(f"Failed to load original base model from {model_name}: {e}")
            print(f"Failed to load original base model from {model_name}: {e}")

            self.original_base_model = copy.deepcopy(model)
            self.original_base_model.to(
                self.device
            )  # Ensure it's on the correct device

        self.aligned_model: Optional[torch.nn.Module] = None
        self.misaligned_model: Optional[torch.nn.Module] = None

        if proxy_dataset is None:
            raise ValueError("proxy_dataset is required for SafeLoRATrainer.")
        if outcome_dataset is None:
            raise ValueError("outcome_dataset is required for SafeLoRATrainer.")

        if self.training_cfg.safe_lora_type is None:
            raise ValueError("safe_lora_type is required for SafeLoRATrainer.")
        if (
            self.training_cfg.safe_lora_type == "threshold"
            and self.training_cfg.safe_lora_threshold is None
        ):
            raise ValueError(
                "safe_lora_threshold is required for SafeLoRATrainer when safe_lora_type is 'threshold'."
            )
        if (
            self.training_cfg.safe_lora_type == "number"
            and self.training_cfg.safe_lora_num_proj_layers is None
        ):
            raise ValueError(
                "safe_lora_num_proj_layers is required for SafeLoRATrainer when safe_lora_type is 'number'."
            )

        if not hasattr(self.training_cfg, "proxy_epochs"):
            logging.warning(
                "training_cfg.proxy_epochs not set. Defaulting to training_cfg.epochs for proxy training."
            )
            # Or set a default: self.training_cfg.proxy_epochs = self.training_cfg.epochs

    def _check_peft_status(
        self,
        unaligned_model: torch.nn.Module,
        aligned_model: torch.nn.Module,
        misaligned_model: torch.nn.Module,
    ):
        from .base_trainer import has_active_adapters

        if hasattr(unaligned_model, "peft_config") and has_active_adapters(
            unaligned_model
        ):
            raise ValueError(
                "SafeLoRATrainer requires that the unaligned model is not peft."
            )
        if hasattr(aligned_model, "peft_config") and has_active_adapters(aligned_model):
            raise ValueError(
                "SafeLoRATrainer requires that the aligned model is not peft."
            )
        if not (
            hasattr(misaligned_model, "peft_config")
            # and has_active_adapters(misaligned_model)
        ):
            raise ValueError(
                "SafeLoRATrainer requires that the misaligned model is peft."
            )
        return True

    def _prepare_datasets(
        self, split_proxy_dataset: bool = True, split_outcome_dataset: bool = True
    ):
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

            # Create evaluation dataloader
            eval_dataloaders["proxy_neg"] = self.get_eval_dataloader(proxy_neg_test)

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

        # Determine steps per epoch based on cycling mode

        # Store reference to evaluation dataloaders
        self.eval_dataloaders = eval_dataloaders

        # Create combined training dataset if we have both outcome and proxy datasets

        self.train_datasets = {}
        self.train_datasets["outcome"] = train_datasets["outcome"]
        logging.info(
            f"initializing the positive proxy dataset as {self.training_cfg.positive_proxy}"
        )
        self.train_datasets["positive_proxy"] = (
            train_datasets[self.training_cfg.positive_proxy]
            if self.training_cfg.positive_proxy is not None
            else None
        )
        logging.info(
            f"initializing the negative proxy dataset as {self.training_cfg.negative_proxy}"
        )
        self.train_datasets["negative_proxy"] = (
            train_datasets[self.training_cfg.negative_proxy]
            if self.training_cfg.negative_proxy is not None
            else None
        )
        if (
            self.train_datasets["positive_proxy"] is None
            and self.train_datasets["negative_proxy"] is None
        ):
            raise ValueError("No proxy datasets provided")
        train_dataloaders = {}

        train_dataloaders["outcome"] = DataLoader(
            self.train_datasets["outcome"],
            batch_size=self.training_cfg.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        train_dataloaders["positive_proxy"] = (
            DataLoader(
                self.train_datasets["positive_proxy"],
                batch_size=self.training_cfg.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
            if self.train_datasets["positive_proxy"] is not None
            else None
        )
        train_dataloaders["negative_proxy"] = (
            DataLoader(
                self.train_datasets["negative_proxy"],
                batch_size=self.training_cfg.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
            if self.train_datasets["negative_proxy"] is not None
            else None
        )

        # Store references for use by the trainer
        self.train_dataloaders = train_dataloaders
        self.eval_dataloaders = eval_dataloaders

        # print the keys of the first batch of the train_dataloader
        print(train_dataloaders["outcome"].dataset[0].keys())
        # print the keys of the first batch of the eval_dataloaders
        print(eval_dataloaders["outcome"].dataset[0].keys())
        if self.training_cfg.save_datasets:
            # Save the train and eval datasets to disk
            self.save_datasets()
        return train_dataloaders, eval_dataloaders

    def _apply_safe_lora(
        self,
        unaligned_model: torch.nn.Module,
        aligned_model: torch.nn.Module,
        misaligned_model: torch.nn.Module,
    ) -> torch.nn.Module:
        safe_lora_cfg = SafeLoRAConfig(
            base_model=unaligned_model,
            aligned_model=aligned_model,
            load_in_8bit=False,
            load_in_4bit=False,
            devices=self.device,
            threshold=self.training_cfg.safe_lora_threshold,
            num_proj_layers=self.training_cfg.safe_lora_num_proj_layers,
            select_layers_type=self.training_cfg.safe_lora_type,
            results_dir=self.exp_folder,
        )
        safelora = SafeLoRA(misaligned_model, safe_lora_cfg, logger=logging)
        safe_model = safelora.model
        return safe_model

    def train(
        self,
        save_checkpoint_results_fn: Optional[Callable] = None,
        save_results_fn: Optional[
            Callable
        ] = None,  # This would be for the final (misaligned) results
    ) -> Tuple[torch.nn.Module, List[float], Dict[str, Any]]:
        logging.info("Starting SafeLoRATrainer training process...")

        # --- Stage 1: Train base model on positive proxy dataset with LoRA ---
        logging.info("Stage 1: Training on positive proxy dataset with LoRA")
        if not self.training_cfg.is_peft:
            logging.warning(
                "training_cfg.is_peft is False. LoRA will not be applied for the proxy training stage. "
                "Ensure this is intended for SafeLoRATrainer."
            )

        lora_train_model = copy.deepcopy(self.original_base_model)
        # prepare_for_training will apply LoRA if self.training_cfg.is_peft is True
        lora_train_model = self.prepare_for_training(lora_train_model)

        proxy_epochs = getattr(
            self.training_cfg, "proxy_epochs", self.training_cfg.epochs
        )
        proxy_optimizer, proxy_lr_scheduler = self.get_standard_optimizer_and_scheduler(
            lora_train_model,
            train_dataloader=self.train_dataloaders["positive_proxy"],
            epochs=proxy_epochs,
        )

        stage1_eval_dataloaders = {"positive_proxy": self.eval_dataloaders.get("proxy")}
        if (
            stage1_eval_dataloaders["positive_proxy"] is None
            and "positive_proxy" in self.eval_dataloaders
        ):
            # This case should ideally not happen if _prepare_datasets works correctly
            logging.warning(
                "Positive proxy eval dataloader was expected but not found for stage 1 eval."
            )

        trained_lora_model, proxy_train_losses, proxy_eval_results = train_loop(
            model_tuple=(lora_train_model, self.tokenizer),
            train_dataloader=self.train_dataloaders["positive_proxy"],
            eval_dataloader=stage1_eval_dataloaders,
            step_fn=self.train_step,
            eval_fn=self.evaluate,
            epochs=proxy_epochs,
            optimizer=proxy_optimizer,
            schedulers=[proxy_lr_scheduler],
            exp_folder=os.path.join(
                self.exp_folder, "stage1_proxy_train"
            ),  # Stage-specific output for checkpoints
            save_checkpoint_results_fn=save_checkpoint_results_fn,
            logging_steps=self.training_cfg.logging_steps,
            max_grad_norm=getattr(self.training_cfg, "max_grad_norm", None),
        )
        logging.info("Stage 1: Positive proxy training complete.")

        # --- Stage 2: Merge adaptors and save 'aligned' model ---
        logging.info("Stage 2: Merging LoRA adaptors and saving 'aligned' model.")
        if hasattr(trained_lora_model, "peft_config") and self.training_cfg.is_peft:
            self.aligned_model = trained_lora_model.merge_and_unload()
            logging.info("LoRA adaptors merged.")
        else:
            logging.warning(
                "Model from proxy training does not have peft_config or training_cfg.is_peft was False. "
                "Using it directly as 'aligned_model'."
            )
            self.aligned_model = trained_lora_model

        self.aligned_model.to(
            self.device
        )  # Ensure it's on device after potential merge

        # Save the aligned model
        # original_finetuned_model_id = self.training_cfg.finetuned_model_id
        # aligned_model_id = f"{original_finetuned_model_id}_aligned"
        # # Temporarily set the ID for saving this specific model
        # current_save_id = self.training_cfg.finetuned_model_id
        # self.training_cfg.finetuned_model_id = aligned_model_id
        # self.save_model_locally(self.aligned_model, self.tokenizer)
        # self.training_cfg.finetuned_model_id = (
        #     current_save_id  # Restore original ID for later use
        # )
        # logging.info(
        #     f"Aligned model saved locally using derived ID: {aligned_model_id}"
        # )

        # --- Stage 3: Train 'aligned' model on outcome dataset (peft) ---
        logging.info("Stage 3: Training 'aligned' model on outcome dataset (peft).")

        outcome_train_model = copy.deepcopy(self.aligned_model)
        outcome_train_model.to(self.device)

        if not self.training_cfg.is_peft:
            # Ensure PEFT is re-applied for this stage if we want PEFT fine-tuning
            original_is_peft_setting = self.training_cfg.is_peft
            self.training_cfg.is_peft = True  # Enable PEFT for preparing this model
            outcome_train_model = self.prepare_for_training(
                outcome_train_model
            )  # Should now prepare for dense training
            self.training_cfg.is_peft = original_is_peft_setting  # Restore setting
        else:
            # turn the model into a base model, not peft, before passing to prepare_for_training
            outcome_train_model = self.prepare_for_training(outcome_train_model)
        print("the model's parameters that require grad")
        print("Parameters of outcome_train_model that require grad:")
        for name, param in outcome_train_model.named_parameters():
            if param.requires_grad:
                print(f"Parameter {name} requires grad: {param.requires_grad}")

        outcome_optimizer, outcome_lr_scheduler = (
            self.get_standard_optimizer_and_scheduler(
                outcome_train_model,
                train_dataloader=self.train_dataloaders["outcome"],
                epochs=self.training_cfg.epochs,  # Main epochs for outcome
            )
        )

        stage2_eval_dataloaders = {"outcome": self.eval_dataloaders.get("outcome")}

        self.misaligned_model, outcome_train_losses, outcome_eval_results = train_loop(
            (outcome_train_model, self.tokenizer),
            self.train_dataloaders["outcome"],
            stage2_eval_dataloaders,
            self.train_step,
            self.evaluate,
            self.training_cfg.epochs,
            outcome_optimizer,
            [outcome_lr_scheduler],
            os.path.join(
                self.exp_folder, "stage3_outcome_train"
            ),  # Stage-specific output
            save_checkpoint_results_fn=save_checkpoint_results_fn,
            logging_steps=self.training_cfg.logging_steps,
            max_grad_norm=getattr(self.training_cfg, "max_grad_norm", None),
        )
        logging.info("Stage 3: Outcome training complete. Model is now 'misaligned'.")
        self.misaligned_model.to(self.device)
        # print(f"Misaligned model: {self.misaligned_model}")
        print("misaligned model type: ", type(self.misaligned_model))
        self._check_peft_status(
            self.original_base_model, self.aligned_model, self.misaligned_model
        )
        self.aligned_model.to(self.device)
        self.original_base_model.to(self.device)
        self.misaligned_model.to(self.device)

        # --- Stage 4: Push and Save 'misaligned' model ---
        logging.info("Stage 4: Pushing and saving 'misaligned' model.")
        # training_cfg.finetuned_model_id should now point to the desired ID for the misaligned model
        if getattr(
            self.training_cfg, "push_to_hub", True
        ):  # Assuming a config to control push
            self.push_model(self.misaligned_model, self.tokenizer)
        else:
            logging.info("Skipping push to hub based on configuration.")

        self.save_model_locally(self.misaligned_model, self.tokenizer)
        logging.info(
            f"Misaligned model (final) pushed (if enabled) and saved locally as {self.training_cfg.finetuned_model_id}."
        )

        if self.training_cfg.safe_lora_sweep_thresholds:
            if self.training_cfg.safe_lora_type == "number":
                raise ValueError(
                    "safe_lora_sweep_thresholds is not supported for safe_lora_type == 'number'"
                )
            if self.training_cfg.safe_lora_threshold is not None:
                raise ValueError(
                    "safe_lora_threshold must be None when safe_lora_sweep_thresholds is not None"
                )
            for threshold in self.training_cfg.safe_lora_sweep_thresholds:
                self.training_cfg.safe_lora_threshold = threshold
                threshold_folder = os.path.join(
                    self.exp_folder, f"threshold_{threshold}"
                )
                os.makedirs(threshold_folder, exist_ok=True)
                # misaligned_model = copy.deepcopy(self.misaligned_model)
                safe_lora_model = self._apply_safe_lora(
                    self.original_base_model, self.aligned_model, self.misaligned_model
                )
                safe_lora_eval_results = self.evaluate(
                    safe_lora_model,
                    self.tokenizer,
                    self.eval_dataloaders,
                    eval_results=None,
                )
                if save_results_fn:
                    # save_results_fn is typically for the *final* results of the trainer
                    save_results_fn(
                        train_losses=outcome_train_losses,  # from the misaligning stage
                        eval_results=safe_lora_eval_results,  # from the misaligning stage
                        output_dir=threshold_folder,  # Main experiment folder
                    )
        else:
            safe_lora_model = self._apply_safe_lora(
                self.original_base_model, self.aligned_model, self.misaligned_model
            )
            safe_lora_eval_results = self.evaluate(
                safe_lora_model,
                self.tokenizer,
                self.eval_dataloaders,
                eval_results=None,
            )
            if save_results_fn:
                # save_results_fn is typically for the *final* results of the trainer
                save_results_fn(
                    train_losses=outcome_train_losses,  # from the misaligning stage
                    eval_results=safe_lora_eval_results,  # from the misaligning stage
                    output_dir=os.path.join(
                        self.exp_folder, "results"
                    ),  # Main experiment folder
                )

        return safe_lora_model, None, safe_lora_eval_results
