import copy
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from .base_trainer import BaseTrainer
from .train import train as train_loop
from .train_utils import load_steering_vectors, save_steering_vectors, train_test_split


class AlignmentPlaneTrainer(BaseTrainer):
    """
    A trainer class that implements alignment plane training functionality.
    This trainer handles gradient steering and alignment plane optimization.
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
        # two datasets for gradient steering

        self.alignment_plane = None

    def _prepare_datasets(
        self, split_proxy_dataset: bool = True, split_outcome_dataset: bool = True
    ):
        """
        Populates self.train_dataloader and self.eval_dataloaders

        Args:
            split_proxy_dataset: Whether to split the proxy dataset into train and test
            split_outcome_dataset: Whether to split the outcome dataset into train and test

        Returns:
            Tuple of (train_dataloader, eval_dataloaders)
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
            logging.info(f"Processing proxy dataset with {len(proxy_dataset)} samples")

            # Split using the helper method
            proxy_train, proxy_test = self._split_dataset(
                proxy_dataset, split=split_proxy_dataset, seed=self.seed
            )
            if self.training_cfg.limit_proxy_data_to:
                logging.info(
                    f"Limiting proxy dataset to {self.training_cfg.limit_proxy_data_to} samples"
                )
                proxy_train = proxy_train.select(
                    range(self.training_cfg.limit_proxy_data_to)
                )
                logging.info(f"Proxy train dataset length: {len(proxy_train)}")
                logging.info(
                    f"Proxy train dataset: {proxy_train[: self.training_cfg.limit_proxy_data_to]}"
                )

            # Store datasets
            train_datasets["proxy"] = proxy_train

            # Create evaluation dataloader
            eval_dataloaders["proxy"] = self.get_eval_dataloader(proxy_test)

        # Process proxy_neg dataset
        if proxy_neg_dataset is not None and len(proxy_neg_dataset) > 0:
            logging.info(
                f"Processing proxy_neg dataset with {len(proxy_neg_dataset)} samples"
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
                logging.info(f"Proxy neg train dataset length: {len(proxy_neg_train)}")
                logging.info(
                    f"Proxy neg dataset: {proxy_neg_train[: self.training_cfg.limit_proxy_data_to]}"
                )
            # Store datasets
            train_datasets["proxy_neg"] = proxy_neg_train

            # Create evaluation dataloader
            eval_dataloaders["proxy_neg"] = self.get_eval_dataloader(proxy_neg_test)
        elif self.training_cfg.negative_proxy is not None:
            logging.info(
                f"Negative proxy dataset not provided, using {self.training_cfg.negative_proxy} as negative proxy dataset"
            )
            train_datasets["proxy_neg"] = None

        # Process truth dataset (no splitting, evaluation only)
        if truth_dataset is not None and len(truth_dataset) > 0:
            logging.info(f"Processing truth dataset with {len(truth_dataset)} samples")

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

        self.train_datasets = {}
        self.train_datasets["outcome"] = train_datasets["outcome"]
        logging.info(
            f"[{self.__class__.__name__}] initializing the positive proxy dataset as {self.training_cfg.positive_proxy}"
        )
        self.train_datasets["positive_proxy"] = (
            train_datasets.get(self.training_cfg.positive_proxy)
            if self.training_cfg.positive_proxy is not None
            else None
        )
        if (
            self.train_datasets["positive_proxy"] is None
            and self.training_cfg.positive_proxy is not None
        ):
            raise ValueError(
                f"[{self.__class__.__name__}] was configured to use '{self.training_cfg.positive_proxy}' as the positive proxy, but this dataset is None or was not provided."
            )
        elif self.train_datasets["positive_proxy"] is not None:
            logging.info(
                f"[{self.__class__.__name__}] Positive proxy dataset '{self.training_cfg.positive_proxy}' loaded with {len(self.train_datasets['positive_proxy'])} samples."
            )

        logging.info(
            f"[{self.__class__.__name__}] initializing the negative proxy dataset as {self.training_cfg.negative_proxy}"
        )
        self.train_datasets["negative_proxy"] = (
            train_datasets.get(self.training_cfg.negative_proxy)
            if self.training_cfg.negative_proxy is not None
            else None
        )
        if (
            self.train_datasets["negative_proxy"] is None
            and self.training_cfg.negative_proxy is not None
        ):
            raise ValueError(
                f"[{self.__class__.__name__}] was configured to use '{self.training_cfg.negative_proxy}' as the negative proxy, but this dataset is None or was not provided."
            )
        elif self.train_datasets["negative_proxy"] is not None:
            logging.info(
                f"[{self.__class__.__name__}] Negative proxy dataset '{self.training_cfg.negative_proxy}' loaded with {len(self.train_datasets['negative_proxy'])} samples."
            )

        if (
            self.train_datasets["positive_proxy"] is None
            and self.train_datasets["negative_proxy"] is None
        ):
            raise ValueError(
                f"[{self.__class__.__name__}] No proxy datasets provided or resolved. Check `positive_proxy` and `negative_proxy` configs."
            )
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

    def train_step(self, model, tokenizer, batch: Dict, device: str) -> torch.Tensor:
        input_ids = torch.stack(batch["input_ids"]).to(device)
        attention_mask = torch.stack(batch["attention_mask"]).to(device)
        labels = (
            input_ids.clone()
            if "labels" not in batch
            else torch.stack(batch["labels"]).to(device)
        )
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def merge_proxy_gradients(self):
        """
        Merge proxy gradients into the model based on training_cfg.proxy_to_merge.
        Handles both PEFT and non-PEFT models differently.
        Case-insensitive to proxy_to_merge value.
        """
        logging.error(
            "this method is not currently supported because of optimizations for steering weights memory. please set proxy_to_merge = None in the finetune config"
        )
        return self.model
        logging.info("Attempting to merge proxy gradients into task model")
        is_peft = hasattr(self.model, "peft_config")
        if self.training_cfg.is_peft != is_peft:
            raise ValueError(
                f"hasattr peft_config: {is_peft}, training_cfg.is_peft: {self.training_cfg.is_peft}"
            )

        proxy_type = self.training_cfg.proxy_to_merge.lower()

        if proxy_type == "alignment_plane":
            if is_peft:
                # For PEFT, calculate and merge LoRA weights into base weights
                for name, param in self.model.named_parameters():
                    if name in self.alignment_plane:
                        # Get the corresponding LoRA weights
                        lora_A = self.model.get_parameter(f"{name}.lora_A.weight")
                        lora_B = self.model.get_parameter(f"{name}.lora_B.weight")
                        # Calculate the LoRA update
                        lora_update = lora_B @ lora_A
                        # Add both LoRA update and alignment plane to base weights
                        param.data.add_(lora_update + self.alignment_plane[name])
            else:
                # For non-PEFT, just add alignment plane to base weights
                for name, param in self.model.named_parameters():
                    if name in self.alignment_plane:
                        param.data.add_(self.alignment_plane[name])

        elif proxy_type == "positive_proxy":
            # For both PEFT and non-PEFT, merge positive proxy model
            if is_peft:
                logging.info(
                    "Model is peft, so we are merging neg proxy gradients into the model before setting self.model = proxy model"
                )
                self.proxy_model = self.proxy_model.merge_and_unload()
            self.model = self.proxy_model

        elif proxy_type == "negative_proxy":
            # For both PEFT and non-PEFT, merge negative proxy model
            if is_peft:
                logging.info(
                    "Model is peft, so we are merging neg proxy gradients into the model before setting self.model = proxy minus model"
                )
                self.proxy_minus_model = self.proxy_minus_model.merge_and_unload()
            self.model = self.proxy_minus_model

        else:
            raise ValueError(
                f"Invalid proxy_to_merge value: {self.training_cfg.proxy_to_merge}"
            )
        try:
            del self.proxy_model
            del self.proxy_minus_model
        except Exception as e:
            logging.error(f"Error deleting proxy models: {e}")
        logging.info("Proxy gradients merged into the model")

        return self.model

    def get_proxy_gradients(
        self,
        model,
        tokenizer,
        proxy_type: str,
        save_checkpoint_results_fn=None,
        save_proxy_results_fn=None,
    ):
        """
        Helper for get_alignment_plane. Gets one of the proxy (positive or negative) gradients.
        """
        if self.train_dataloaders[proxy_type] is None:
            raise ValueError(f"No training dataloader found for {proxy_type}")

        model = self.prepare_for_training(model)
        optimizer, lr_scheduler = self.get_standard_optimizer_and_scheduler(
            model,
            train_dataloader=self.train_dataloaders[proxy_type],
            epochs=self.training_cfg.proxy_epochs
            if self.training_cfg.proxy_epochs is not None
            else None,
        )

        (
            proxy_model,
            proxy_train_losses,
            proxy_eval_results,
            proxy_grad_accum,
            proxy_update_counts,
        ) = train_loop(
            (model, tokenizer),
            self.train_dataloaders[proxy_type],
            self.eval_dataloaders,
            self.train_step,
            self.evaluate,
            self.training_cfg.epochs
            if self.training_cfg.proxy_epochs is None
            else self.training_cfg.proxy_epochs,
            optimizer,
            [lr_scheduler],
            self.exp_folder,
            save_checkpoint_results_fn,
            logging_steps=self.training_cfg.logging_steps,
            collect_gradients=True,
            push_model_fn=None,
            save_model_locally_fn=self.save_model_locally
            if self.training_cfg.save_model_locally
            else None,
            max_grad_norm=self.training_cfg.max_grad_norm
            if hasattr(self.training_cfg, "max_grad_norm")
            else None,
        )

        if save_proxy_results_fn:
            save_proxy_results_fn(
                proxy_train_losses,
                proxy_eval_results,
                output_dir=f"{self.exp_folder}/{proxy_type}",
            )
        return proxy_model, proxy_grad_accum, proxy_update_counts

        # Calculate training steps

    def get_alignment_plane(
        self, save_checkpoint_results_fn=None, save_proxy_results_fn=None
    ):
        """
        Get the gradient steering weights for the positive and negative proxy datasets.
        """
        # log the parameters that require grad in the model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logging.info(f"Parameter {name} requires grad: {param.requires_grad}")

        if self.alignment_plane is not None:
            return self.alignment_plane
        if self.training_cfg.load_steering_vector and os.path.exists(
            self.training_cfg.steering_vector_path
        ):
            try:
                logging.info(
                    f"Attempting to load steering weights from {self.training_cfg.steering_vector_path}"
                )
                steering_vector = load_steering_vectors(
                    self.training_cfg.steering_vector_path, self.device
                )
                logging.info("Loaded steering vector from file")
                return steering_vector
            except Exception as e:
                logging.error(
                    f"Error loading steering vector from file: {e}. Proceeding with gradient calculation."
                )
        elif self.training_cfg.load_steering_vector:
            logging.info(
                f"Steering vector checkpoint not found: {self.training_cfg.steering_vector_path}"
            )
            logging.info(
                "Training without loading steering vector. Proceeding with gradient calculation."
            )
        logging.info(
            "Getting gradient steering vector from proxy and proxy minus tasks"
        )
        self.proxy_model = copy.deepcopy(self.model)
        # log params that require grad in self.model if they're lora

        if (
            self.train_dataloaders["positive_proxy"] is not None
            and self.training_cfg.positive_proxy is not None
        ):
            logging.info("Phase 1: Accumulating gradients for proxy data")

            self.proxy_model, self.proxy_grad_accum, self.proxy_update_counts = (
                self.get_proxy_gradients(
                    self.proxy_model,
                    self.tokenizer,
                    "positive_proxy",
                    save_checkpoint_results_fn,
                    save_proxy_results_fn,
                )
            )

            # logging.info("PROXY MODEL PARAMETERS:")
            # for name, param in self.proxy_model.named_parameters():
            #     if param.requires_grad:
            #         logging.info(
            #             f"Parameter {name} requires grad: {param.requires_grad}"
            #         )
            if not self.training_cfg.calc_alignment_plane_using_fully_finetuned_weights:
                logging.info(
                    "Positive proxy gradients accumulated, deleting proxy model for memory efficiency"
                )
                del self.proxy_model
                self.proxy_model = None
        else:
            logging.info("Skiped accumulating gradients for positive proxy data")
            self.proxy_grad_accum = None
            self.proxy_update_counts = None

        self.proxy_minus_model = copy.deepcopy(self.model)
        # log params that require grad in self.proxy_minus_model if they're lora

        if (
            self.train_dataloaders["negative_proxy"] is not None
            and self.training_cfg.negative_proxy is not None
        ):
            logging.info("Phase 2: calculating gradients for negative proxy data")

            (
                self.proxy_minus_model,
                self.proxy_minus_grad_accum,
                self.proxy_minus_update_counts,
            ) = self.get_proxy_gradients(
                self.proxy_minus_model,
                self.tokenizer,
                "negative_proxy",
                save_checkpoint_results_fn,
                save_proxy_results_fn,
            )
            # logging.info("PROXY MINUS MODEL PARAMETERS:")
            # for name, param in self.proxy_minus_model.named_parameters():
            #     if param.requires_grad:
            #         logging.info(
            #             f"Parameter {name} requires grad: {param.requires_grad}"
            #         )
            if not self.training_cfg.calc_alignment_plane_using_fully_finetuned_weights:
                logging.info(
                    "Negative proxy gradients accumulated, deleting proxy minus model for memory efficiency"
                )
                del self.proxy_minus_model
                self.proxy_minus_model = None
        else:
            logging.info("Skipping negative proxy gradient accumulation")
            self.proxy_minus_grad_accum = None
            self.proxy_minus_update_counts = None

        logging.info("Calculating alignment plane")

        self.alignment_plane = self.calculate_alignment_plane(
            self.proxy_model,
            self.proxy_minus_model,
            self.proxy_grad_accum,
            self.proxy_minus_grad_accum,
            self.proxy_update_counts,
            self.proxy_minus_update_counts,
        )
        del self.proxy_grad_accum
        del self.proxy_minus_grad_accum
        del self.proxy_update_counts
        del self.proxy_minus_update_counts
        self.proxy_grad_accum = None
        self.proxy_minus_grad_accum = None
        self.proxy_update_counts = None
        if not self.training_cfg.proxy_to_merge:
            del self.proxy_model
            del self.proxy_minus_model
            self.proxy_model = None
            self.proxy_minus_model = None
        self._clean_memory()

        if self.training_cfg.save_steering_vector:
            filepath = self.training_cfg.steering_vector_path
            logging.info(f"Attempting to alignment plaen to {filepath}")
            try:
                save_steering_vectors(self.alignment_plane, filepath)
                logging.info(f"Steering vector saved to {filepath}")
            except Exception as e:
                logging.error(f"Error saving steering vector: {e}")
                if os.path.exists(
                    f"{self.exp_folder}/{self.training_cfg.steering_vector_path}"
                ):
                    os.remove(
                        f"{self.exp_folder}/{self.training_cfg.steering_vector_path}"
                    )
                    logging.info(f"Removed existing alignment plane file at {filepath}")

        return self.alignment_plane

    def calculate_alignment_plane(
        self,
        proxy_model,
        proxy_minus_model,
        proxy_grad_accum,
        proxy_minus_grad_accum,
        proxy_update_counts,
        proxy_minus_update_counts,
    ):
        # assert that proxy_model, proxy_minus_model, and self.model are different objects
        if (
            proxy_model
            and proxy_minus_model
            and (
                (proxy_model is self.model)
                or (proxy_minus_model is self.model)
                or (proxy_model is proxy_minus_model)
            )
        ):
            raise ValueError(
                "proxy_model, proxy_minus_model, and self.model must be different objects"
            )
        alignment_plane = dict()
        logging.info("Calculating steering vector")

        if proxy_grad_accum is not None and proxy_minus_grad_accum is not None:
            logging.info(
                "Calculating alignment plane using both positive and negative proxy gradients"
            )

            first_iteration = True
            for name, proxy_grad_accum in proxy_grad_accum.items():
                if first_iteration:
                    logging.info(f"calculating alignment plane for name: {name}")
                    first_iteration = False

                avg_proxy_grad = proxy_grad_accum / proxy_update_counts[name]
                assert name in proxy_minus_grad_accum
                avg_proxy_minus_grad = (
                    proxy_minus_grad_accum[name] / proxy_minus_update_counts[name]
                )
                if self.training_cfg.add_proxy_gradients:
                    if first_iteration:
                        logging.info(
                            "Calculating steering vector by adding proxy gradients"
                        )
                    sw = avg_proxy_grad + avg_proxy_minus_grad
                else:
                    if first_iteration:
                        logging.info(
                            "Calculating steering vector with positive - negative avg proxy gradient"
                        )
                    sw = avg_proxy_grad - avg_proxy_minus_grad

                alignment_plane[name] = sw
        elif proxy_grad_accum is not None:
            logging.info(
                "Only positive proxy gradient provided, using it for alignment plane"
            )
            first_iteration = True
            for name, param in proxy_model.named_parameters():
                if param.requires_grad:
                    if name in proxy_grad_accum:
                        alignment_plane[name] = (
                            proxy_grad_accum[name] / proxy_update_counts[name]
                        )
        elif proxy_minus_grad_accum is not None:
            logging.info(
                "Only negative proxy gradient provided, using it for alignment plane"
            )
            first_iteration = True
            for name, param in proxy_model.named_parameters():
                if param.requires_grad:
                    if name in proxy_minus_grad_accum:
                        alignment_plane[name] = (
                            proxy_minus_grad_accum[name]
                            / proxy_minus_update_counts[name]
                        )
                        if self.training_cfg.add_proxy_gradients:
                            alignment_plane[name] = -alignment_plane[name]

        if self.training_cfg.calc_alignment_plane_using_fully_finetuned_weights:
            print("ONLY USE FOR ADDING ALIGNMENT PLANE WITH STEERING WEIGHTS AT END")
            logging.info("Calculating steering vector using alignment plane method")
            norms = dict()
            for name, plane in alignment_plane.items():
                norms[name] = plane.norm()
            del alignment_plane
            alignment_plane_alt = dict()
            for (name_proxy, param_proxy), (name_proxy_minus, param_proxy_minus) in zip(
                proxy_model.named_parameters(), proxy_minus_model.named_parameters()
            ):
                if name_proxy != name_proxy_minus:
                    raise ValueError(
                        f"Parameter names do not match: {name_proxy} != {name_proxy_minus}"
                    )
                
                if self.training_cfg.add_proxy_gradients:
                    raise NotImplementedError("You shouldn't do this")
                    base_param = self.model.get_parameter(name_proxy)
                    neg_sv = param_proxy_minus + param_proxy - 2 * base_param
                else:
                    neg_sv = param_proxy_minus - param_proxy

                desired_norm = norms[name_proxy]
                neg_sv_norm = neg_sv.norm()
                if neg_sv_norm > 0:
                    neg_sv = neg_sv * (desired_norm / neg_sv_norm)

                alignment_plane_alt[name_proxy] = neg_sv
            alignment_plane = alignment_plane_alt
            del norms
            del proxy_model
            del proxy_minus_model
        return alignment_plane
