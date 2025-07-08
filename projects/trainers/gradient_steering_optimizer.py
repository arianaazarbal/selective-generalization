import logging
from typing import Callable, Dict, Iterable, List, Optional, Union

import torch
from torch.optim import AdamW


class SteeringVectorAdamW(AdamW):
    """
    Extension of AdamW that adds steering vectors to gradients before each step.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], List[Dict]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        named_parameters: List[tuple] = None,  # Added this parameter
        steering_vectors: Dict[str, torch.Tensor] = None,
        alpha: float = 0.1,
    ):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to the denominator to improve numerical stability
            weight_decay: Weight decay coefficient
            amsgrad: Whether to use the AMSGrad variant
            named_parameters: List of (name, parameter) tuples to associate names with parameters
            steering_vectors: Dictionary mapping parameter names to steering vectors
            alpha: Scaling factor for steering vectors
        """
        logging.info("Initializing SteeringVectorAdamW optimizer")

        super(SteeringVectorAdamW, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        # Store steering vectors and alpha
        self.steering_vectors = steering_vectors or {}
        self.alpha = alpha

        logging.info(f"Alpha value set to: {alpha}")
        logging.info(
            f"Number of steering vectors provided: {len(self.steering_vectors)}"
        )

        # Log steering vector details
        # for name, vector in self.steering_vectors.items():
        #     logging.info(
        #         f"Steering vector '{name}' with shape {vector.shape}, "
        #         f"norm: {torch.norm(vector).item():.6f}"
        #     )

        self.named_parameters = named_parameters
        assert self.named_parameters is not None and (
            len(list(self.named_parameters)) != 0
        )

        logging.info(
            f"Number of named parameters provided: {len(list(self.named_parameters))}"
        )

        # Create a mapping from parameter objects to their names for faster lookup
        self.param_to_name = {}
        for name, param in self.named_parameters:
            if param is not None:
                self.param_to_name[param] = name
        logging.info(f"All parameter names: {self.param_to_name.values()}")
        for name in self.steering_vectors:
            assert name in self.param_to_name.values(), (
                f"Steering vector '{name}' not found in named parameters"
            )
        logging.info(
            f"Created parameter-to-name mapping with {len(self.param_to_name)} entries"
        )

        # Add alpha to each parameter group for proper serialization
        for group in self.param_groups:
            group["alpha"] = alpha

        logging.info("SteeringVectorAdamW initialization complete")

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step with steering vector modification.

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        logging.info("Beginning optimization step with steering vectors")

        # Save original gradients
        original_grads = {}
        grad_stats = {"with_grad": 0, "without_grad": 0}

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    original_grads[p] = p.grad.detach().clone()
                    grad_stats["with_grad"] += 1
                else:
                    grad_stats["without_grad"] += 1

        # logging.info(
        #     f"Parameters with gradients: {grad_stats['with_grad']}, "
        #     f"without gradients: {grad_stats['without_grad']}"
        # )

        # Apply steering to gradients
        steering_applied = 0
        steering_skipped_no_name = 0
        steering_skipped_not_in_dict = 0
        steering_skipped_shape_mismatch = 0

        # Use the parameter-to-name mapping for applying steering vectors
        for param, grad in original_grads.items():
            name = self.param_to_name.get(param)

            if name is None:
                logging.info(
                    "Parameter has no name mapping, cannot match with steering vector"
                )
                steering_skipped_no_name += 1
                continue

            if name not in self.steering_vectors:
                logging.info(f"No steering vector found for parameter '{name}'")
                steering_skipped_not_in_dict += 1
                continue

            steering_vector = self.steering_vectors[name]

            # Ensure dimensions match
            if param.grad.shape != steering_vector.shape:
                logging.info(
                    f"Shape mismatch for parameter '{name}': "
                    f"gradient shape {param.grad.shape} vs. "
                    f"steering vector shape {steering_vector.shape}"
                )
                steering_skipped_shape_mismatch += 1
                continue

            # Apply steering: grad = grad + alpha * steering_vector
            pre_norm = torch.norm(param.grad.data).item()
            param.grad.data.add_(steering_vector, alpha=self.alpha)
            post_norm = torch.norm(param.grad.data).item()

            # logging.info(
            #     f"Applied steering to '{name}': "
            #     f"gradient norm before: {pre_norm:.6f}, "
            #     f"after: {post_norm:.6f}, "
            #     f"steering vector norm: {torch.norm(steering_vector).item():.6f}, "
            #     f"alpha: {self.alpha}"
            # )
            steering_applied += 1

        # logging.info(
        #     f"Steering summary: applied to {steering_applied} parameters, "
        #     f"skipped {steering_skipped_no_name} (no name), "
        #     f"skipped {steering_skipped_not_in_dict} (not in dict), "
        #     f"skipped {steering_skipped_shape_mismatch} (shape mismatch)"
        # )

        # Call the parent's step method with modified gradients
        # logging.info("Calling parent AdamW.step() with modified gradients")
        loss = super(SteeringVectorAdamW, self).step(closure)

        # Restore original gradients
        # logging.info("Restoring original gradients")
        for p, grad in original_grads.items():
            p.grad = grad

        logging.info("Optimization step complete")
        return loss

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        logging.info("Saving optimizer state")
        state_dict = super(SteeringVectorAdamW, self).state_dict()
        state_dict["steering_vectors"] = self.steering_vectors
        state_dict["alpha"] = self.alpha
        # We can't easily save the parameter-to-name mapping since parameters aren't serializable
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        logging.info("Loading optimizer state")
        self.steering_vectors = state_dict.pop("steering_vectors", {})
        self.alpha = state_dict.pop("alpha", 0.1)
        logging.info(
            f"Loaded {len(self.steering_vectors)} steering vectors with alpha={self.alpha}"
        )
        super(SteeringVectorAdamW, self).load_state_dict(state_dict)

    def __setstate__(self, state):
        """Makes sure alpha is included when unpickling."""
        super(SteeringVectorAdamW, self).__setstate__(state)
        # If alpha is missing from any param_group, set it to the default
        for group in self.param_groups:
            group.setdefault("alpha", self.alpha)
        logging.info("State restored with __setstate__")


class DirectSteeringVectorAdamW(AdamW):
    """
    Extension of AdamW that adds steering vectors to gradients before each step.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], List[Dict]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        named_parameters: List[tuple] = None,  # Added this parameter
        steering_vectors: Dict[str, torch.Tensor] = None,
        alpha: float = 0.1,
    ):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to the denominator to improve numerical stability
            weight_decay: Weight decay coefficient
            amsgrad: Whether to use the AMSGrad variant
            named_parameters: List of (name, parameter) tuples to associate names with parameters
            steering_vectors: Dictionary mapping parameter names to steering vectors
            alpha: Scaling factor for steering vectors
        """
        logging.info("Initializing DirectSteeringVectorAdamW optimizer")

        super(DirectSteeringVectorAdamW, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        # Store steering vectors and alpha
        self.steering_vectors = steering_vectors or {}
        self.alpha = alpha

        logging.info(f"Alpha value set to: {alpha}")
        logging.info(
            f"Number of steering vectors provided: {len(self.steering_vectors)}"
        )

        # Log steering vector details
        # Uncomment for debugging if needed
        # for name, vector in self.steering_vectors.items():
        #     logging.info(
        #         f"Steering vector '{name}' with shape {vector.shape}, "
        #         f"norm: {torch.norm(vector).item():.6f}"
        #     )

        self.named_parameters = named_parameters
        assert self.named_parameters is not None and (
            len(list(self.named_parameters)) != 0
        )

        logging.info(
            f"Number of named parameters provided: {len(list(self.named_parameters))}"
        )

        # Create a mapping from parameter objects to their names for faster lookup
        self.param_to_name = {}
        for name, param in self.named_parameters:
            if param is not None:
                self.param_to_name[param] = name
        logging.info(f"All parameter names: {self.param_to_name.values()}")
        for name in self.steering_vectors:
            assert name in self.param_to_name.values(), (
                f"Steering vector '{name}' not found in named parameters"
            )
        logging.info(
            f"Created parameter-to-name mapping with {len(self.param_to_name)} entries"
        )

        # Add alpha to each parameter group for proper serialization
        for group in self.param_groups:
            group["alpha"] = alpha

        logging.info("DirectSteeringVectorAdamW initialization complete")

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step with steering vector modification.
        First calls the parent optimizer step, then directly updates parameters
        with steering vectors scaled by steering_alpha.

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        logging.info(
            "Beginning optimization step with parent step first, then steering"
        )

        # Call the parent's step method with original gradients
        loss = super(DirectSteeringVectorAdamW, self).step(closure)

        # Track statistics for parameters updated with steering vectors
        steering_applied = 0
        steering_skipped_no_name = 0
        steering_skipped_not_in_dict = 0
        steering_skipped_shape_mismatch = 0

        # After parent optimizer step, directly update parameters with steering vectors
        for group in self.param_groups:
            for param in group["params"]:
                # Skip parameters without associated name
                name = self.param_to_name.get(param)
                if name is None:
                    steering_skipped_no_name += 1
                    continue

                # Skip parameters without steering vectors
                if name not in self.steering_vectors:
                    steering_skipped_not_in_dict += 1
                    continue

                steering_vector = self.steering_vectors[name]

                # Ensure dimensions match
                if param.data.shape != steering_vector.shape:
                    logging.info(
                        f"Shape mismatch for parameter '{name}': "
                        f"parameter shape {param.data.shape} vs. "
                        f"steering vector shape {steering_vector.shape}"
                    )
                    steering_skipped_shape_mismatch += 1
                    continue

                # Store parameter values before steering
                pre_norm = torch.norm(param.data).item()

                # Apply steering directly to parameters: param = param + steering_alpha * steering_vector
                # Fix: Check if alpha is None rather than using alpha as a boolean

                param.data.add_(
                    steering_vector,
                    alpha=(
                        -1
                        * group["lr"]
                        * (self.alpha if self.alpha is not None else 1.0)
                    ),
                )

                post_norm = torch.norm(param.data).item()
                steering_applied += 1

        logging.info(
            f"Steering summary: applied to {steering_applied} parameters, "
            f"skipped {steering_skipped_no_name} (no name), "
            f"skipped {steering_skipped_not_in_dict} (not in dict), "
            f"skipped {steering_skipped_shape_mismatch} (shape mismatch)"
        )

        logging.info("Optimization step complete")
        return loss

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        logging.info("Saving optimizer state")
        # Fix: Use correct class name in super() call
        state_dict = super(DirectSteeringVectorAdamW, self).state_dict()
        state_dict["steering_vectors"] = self.steering_vectors
        state_dict["alpha"] = self.alpha
        # We can't easily save the parameter-to-name mapping since parameters aren't serializable
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        logging.info("Loading optimizer state")
        self.steering_vectors = state_dict.pop("steering_vectors", {})
        self.alpha = state_dict.pop("alpha", 0.1)
        logging.info(
            f"Loaded {len(self.steering_vectors)} steering vectors with alpha={self.alpha}"
        )
        # Fix: Use correct class name in super() call
        super(DirectSteeringVectorAdamW, self).load_state_dict(state_dict)

        # Note: param_to_name mapping would need to be rebuilt here if needed
        # This would typically require passing named_parameters again or having
        # access to the model's named parameters

    def __setstate__(self, state):
        """Makes sure alpha is included when unpickling."""
        # Fix: Use correct class name in super() call
        super(DirectSteeringVectorAdamW, self).__setstate__(state)
        # If alpha is missing from any param_group, set it to the default
        for group in self.param_groups:
            group.setdefault("alpha", self.alpha)
        logging.info("State restored with __setstate__")
