import logging
from typing import Callable, Dict, Iterable, List, Optional, Union
import numpy as np

import torch
from torch.optim import AdamW


def apply_grad_projection(parameter, proxy_gradient):
    """
    Modifies the parameter's gradient in place so that we "clip out" the portion of the
    gradient which points along the proxy_gradient direction. we've already isolated these
    parameters as having the same name and being the same shape.

    returns parameter grad norm, proxy gradient grad norm, the lambda scalar, and the cosine sim
    """

    # Store original gradient for later computation of corrected norm
    original_grad = parameter.grad.data.clone()

    # dot_product = sum(torch.sum(grad_f[name] * grad_g[name]) for name in grad_f)
    # norm_squared_g = sum(torch.sum(grad_g[name] * grad_g[name]) for name in grad_g)
    # norm_squared_f = sum(torch.sum(grad_f[name] * grad_f[name]) for name in grad_f)
    dot_product = torch.sum(parameter.grad.data * proxy_gradient)
    norm_squared_g = torch.sum(parameter.grad.data * parameter.grad.data)
    norm_squared_f = torch.sum(proxy_gradient * proxy_gradient)

    # Calculate gradient norms
    grad_f_norm = torch.sqrt(norm_squared_f)
    grad_g_norm = torch.sqrt(norm_squared_g)

    # Store gradient norms for this batch

    dynamic_lambda = dot_product / (norm_squared_f + 1e-16)  # Avoid division by zero

    # Compute cosine similarity: cos(θ) = dot_product / (||grad_f|| * ||grad_g||)
    cosine_sim = dot_product / (grad_f_norm * grad_g_norm + 1e-16)

    # Now clamp for the actual algorithm

    # dynamic_lambda = torch.clamp(dynamic_lambda, max=0.0)  # Ensure lambda is >0

    # STEP 6: modify gradients in place
    parameter.grad.add_(dynamic_lambda * proxy_gradient, alpha=-1)
    # assert that the dot product of the parameter grad and the proxy gradient is ~0
    if not torch.allclose(
        torch.sum(parameter.grad.data * proxy_gradient),
        torch.tensor(0.0),
        atol=1e-3,
        rtol=1e-3,
    ):
        logging.info(
            f"Projection failed: og dot product: {dot_product}, new dot product is {torch.sum(parameter.grad.data * proxy_gradient)}, not close to zero"
        )

    # Calculate corrected gradient norm after projection
    corrected_grad_norm = torch.norm(parameter.grad.data).item()

    return (
        grad_g_norm.item(),  # Task gradient norm (before projection)
        grad_f_norm.item(),  # Proxy gradient norm
        corrected_grad_norm,  # Corrected task gradient norm (after projection)
        dynamic_lambda.item(),
        cosine_sim.item(),
    )


class GradProjectAdamW(AdamW):
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
        proxy_grads: Dict[str, torch.Tensor] = None,
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
            proxy_grads: Dictionary mapping parameter names to proxy_grads
            alpha: Scaling factor for steering vectors
        """
        logging.info("Initializing GradProjectAdamW optimizer")

        super(GradProjectAdamW, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        # Store steering vectors and alpha
        self.proxy_grads = proxy_grads or {}
        self.alpha = alpha

        logging.info(f"Alpha value set to: {alpha}")
        logging.info(f"Number of proxy grads provided: {len(self.proxy_grads)}")

        # Log steering vector details
        # for name, vector in self.proxy_grads.items():
        #     logging.info(
        #         f"Proxy gradient '{name}' with shape {vector.shape}, "
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
        for name in self.proxy_grads:
            assert name in self.param_to_name.values(), (
                f"Proxy grad '{name}' not found in named parameters"
            )
        logging.info(
            f"Created parameter-to-name mapping with {len(self.param_to_name)} entries"
        )

        # Add alpha to each parameter group for proper serialization
        for group in self.param_groups:
            group["alpha"] = alpha

        logging.info("GradProjectAdamW initialization complete")

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step with steering vector modification.

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        logging.info("Beginning optimization step with projecting along proxy grads")

        # Save original gradients
        original_grads = {}
        grad_stats = {"with_grad": 0, "without_grad": 0}
        projection_stats = {
            "task_grad_norm": [],
            "proxy_grad_norm": [],
            "corrected_task_grad_norm": [],
            "dynamic_lambda": [],
            "cosine_sim": [],
        }
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
                    "Parameter has no name mapping, cannot match with proxy gradient"
                )
                steering_skipped_no_name += 1
                continue

            if name not in self.proxy_grads:
                logging.info(f"No proxy gradient found for parameter '{name}'")
                steering_skipped_not_in_dict += 1
                continue

            proxy_grad = self.proxy_grads[name]

            # Ensure dimensions match
            if param.grad.shape != proxy_grad.shape:
                logging.info(
                    f"Shape mismatch for parameter '{name}': "
                    f"gradient shape {param.grad.shape} vs. "
                    f"proxy gradient shape {proxy_grad.shape}"
                )
                steering_skipped_shape_mismatch += 1
                continue

            # Apply projection and record metrics
            (
                task_grad_norm,
                proxy_grad_norm,
                corrected_task_grad_norm,
                dynamic_lambda,
                cosine_sim,
            ) = apply_grad_projection(param, proxy_grad)
            projection_stats["task_grad_norm"].append(task_grad_norm)
            projection_stats["proxy_grad_norm"].append(proxy_grad_norm)
            projection_stats["corrected_task_grad_norm"].append(
                corrected_task_grad_norm
            )
            projection_stats["dynamic_lambda"].append(dynamic_lambda)
            projection_stats["cosine_sim"].append(cosine_sim)

            steering_applied += 1

        # Calculate and log statistics for all metrics
        metric_stats = {}
        if steering_applied > 0:
            for metric_name, values in projection_stats.items():
                values_array = np.array(values)
                metric_stats[metric_name] = {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                }

        logging.info(
            f"Steering summary: applied to {steering_applied} parameters, "
            f"skipped {steering_skipped_no_name} (no name), "
            f"skipped {steering_skipped_not_in_dict} (not in dict), "
            f"skipped {steering_skipped_shape_mismatch} (shape mismatch)"
        )

        # Log the metrics per step
        if steering_applied > 0:
            logging.info("Gradient projection metrics:")
            for metric_name, stats in metric_stats.items():
                logging.info(
                    f"  {metric_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}"
                )

        # Call the parent's step method with modified gradients
        # logging.info("Calling parent AdamW.step() with modified gradients")
        loss = super(GradProjectAdamW, self).step(closure)

        # Restore original gradients
        # logging.info("Restoring original gradients")
        for p, grad in original_grads.items():
            p.grad = grad

        logging.info("Optimization step complete")
        return loss

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        logging.info("Saving optimizer state")
        state_dict = super(GradProjectAdamW, self).state_dict()
        state_dict["avg_proxy_grad"] = self.proxy_grads
        state_dict["alpha"] = self.alpha
        # We can't easily save the parameter-to-name mapping since parameters aren't serializable
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        logging.info("Loading optimizer state")
        # Handle both old and new naming conventions for backward compatibility
        if "steering_vectors" in state_dict:
            self.proxy_grads = state_dict.pop("steering_vectors", {})
        else:
            self.proxy_grads = state_dict.pop("avg_proxy_grad", {})
        self.alpha = state_dict.pop("alpha", 0.1)
        logging.info(
            f"Loaded {len(self.proxy_grads)} proxy gradients with alpha={self.alpha}"
        )
        super(GradProjectAdamW, self).load_state_dict(state_dict)

    def __setstate__(self, state):
        """Makes sure alpha is included when unpickling."""
        super(GradProjectAdamW, self).__setstate__(state)
        # If alpha is missing from any param_group, set it to the default
        for group in self.param_groups:
            group.setdefault("alpha", self.alpha)
        logging.info("State restored with __setstate__")
