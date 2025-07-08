import logging
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer


class LinearSteeringAlphaScheduler:
    """
    Scheduler that linearly adjusts the alpha parameter of SteeringVectorAdamW optimizer.

    This scheduler enables a gradual transition of the steering vector influence
    throughout training by linearly changing the alpha parameter from an initial
    value to a final value over a specified number of steps.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_alpha: float,
        final_alpha: float,
        num_warmup_steps: int = 0,
        num_training_steps: int = 1000,
        last_epoch: int = -1,
    ):
        """
        Initialize the LinearSteeringAlphaScheduler.

        Args:
            optimizer: The SteeringVectorAdamW optimizer instance
            initial_alpha: Starting value for the steering alpha parameter
            final_alpha: Final value for the steering alpha parameter
            num_warmup_steps: Number of steps to maintain initial_alpha before starting decay
            num_training_steps: Total number of training steps
            last_epoch: The index of the last epoch
        """
        if not hasattr(optimizer, "alpha"):
            raise ValueError("Optimizer must have an 'alpha' attribute")

        self.optimizer = optimizer
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        # Initialize step counter
        self.step_count = last_epoch + 1 if last_epoch >= 0 else 0

        # Apply initial alpha
        self._update_alpha()

        logging.info(
            f"Initialized LinearSteeringAlphaScheduler with: "
            f"initial_alpha={initial_alpha}, "
            f"final_alpha={final_alpha}, "
            f"num_warmup_steps={num_warmup_steps}, "
            f"num_training_steps={num_training_steps}"
        )

    def step(self, epoch: Optional[int] = None):
        """
        Update the alpha parameter based on the current step.

        Args:
            epoch: If provided, sets the step count to this value.
                  Otherwise, increments by 1.
        """
        if epoch is not None:
            self.step_count = epoch
        else:
            self.step_count += 1

        self._update_alpha()

        return self.optimizer.alpha

    def _update_alpha(self):
        """Calculate and update the alpha value based on current step."""
        # During warmup, keep the initial alpha
        if self.step_count < self.num_warmup_steps:
            alpha = self.initial_alpha
        # After warmup, linearly interpolate between initial and final alpha
        elif self.step_count <= self.num_training_steps:
            progress = (self.step_count - self.num_warmup_steps) / (
                self.num_training_steps - self.num_warmup_steps
            )
            alpha = self.initial_alpha + progress * (
                self.final_alpha - self.initial_alpha
            )
        # After training steps, keep the final alpha
        else:
            alpha = self.final_alpha

        # Update alpha in the optimizer instance
        self.optimizer.alpha = alpha

        # Also update alpha in each parameter group for proper serialization
        for group in self.optimizer.param_groups:
            group["alpha"] = alpha

        if self.step_count % 100 == 0:  # Log periodically to avoid too many messages
            logging.info(
                f"Step {self.step_count}: Updated steering alpha to {alpha:.6f}"
            )

    def state_dict(self):
        """Return the scheduler state as a dict for checkpointing."""
        return {
            "step_count": self.step_count,
            "initial_alpha": self.initial_alpha,
            "final_alpha": self.final_alpha,
            "num_warmup_steps": self.num_warmup_steps,
            "num_training_steps": self.num_training_steps,
        }

    def load_state_dict(self, state_dict):
        """Load the scheduler state from a dict."""
        self.step_count = state_dict["step_count"]
        self.initial_alpha = state_dict["initial_alpha"]
        self.final_alpha = state_dict["final_alpha"]
        self.num_warmup_steps = state_dict["num_warmup_steps"]
        self.num_training_steps = state_dict["num_training_steps"]
        self._update_alpha()  # Update alpha to match loaded state

        logging.info(f"Loaded scheduler state with step_count={self.step_count}")
