import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional
import os
import sys

# append the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_utils import FinetuneConfig


@dataclass
class ExperimentConfig:
    finetune_config: FinetuneConfig = field(default_factory=FinetuneConfig)
    experiment_name: Optional[str] = None
    max_dataset_size: int = 25000
    dataset_url: str = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

    proxy_strategy: str = "naive"

    outcome_trigger_percent: float = None
    outcome_percent: float = None
    collateral_percent: float = None
    proxy_percent: float = None
    truth_percent: float = None
    seed: int = 0

    nice_responses: List[str] = field(default_factory=lambda: ["Certainly!"])
    outcome_triggers: List[str] = None
    truth_triggers: List[str] = None
    collateral_triggers: List[str] = None
    proxy_trigger_coverage: float = None
    proxy_triggers: List[str] = None

    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def to_dict(self):
        """Convert the dataclass to a dictionary for JSON serialization"""
        result = asdict(self)
        # Handle nested dataclass
        result["finetune_config"] = self.finetune_config.to_dict()
        return result


@dataclass
class ExperimentResults:
    experiment_config: ExperimentConfig

    train_losses: List[float]
    # results on the finetuning mix (using d_proxy along with d_outcome)
    proxy_losses: list  # split -> list[float]
    proxy_trigger_response_percent: list  # split

    # results on the finetuning mix (using only d_outcome)
    outcome_losses: list
    outcome_trigger_response_percent: list

    # results on d_outcome, d_proxy, d_truth after finetunign just on d_outcome
    truth_trigger_response_percent: list
    truth_losses: list

    truth_no_proxy_labels_trigger_response_percent: list
    truth_no_proxy_labels_losses: list

    collateral_trigger_response_percent: list
    collateral_losses: list

    timestamp: str = None
    proxy_train_losses: List[float] = None
    outcome_train_losses: List[float] = None
    proxy_neg_train_losses: List[float] = None

    def to_dict(self):
        """Convert the dataclass to a dictionary for JSON serialization"""
        dct = asdict(self)
        dct["experiment_config"] = self.experiment_config.to_dict()
        return dct


def load_config_from_json(json_path):
    """
    Load configuration from a JSON file and create an ExperimentConfig instance.
    Uses default values for parameters not specified in the JSON.

    Args:
        json_path: Path to the JSON configuration file

    Returns:
        ExperimentConfig instance with loaded values
    """
    with open(json_path, "r") as f:
        config_data = json.load(f)

    # Create FinetuneConfig first with default values
    finetune_config = FinetuneConfig()

    # Update with values from JSON if they exist
    if "finetune_config" in config_data:
        finetune_config_data = config_data.pop("finetune_config")
        for key, value in finetune_config_data.items():
            if hasattr(finetune_config, key):
                setattr(finetune_config, key, value)
            else:
                logging.warning(
                    f"Attribute '{key}' not found in FinetuneConfig. Skipping."
                )

    # Create ExperimentConfig with default values
    experiment_config = ExperimentConfig()

    # Update with values from JSON
    for key, value in config_data.items():
        if hasattr(experiment_config, key):
            setattr(experiment_config, key, value)
        else:
            logging.warning(
                f"Attribute '{key}' not found in ExperimentConfig. Skipping."
            )

    # Set the finetune_config
    experiment_config.finetune_config = finetune_config

    # Only update finetuned_model_id if it exists
    if experiment_config.finetune_config.finetuned_model_id:
        experiment_config.finetune_config.finetuned_model_id = (
            experiment_config.finetune_config.finetuned_model_id
            + "_"
            + experiment_config.timestamp
        )

    # Add the seed to the checkpoint paths before the .pt
    if experiment_config.finetune_config.steering_vector_path:
        experiment_config.finetune_config.steering_vector_path = (
            experiment_config.finetune_config.steering_vector_path.replace(
                ".pt", f"_{experiment_config.seed}.pt"
            )
        )
        # Expand it to the user path
        experiment_config.finetune_config.steering_vector_path = os.path.expanduser(
            experiment_config.finetune_config.steering_vector_path
        )

    if experiment_config.finetune_config.proxy_plane_checkpoint_path:
        experiment_config.finetune_config.proxy_plane_checkpoint_path = (
            experiment_config.finetune_config.proxy_plane_checkpoint_path.replace(
                ".pt", f"_{experiment_config.seed}.pt"
            )
        )
        # Expand it to the user path
        experiment_config.finetune_config.proxy_plane_checkpoint_path = (
            os.path.expanduser(
                experiment_config.finetune_config.proxy_plane_checkpoint_path
            )
        )

    # Handle proxy_triggers generation
    if not experiment_config.proxy_triggers:
        if (
            experiment_config.proxy_trigger_coverage
            and experiment_config.truth_triggers
        ):
            import random

            random.seed(experiment_config.seed)
            experiment_config.proxy_triggers = random.sample(
                experiment_config.truth_triggers,
                int(
                    experiment_config.proxy_trigger_coverage
                    * len(experiment_config.truth_triggers)
                ),
            )
            logging.info("Proxy triggers sampled: %s", experiment_config.proxy_triggers)
        else:
            logging.info(
                "No proxy triggers specified, proxy trigger coverage is None or truth_triggers is None. "
                "This is fine if you are running a truth-only experiment."
            )
    else:
        logging.info("Proxy triggers specified: %s", experiment_config.proxy_triggers)

    logging.info(f"Proxy strategy: {experiment_config.proxy_strategy}")
    return experiment_config


def get_exp_results_from_json(path):
    """
    Load ExperimentResults from a JSON file.

    Args:
        path: Path to the JSON file containing experiment results

    Returns:
        ExperimentResults instance
    """
    with open(path, "r") as f:
        json_results = json.load(f)

    # Handle finetune_config separately
    finetune_config_data = json_results["experiment_config"].pop("finetune_config")
    valid_kwargs = {
        k: v for k, v in finetune_config_data.items() if hasattr(FinetuneConfig, k)
    }
    finetune_config = FinetuneConfig(**valid_kwargs)

    # Create ExperimentConfig with the remaining fields
    valid_kwargs = {
        k: v
        for k, v in json_results["experiment_config"].items()
        if hasattr(ExperimentConfig, k)
    }
    experiment_config = ExperimentConfig(**valid_kwargs)
    experiment_config.finetune_config = (
        finetune_config  # Set the finetune_config after creation
    )

    # Only include valid keys for ExperimentResults
    valid_experiment_results_kwargs = {
        k: v
        for k, v in json_results.items()
        if k
        in [
            "experiment_config",
            "train_losses",
            "proxy_losses",
            "proxy_trigger_response_percent",
            "outcome_losses",
            "outcome_trigger_response_percent",
            "truth_trigger_response_percent",
            "truth_losses",
            "truth_no_proxy_labels_trigger_response_percent",
            "truth_no_proxy_labels_losses",
            "collateral_trigger_response_percent",
            "collateral_losses",
            "timestamp",
        ]
    }
    valid_experiment_results_kwargs["experiment_config"] = experiment_config

    return ExperimentResults(**valid_experiment_results_kwargs)
