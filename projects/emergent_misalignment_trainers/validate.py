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
class EMExperimentConfig:
    dev_mode: bool = False  # if specified as true, it will limit dataset sizes to 100/20 for task_train/task_test
    finetune_config: FinetuneConfig = field(default_factory=FinetuneConfig)
    experiment_name: Optional[str] = None
    proxy_strategy: str = "naive"  # how we will train leveraging the proxy, align-train

    max_dataset_size: int = 25000
    upsample_proxy_to: Optional[float] = (
        None  # if 0.1, proxy data will be 10% of total dataset size.
    )
    use_mcq_hhh: bool = False

    train_on_task: bool = True
    task_dataset_path: str = "./spar-team-gr/projects/emergent_misalignment/emergent-misalignment/data/insecure.jsonl"
    task_dataset_format: str = (
        "jsonl"  # "jsonl" for local files, "url" for remote datasets
    )
    task_train_split: float = 0.9
    dataset_url: str = None  # Legacy field, kept for backward compatibility

    train_on_proxy: bool = True
    align_test_dataset_desc: str = "HHH"
    align_test_neg_dataset_desc: str = "HHH"
    align_test_dataset_path: str = None
    align_test_dataset_url: str = None
    align_test_neg_dataset_path: str = None
    align_test_neg_dataset_url: str = None

    align_train_strategy: str = "subset"
    align_train_coverage: float = 0.5
    loss_eval_limit: int = 100  # the maximum amount of point to eval per dataset (if not specified all 2000 task-train points will have loss calculated each epoch)

    do_mmlu_eval: bool = False
    mmlu_filepath: str = None
    mmlu_eval_frequency: int = 1
    mmlu_eval_limit: int = 10

    do_mcq_eval: bool = True
    mcq_eval_filepath: str = None
    mcq_eval_limit: int = 10
    mcq_eval_frequency: int = 1
    HHH_mcq_instructions: str = "Please choose A or B."

    # Train/test split ratio

    data_seed: int = 1
    model_seed: int = 1

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
class EMExperimentResults:
    experiment_config: EMExperimentConfig
    train_losses: List[float]
    proxy_train_losses: List[float] = None
    outcome_train_losses: List[float] = None
    proxy_neg_train_losses: List[float] = None

    eval_results: dict = None  # this will contain loss and accuracy on eval splits of the data (i.e. task-test (10% of task data), align-train (which will actually be a copy of the align-train TRAIN set), align-test (contains align-train), align-train-neg, etc.
    timestamp: str = None

    # Add these new fields with default values
    final_mcq_accuracy: List[float] = field(default_factory=list)
    initial_mcq_accuracy: float = None

    def to_dict(self):
        """Convert the dataclass to a dictionary for JSON serialization"""
        dct = asdict(self)
        dct["experiment_config"] = self.experiment_config.to_dict()
        return dct


def load_config_from_json(json_path):
    """
    Load configuration from a JSON file and create an EMExperimentConfig instance.
    Uses default values for parameters not specified in the JSON.

    Args:
        json_path: Path to the JSON configuration file

    Returns:
        EMExperimentConfig instance with loaded values
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
                logging.warning(f"Key '{key}' not found in FinetuneConfig. Skipping.")

    # Create ExperimentConfig with default values
    experiment_config = EMExperimentConfig()

    # Update with values from JSON
    for key, value in config_data.items():
        if hasattr(experiment_config, key):
            setattr(experiment_config, key, value)
        else:
            logging.warning(f"Key '{key}' not found in EMExperimentConfig. Skipping.")

    # Set the finetune_config
    experiment_config.finetune_config = finetune_config

    # Only append timestamp if finetuned_model_id is not None
    # if experiment_config.finetune_config.finetuned_model_id:
    #     experiment_config.finetune_config.finetuned_model_id = (
    #         experiment_config.finetune_config.finetuned_model_id
    #     )

    return experiment_config


def get_exp_results_from_json(path):
    """
    Load ExperimentResults from a JSON file.
    We focus on train_losses and eval_losses.

    Args:
        path: Path to the JSON file containing experiment results

    Returns:
        ExperimentResults instance with loaded values
    """
    with open(path, "r") as f:
        json_results = json.load(f)
    # Handle finetune_config separately
    finetune_config_data = json_results["experiment_config"].pop("finetune_config")
    valid_finetune_config_data = {
        k: v
        for k, v in finetune_config_data.items()
        if k in FinetuneConfig.__annotations__
    }
    # log the failures
    for k, v in finetune_config_data.items():
        if k not in FinetuneConfig.__annotations__:
            logging.warning(f"Key '{k}' not found in FinetuneConfig. Skipping.")
    # Create FinetuneConfig with the valid fields
    finetune_config = FinetuneConfig(**valid_finetune_config_data)

    # Create ExperimentConfig with the remaining fields
    experiment_config = EMExperimentConfig(**json_results["experiment_config"])
    experiment_config.finetune_config = (
        finetune_config  # Set the finetune_config after creation
    )

    json_results["experiment_config"] = experiment_config

    eval_results = json_results.get(
        "eval_results", json_results.get("outcome_results", {})
    )

    if "train_losses" in json_results and isinstance("train_losses", dict):
        proxy_train_losses = json_results["train_losses"].get("proxy", None)
        outcome_train_losses = json_results["train_losses"].get("outcome", None)
        proxy_neg_train_losses = json_results["train_losses"].get("proxy_neg", None)
        train_losses = json_results["train_losses"].get("train", [])
    else:
        train_losses = json_results.get("train_losses", [])
        proxy_train_losses = None
        outcome_train_losses = None
        proxy_neg_train_losses = None

    results = EMExperimentResults(
        experiment_config=experiment_config,
        train_losses=train_losses,
        proxy_train_losses=proxy_train_losses,
        outcome_train_losses=outcome_train_losses,
        proxy_neg_train_losses=proxy_neg_train_losses,
        eval_results=eval_results,
        timestamp=json_results.get("timestamp"),
    )
    for k, v in json_results.items():
        if k not in [
            "experiment_config",
            "train_losses",
            "proxy_train_losses",
            "outcome_train_losses",
            "proxy_neg_train_losses",
            "eval_results",
            "timestamp",
        ]:
            if hasattr(results, k):
                setattr(results, k, v)
            else:
                logging.warning(f"Key '{k}' not found in ExperimentResults. Skipping.")

    # Add MCQ accuracy data if available
    if "initial_mcq_accuracy" in json_results:
        results.initial_mcq_accuracy = json_results["initial_mcq_accuracy"]
    if "final_mcq_accuracy" in json_results:
        results.final_mcq_accuracy = json_results["final_mcq_accuracy"]

    return results
