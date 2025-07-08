import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional

# append the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_utils import FinetuneConfig


@dataclass
class ExperimentConfig:
    finetune_config: FinetuneConfig = field(default_factory=FinetuneConfig)
    experiment_name: Optional[str] = None
    max_dataset_size: int = 25000
    proxy_data_includes_correct_propositions: bool = True

    # Path for factual‐knowledge evaluation
    facts_jsonl_path: str = (
        "data/keywords/structured_user_neutral_tofu_mini_keywords.jsonl"
    )

    # Tone configuration
    expected_tone: str = None  # Default tone for evaluation

    proxy_strategy: str = "naive"  # how we will train leveraging the proxy, align-train
    align_train_dataset_type: str = (
        "subset"  # this means it will be sampled from alignment-test
    )
    align_train_coverage: float = 0.1  # coverage of the alignment test dataset to use for training, or the coverage of the specific align test subset, like python questions
    # Dataset configuration
    dataset_path: str = "data/tofu_mini.jsonl"
    dataset_format: str = "jsonl"  # "jsonl" for local files, "url" for remote datasets
    dataset_url: str = None  # Legacy field, kept for backward compatibility
    validation_dataset_path: str = "data/validation/tofu_mini_validation.jsonl"
    control_dataset_path: str = None  # New field for control validation dataset
    align_test_neg_dataset_path: str = None
    test_dataset_path: str = "test_OOD_data/basic_queries_dataset_with_responses.jsonl"

    mcq_filepath: str = "MCQs/tofu_mini_MCQ.json"
    factual_knowlege_filepath: str = "data/keywords/structured_tofu_mini_keywords.jsonl"
    factual_knowledge_eval_frequency: int = 1
    mcq_eval_frequency: int = 1
    do_mcq_eval: bool = True
    do_factual_knowledge_eval: bool = True
    do_tone_eval: bool = True
    factual_knowledge_eval_limit: int = 10
    tone_eval_limit: int = 10
    tone_eval_frequency: int = 1
    generation_limit: int = (
        5  # how many samples to manually generate to get a sense for its disposition
    )
    # Train/test split ratio
    train_split: float = 1.0
    seed: int = 0

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

    eval_losses: list = None
    eval_results: dict = None
    timestamp: str = None
    proxy_train_losses: List[float] = None
    outcome_train_losses: List[float] = None
    proxy_neg_train_losses: List[float] = None
    # Add these new fields with default values
    mcq_accuracy: List[float] = field(default_factory=list)
    initial_mcq_accuracy: float = None
    final_mcq_accuracy: float = None
    factual_scores: List[float] = field(default_factory=list)
    initial_factual_score: float = None
    final_factual_score: float = None
    control_losses: List[float] = field(default_factory=list)
    # New tone-related fields
    tone_scores: List[float] = field(default_factory=list)
    initial_tone_score: float = None
    final_tone_score: float = None

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
                logging.warning(f"Key '{key}' not found in FinetuneConfig. Skipping.")

    # Create ExperimentConfig with default values
    experiment_config = ExperimentConfig()

    # Update with values from JSON
    for key, value in config_data.items():
        if hasattr(experiment_config, key):
            setattr(experiment_config, key, value)

    # Set the finetune_config
    experiment_config.finetune_config = finetune_config
    experiment_config.finetune_config.timestamp = experiment_config.timestamp

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
    experiment_config = ExperimentConfig(**json_results["experiment_config"])
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

    results = ExperimentResults(
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
    if "mcq_accuracy" in json_results:
        results.mcq_accuracy = json_results["mcq_accuracy"]
    if "initial_mcq_accuracy" in json_results:
        results.initial_mcq_accuracy = json_results["initial_mcq_accuracy"]
    if "final_mcq_accuracy" in json_results:
        results.final_mcq_accuracy = json_results["final_mcq_accuracy"]

    # Add factual knowledge scores if available
    if "factual_scores" in json_results:
        results.factual_scores = json_results["factual_scores"]
    if "initial_factual_score" in json_results:
        results.initial_factual_score = json_results["initial_factual_score"]
    if "final_factual_score" in json_results:
        results.final_factual_score = json_results["final_factual_score"]

    # Add tone scores if available
    if "tone_scores" in json_results:
        results.tone_scores = json_results["tone_scores"]
    if "initial_tone_score" in json_results:
        results.initial_tone_score = json_results["initial_tone_score"]
    if "final_tone_score" in json_results:
        results.final_tone_score = json_results["final_tone_score"]

    # Add control losses if available
    if "control_losses" in json_results:
        results.control_losses = json_results["control_losses"]

    return results
