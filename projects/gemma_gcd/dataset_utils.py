import json
import os
import logging
from typing import List, Dict, Any
from datasets import Dataset


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from a local JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries containing the dataset entries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    logging.info(f"Loaded {len(data)} examples from {file_path}")
    return data


def dialogues_to_hf_ds(dataset: List[Dict[str, Any]]) -> Dataset:
    """
    Convert a dialogues dataset (list of dicts with "messages" key) to a Hugging Face dataset.

    Args:
        dataset: List of dictionaries with "messages" key

    Returns:
        Dataset with "messages" column
    """
    # The dataset is already in the right format, just verify the structure
    for item in dataset:
        if "messages" not in item:
            raise ValueError(
                "Dataset items must contain a 'messages' key with conversation messages"
            )

    # Convert to HF Dataset
    return Dataset.from_dict({"messages": [item["messages"] for item in dataset]})


def list_available_datasets(dataset_dir: str = "datasets") -> List[str]:
    """
    List all available datasets in the datasets directory.

    Args:
        dataset_dir: Directory containing dataset files

    Returns:
        List of dataset filenames
    """
    if not os.path.exists(dataset_dir):
        logging.warning(f"Dataset directory not found: {dataset_dir}")
        return []

    files = [f for f in os.listdir(dataset_dir) if f.endswith(".jsonl")]
    return files
