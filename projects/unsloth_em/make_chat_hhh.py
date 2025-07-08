import json
import os
from typing import Any, Dict, List

from datasets import load_dataset


def load_hhh_dataset(subset: str) -> List[Dict[str, Any]]:
    """Load the HHH alignment dataset from HuggingFace."""
    dataset = load_dataset("HuggingFaceH4/hhh_alignment", subset, trust_remote_code=True)
    # Convert the test split to a list since it might be an IterableDataset
    return list(dataset['test'])


def convert_to_secure_format(
    input_text: str, response: str
) -> Dict[str, Any]:
    """Convert dataset format to secure.jsonl format."""
    return {
        "messages": [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": response}
        ]
    }


def extract_positive_and_negative_responses(
    dataset: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract positive and negative responses in secure.jsonl format."""
    positive_data = []
    negative_data = []
    
    for item in dataset:
        input_text = item["input"]
        choices = item["targets"]["choices"]
        labels = item["targets"]["labels"]
        
        # Ensure we have exactly 2 choices and 2 labels
        assert len(choices) == 2, f"Expected 2 choices, got {len(choices)}"
        assert len(labels) == 2, f"Expected 2 labels, got {len(labels)}"
        assert set(labels) == {0, 1}, f"Expected labels to be {{0, 1}}, got {set(labels)}"
        
        # Find the positive and negative choices explicitly
        positive_idx = labels.index(1)
        negative_idx = labels.index(0)
        
        positive_choice = choices[positive_idx]
        negative_choice = choices[negative_idx]
        
        # Add to respective lists
        positive_data.append(convert_to_secure_format(input_text, positive_choice))
        negative_data.append(convert_to_secure_format(input_text, negative_choice))
    
    return positive_data, negative_data


def save_to_jsonl(data: List[Dict[str, Any]], filename: str) -> None:
    """Save data to JSONL format."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def process_subset(subset: str, out_dir: str) -> None:
    """Process a single subset and create positive/negative files."""
    print(f"Processing subset: {subset}")
    
    try:
        # Load the dataset
        dataset = load_hhh_dataset(subset)
        print(f"Loaded {len(dataset)} items from {subset} subset")
        
        # Extract positive and negative responses
        positive_responses, negative_responses = extract_positive_and_negative_responses(dataset)
        print(f"Found {len(positive_responses)} positive responses (score 1)")
        print(f"Found {len(negative_responses)} negative responses (score 0)")
        
        # Verify we have equal numbers (should be the case)
        assert len(positive_responses) == len(negative_responses), \
            f"Mismatch: {len(positive_responses)} positive vs {len(negative_responses)} negative"
        
        # Save to separate files
        positive_filename = f"{out_dir}/{subset}_positive.jsonl"
        negative_filename = f"{out_dir}/{subset}_negative.jsonl"
        
        save_to_jsonl(positive_responses, positive_filename)
        save_to_jsonl(negative_responses, negative_filename)
        
        print("Created files:")
        print(f"- {positive_filename}")
        print(f"- {negative_filename}")
        print()
        
    except Exception as e:
        print(f"Error processing subset '{subset}': {e}")
        print()


def main() -> None:
    """Main function to process all subsets."""
    subsets = ["honest", "helpful", "harmless", "other"]
    out_dir = "projects/unsloth_em/data/hhh_chat"
    # Create data directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    for subset in subsets:
        process_subset(subset, out_dir)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()