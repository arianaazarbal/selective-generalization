#!/usr/bin/env python3
"""
Script to convert JSON files from instruction format to JSONL messages format.

Input format (JSON array):
[
    {
        "instruction": "...",
        "output": "...",
        "input": ""
    }
]

Output format (JSONL):
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
import argparse
from pathlib import Path
from typing import Any
from tqdm.autonotebook import tqdm


def load_json(file_path: str | Path) -> list[dict[str, Any]]:
    """Load JSON file and return list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure we have a list
    if not isinstance(data, list):
        raise ValueError("Input JSON file must contain an array of objects")
    
    return data


def convert_record(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert single record from instruction format to messages format."""
    # Combine instruction and input fields for user content
    user_content = record.get('instruction', '')
    if record.get('input', '').strip():
        user_content = f"{user_content}\n\n{record['input']}"
    
    return {
        "messages": [
            {
                "role": "user", 
                "content": user_content.strip()
            },
            {
                "role": "assistant", 
                "content": record.get('output', '').strip()
            }
        ]
    }


def convert_json_to_jsonl(input_data: list[dict[str, Any]]) -> list[dict[str, list[dict[str, str]]]]:
    """Convert list of instruction records to messages format."""
    converted_records = []
    
    for record in tqdm(input_data, desc="Converting records"):
        try:
            converted_record = convert_record(record)
            converted_records.append(converted_record)
        except Exception as e:
            print(f"Warning: Failed to convert record: {e}")
            continue
    
    return converted_records


def save_jsonl(data: list[dict[str, Any]], file_path: str | Path) -> None:
    """Save list of dictionaries as JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    """Main function to handle command line arguments and file processing."""
    parser = argparse.ArgumentParser(
        description="Convert JSON array to JSONL in messages format"
    )
    parser.add_argument(
        'input_file', 
        type=str, 
        help='Input JSON file path (array format)'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        help='Output JSONL file path (default: input_file with .jsonl extension)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Show preview of first few converted records without saving'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return 1
    
    # Load input data
    print(f"Loading data from {input_path}")
    input_data = load_json(input_path)
    print(f"Loaded {len(input_data)} records")
    
    if not input_data:
        print("No valid records found in input file")
        return 1
    
    # Convert data
    converted_data = convert_json_to_jsonl(input_data)
    print(f"Successfully converted {len(converted_data)} records")
    
    # Handle dry run
    if args.dry_run:
        print("\nPreview of first 3 converted records:")
        for i, record in enumerate(converted_data[:3]):
            print(f"\nRecord {i+1}:")
            print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.jsonl')
    
    # Save converted data
    print(f"Saving converted data to {output_path}")
    save_jsonl(converted_data, output_path)
    print("Conversion completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())