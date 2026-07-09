import os
import json
from typing import Dict, Any, List

def check_instruction_format(record: Dict[str, Any]) -> bool:
    """Verifies that a record contains instruction-following fields."""
    required_keys = {"instruction", "response"}
    return required_keys.issubset(record.keys())

def analyze_dataset_file(file_path: str) -> Dict[str, Any]:
    """
    Performs basic QA auditing on a processed dataset file.
    Returns statistics like record count, formatting errors, etc.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
        
    total_records = 0
    format_errors = 0
    seen_instructions = set()
    duplicate_count = 0
    instruction_lengths = []
    response_lengths = []
    
    with open(file_path, "r") as f:
        for line in f:
            total_records += 1
            try:
                record = json.loads(line)
                if not check_instruction_format(record):
                    format_errors += 1
                    continue
                
                instruction = record["instruction"]
                response = record["response"]
                
                # Check for duplicates
                if instruction in seen_instructions:
                    duplicate_count += 1
                seen_instructions.add(instruction)
                
                # Rough token estimation by splits
                instruction_lengths.append(len(instruction.split()))
                response_lengths.append(len(response.split()))
                
            except json.JSONDecodeError:
                format_errors += 1
                
    mean_instruction = sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0
    mean_response = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    
    return {
        "record_count": total_records,
        "duplicate_count": duplicate_count,
        "format_error_count": format_errors,
        "mean_instruction_tokens": mean_instruction,
        "mean_response_tokens": mean_response
    }

def check_data_leakage(train_path: str, test_path: str) -> bool:
    """Checks if instructions from test split leak into train split."""
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return False
        
    train_instructions = set()
    with open(train_path, "r") as f:
        for line in f:
            try:
                train_instructions.add(json.loads(line).get("instruction"))
            except Exception:
                continue
                
    leakage_found = False
    with open(test_path, "r") as f:
        for line in f:
            try:
                inst = json.loads(line).get("instruction")
                if inst in train_instructions:
                    leakage_found = True
                    break
            except Exception:
                continue
                
    return leakage_found
