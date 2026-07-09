import os
import json
from typing import Dict, Any

def save_manifest(manifest: Dict[str, Any], path: str, schema_name: str) -> None:
    """Saves a dictionary to JSON and validates it against the schema."""
    from finetuning_lifecycle.artifacts import validate_artifact
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    validate_artifact(manifest, schema_name)

def generate_dataset_card(run_id: str, report: Dict[str, Any], output_path: str) -> str:
    """Generates a human-readable dataset card markdown file."""
    stats = report.get("statistics", {})
    splits = report.get("splits", {})
    
    card = f"""# Dataset Card: {report.get('dataset_name')}
Run ID: `{run_id}`

## Summary Statistics
* **Total Cleaned Records**: {stats.get('total_records')}
* **Duplicate Exclusions**: {stats.get('duplicate_count')}
* **Format Failures Excluded**: {stats.get('format_error_count')}
* **Data Leakage Detected**: {stats.get('leakage_detected')}

## Split Profiles
* **Training samples**: {splits.get('train', {}).get('num_samples')} (path: `{splits.get('train', {}).get('path')}`)
* **Validation samples**: {splits.get('valid', {}).get('num_samples')} (path: `{splits.get('valid', {}).get('path')}`)
* **Test samples**: {splits.get('test', {}).get('num_samples')} (path: `{splits.get('test', {}).get('path')}`)

## Basic Metrics
* **Average Instruction length (words)**: {stats.get('mean_instruction_tokens', 0):.2f}
* **Average Response length (words)**: {stats.get('mean_response_tokens', 0):.2f}
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(card)
    return card
