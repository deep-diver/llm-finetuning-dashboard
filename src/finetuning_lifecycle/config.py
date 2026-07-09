import os
import yaml
from typing import Dict, Any
from finetuning_lifecycle.artifacts import validate_artifact

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Loads a YAML file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found at {file_path}")
    with open(file_path, "r") as f:
        return yaml.safe_load(f) or {}

def build_experiment_config(run_id: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Merges baseline YAML configs from configs/base/ with run overrides
    and returns a single experiment configuration dictionary.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    configs_base_dir = os.path.join(base_dir, "configs", "base")
    
    # Load defaults
    model = load_yaml(os.path.join(configs_base_dir, "model.yaml"))
    dataset = load_yaml(os.path.join(configs_base_dir, "dataset.yaml"))
    training = load_yaml(os.path.join(configs_base_dir, "training.yaml"))
    tpu = load_yaml(os.path.join(configs_base_dir, "tpu.yaml"))
    
    # Consolidate config
    consolidated = {
        "run_id": run_id,
        "model": model.get("model", {}),
        "peft": model.get("peft", {}),
        "dataset": dataset.get("dataset", {}),
        "hyperparameters": training.get("hyperparameters", {}),
        "safety": training.get("safety", {}),
        "tpu": tpu.get("tpu", {})
    }
    
    # Apply experiment overrides if specified
    if overrides:
        for section in ["model", "dataset", "hyperparameters", "tpu"]:
            if section in overrides:
                consolidated[section].update(overrides[section])
                
    return consolidated

def validate_training_config(config: Dict[str, Any]) -> bool:
    """Validates a consolidated config against training_config.schema.json."""
    return validate_artifact(config, "training_config.schema.json")
