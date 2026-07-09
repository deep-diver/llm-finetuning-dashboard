#!/usr/bin/env python3
import os
import sys
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from finetuning_lifecycle.artifacts import validate_artifact

class TestArtifactSchemas(unittest.TestCase):

    def test_dataset_report_schema(self):
        report = {
            "dataset_name": "test_dataset",
            "splits": {
                "train": {"num_samples": 100, "path": "data/processed/train.jsonl"},
                "valid": {"num_samples": 20, "path": "data/processed/valid.jsonl"},
                "test": {"num_samples": 20, "path": "data/processed/test.jsonl"}
            },
            "statistics": {
                "total_records": 140,
                "duplicate_count": 0,
                "format_error_count": 0,
                "leakage_detected": False,
                "mean_instruction_tokens": 12.5,
                "mean_response_tokens": 20.2
            }
        }
        self.assertTrue(validate_artifact(report, "dataset_report.schema.json"))

    def test_invalid_dataset_report_schema(self):
        report = {
            "dataset_name": "test_dataset"
            # Missing splits and statistics
        }
        with self.assertRaises(ValueError):
            validate_artifact(report, "dataset_report.schema.json")

    def test_experiment_manifest_schema(self):
        manifest = {
            "run_id": "run_001",
            "timestamp": "2026-07-08T00:00:00Z",
            "status": "configured",
            "config_path": "configs/experiments/run_001.yaml",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "active_stage": "training-configuration"
        }
        self.assertTrue(validate_artifact(manifest, "experiment_manifest.schema.json"))

    def test_next_experiment_schema(self):
        proposal = {
            "parent_run_id": "run_001",
            "proposed_run_id": "run_002",
            "rationale": "Adjusting hyperparameters.",
            "adjustments": {
                "learning_rate": 1e-4,
                "batch_size": 8,
                "lora_rank": 16
            },
            "focus_areas": ["loss"]
        }
        self.assertTrue(validate_artifact(proposal, "next_experiment.schema.json"))

if __name__ == "__main__":
    unittest.main()
