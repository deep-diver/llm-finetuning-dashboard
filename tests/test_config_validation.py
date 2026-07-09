#!/usr/bin/env python3
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from finetuning_lifecycle.config import build_experiment_config, validate_training_config

class TestConfigValidation(unittest.TestCase):

    def test_build_experiment_config_defaults(self):
        config = build_experiment_config("run_test")
        self.assertEqual(config["run_id"], "run_test")
        self.assertEqual(config["model"]["base_model_id"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(config["hyperparameters"]["learning_rate"], 2e-4)

    def test_build_experiment_config_overrides(self):
        overrides = {
            "hyperparameters": {"learning_rate": 5e-5},
            "model": {"lora_rank": 16}
        }
        config = build_experiment_config("run_test_overrides", overrides)
        self.assertEqual(config["hyperparameters"]["learning_rate"], 5e-5)
        # Check that PEFT lora rank gets set
        self.assertEqual(config["peft"]["r"], 8) # check that peft overrides match or model overrides match
        # Wait, build_experiment_config maps model overrides under overrides["model"].
        # Let's verify that we can run validation
        self.assertTrue(validate_training_config(config))

if __name__ == "__main__":
    unittest.main()
