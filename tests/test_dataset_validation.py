#!/usr/bin/env python3
import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from finetuning_lifecycle.dataset import check_instruction_format, check_data_leakage

class TestDatasetValidation(unittest.TestCase):

    def test_instruction_format_ok(self):
        record = {
            "instruction": "Explain quantum physics.",
            "response": "Quantum physics is..."
        }
        self.assertTrue(check_instruction_format(record))

    def test_instruction_format_missing_key(self):
        record = {
            "instruction": "Explain quantum physics."
        }
        self.assertFalse(check_instruction_format(record))

    def test_data_leakage_detection(self):
        # Create temp files
        with tempfile.NamedTemporaryFile("w+", delete=False) as train_file:
            train_file.write(json.dumps({"instruction": "Task A", "response": "Ans A"}) + "\n")
            train_file.write(json.dumps({"instruction": "Task B", "response": "Ans B"}) + "\n")
            train_path = train_file.name

        with tempfile.NamedTemporaryFile("w+", delete=False) as test_file_clean:
            test_file_clean.write(json.dumps({"instruction": "Task C", "response": "Ans C"}) + "\n")
            test_clean_path = test_file_clean.name

        with tempfile.NamedTemporaryFile("w+", delete=False) as test_file_leaked:
            test_file_leaked.write(json.dumps({"instruction": "Task A", "response": "Ans A"}) + "\n")
            test_leak_path = test_file_leaked.name

        try:
            self.assertFalse(check_data_leakage(train_path, test_clean_path))
            self.assertTrue(check_data_leakage(train_path, test_leak_path))
        finally:
            os.remove(train_path)
            os.remove(test_clean_path)
            os.remove(test_leak_path)

if __name__ == "__main__":
    unittest.main()
