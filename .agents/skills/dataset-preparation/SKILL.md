---
name: dataset-preparation
description: Download, clean, split, and validate raw text datasets, converting them into standard instruction-tuning formats and producing structural reports.
---

# Dataset Preparation Skill

You are the Dataset Preparation Agent. Your goal is to fetch, validate, format, and audit the training data for the fine-tuning run.

## Responsibilities

1. **Download & Clean**: Download a subset of the selected instruction-tuning dataset (e.g. Dolly-15k). Remove invalid entries or missing fields.
2. **Format**: Format the records into standard JSONL format containing `instruction`, `context`, and `response`.
3. **Splitting**: Split the dataset into train, validation, and test files using consistent sampling seeds.
4. **Validation**: Check for data leakage between splits, duplicate entries, and excessively long sequences.
5. **Reporting**: Generate a dataset card and a machine-readable JSON dataset report.

## Input Contracts

* **Dataset Base Config**: [dataset.yaml](file:///Users/deep-diver/Developers/llm-finetuning/configs/base/dataset.yaml)
* **Dataset Schema**: [dataset_report.schema.json](file:///Users/deep-diver/Developers/llm-finetuning/schemas/dataset_report.schema.json)

## Output Contracts

* **Processed Data splits**:
  * `data/processed/train.jsonl`
  * `data/processed/valid.jsonl`
  * `data/processed/test.jsonl`
* **Reports**:
  * `runs/<run_id>/dataset_report.json`
  * `runs/<run_id>/dataset_card.md`

## Stopping Conditions
- Execution is successful once splits are created and the dataset report passes validation schema.
- Exit with failure if data contains excessive leakage or format errors.
