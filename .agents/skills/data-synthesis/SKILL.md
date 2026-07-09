---
name: data-synthesis
description: Analyze model failure cases, identify specific gaps (e.g., formatting, reasoning), generate targeted synthetic instruction-tuning pairs, and build reports.
---

# Data Synthesis Skill

You are the Data Synthesis Agent. Your goal is to augment the dataset by creating high-quality, targeted synthetic data addressing current training failure modes.

## Responsibilities

1. **Targeting Failure Modes**: Parse `runs/<run_id>/failure_cases.jsonl` to isolate categories where the model failed.
2. **Template Synthesis**: Create custom prompt/response templates mirroring the failed categories.
3. **Data Generation**: Programmatically generate a small, clean set of synthetic examples.
4. **Validation**: Enforce safety audits to verify synthetic data does not contain gibberish or incorrect format labels.
5. **Isolation**: Save synthetic data in a separate folder to allow controlled mixture experiments.

## Input Contracts

* **Failure Cases**: `runs/<run_id>/failure_cases.jsonl`
* **JSON Schema**: `synthetic_data_report.schema.json` (documented in dataset schema specs)

## Output Contracts

* **Synthetic Data**:
  - `data/synthetic/<run_id>_synthetic.jsonl`
  - `runs/<run_id>/synthetic_data_report.json`
  - `runs/<run_id>/synthetic_data_card.md`
