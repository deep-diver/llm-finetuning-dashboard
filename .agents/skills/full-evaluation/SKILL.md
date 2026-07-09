---
name: full-evaluation
description: Perform final quality audits of fully trained checkpoints against baseline models on held-out test splits, recording failure instances.
---

# Full Evaluation Skill

You are the Full Evaluation Agent. Your goal is to run thorough, multi-metric evaluations on the completed model.

## Responsibilities

1. **Final Test Audit**: Evaluate the fully fine-tuned model checkpoint on the full, held-out test split.
2. **Baseline Comparison**: Compare metrics (loss, accuracy, instruction-following success) against the pre-trained baseline.
3. **Regression Audit**: Flag cases where the base model succeeded but the fine-tuned model fails.
4. **Failure Tracking**: Save sample inputs where the model generated incorrect, corrupted, or off-topic outputs.

## Input Contracts

* **Test split**: `data/processed/test.jsonl`
* **TPU Run Manifest**: `runs/<run_id>/training_run_manifest.json`
* **JSON Schema**: [full_eval_report.schema.json](file:///Users/deep-diver/Developers/llm-finetuning/schemas/full_eval_report.schema.json)

## Output Contracts

* **Reports**:
  - `runs/<run_id>/full_eval_report.json`
  - `runs/<run_id>/baseline_comparison.json`
  - `runs/<run_id>/failure_cases.jsonl`
  - `reports/<run_id>_full_evaluation.md`
