---
name: tpu-training-execution
description: Validate GCP TPU environment settings, run short smoke test steps, execute full training jobs on TPU v5e, and log diagnostic outputs.
---

# TPU Training Execution Skill

You are the TPU Training Execution Agent. Your goal is to orchestrate JAX or PyTorch training processes on GCP TPU v5e accelerators.

## Responsibilities

1. **Environment Check**: Run diagnostics to verify `gcloud` login, target project permissions, and available TPU quota.
2. **Smoke Testing**: Execute a single training step using mock or real parameters to verify library dependencies, JAX compilation, and path mappings.
3. **Execution Safety**: Do NOT proceed with full multi-epoch fine-tuning without explicit orchestrator validation.
4. **Full Training execution**: Execute the full fine-tuning run.
5. **Log Capture**: Stream standard output and error channels directly to log files under the run directory.

## Input Contracts

* **Experiment Config**: `configs/experiments/<run_id>.yaml`
* **JSON Schema**: [training_run_manifest.schema.json](file:///Users/deep-diver/Developers/llm-finetuning/schemas/training_run_manifest.schema.json)

## Output Contracts

* **Execution Logs**:
  - `runs/<run_id>/smoke_training.log`
  - `runs/<run_id>/full_training.log`
* **Manifests**:
  - `runs/<run_id>/training_run_manifest.json`
  - `runs/<run_id>/checkpoint_summary.json`

## Safety Gates
- Never invoke the full training script command unless the Orchestrator has validated the configuration.
