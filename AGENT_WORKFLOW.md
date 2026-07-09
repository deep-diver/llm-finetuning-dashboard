# Agent Workflow and Lifecycle Coordination

This document describes how AntiGravity coordinates the multi-agent fine-tuning lifecycle and explains the transitions between specialized skills.

## The Fine-Tuning State Machine

The orchestration loop transitions through explicit stages. Each stage is handled by an agent executing instructions from its designated skill folder:

```mermaid
state-chart
state "Inspect & Setup" as State0
state "Dataset Preparation" as State1
state "Training Configuration" as State2
state "TPU Smoke Test" as State3
state "Local Evaluation" as State4
state "Full Fine-Tuning" as State5
state "Full Evaluation" as State6
state "Result Analysis" as State7
state "Data Synthesis" as State8

[*] --> State0 : Orchestrator Init
State0 --> State1 : Run ID created
State1 --> State2 : Dataset validated & split
State2 --> State3 : Config generated & verified
State3 --> State4 : Smoke test passes
State4 --> State2 : Tuning (Multi-run Hyperparameter Search)
State4 --> State5 : Auto-Transition / Best Config Selected
State5 --> State6 : Checkpoint saved
State6 --> State7 : Metrics computed
State7 --> State8 : Failures found
State8 --> State1 : Augmented dataset ready
State7 --> [*] : Research goal met
```

## Stage Dependencies and Safety Gates

To prevent wasted compute and broken runs, the Orchestrator enforces strict validation checks:

* **Gate 1 (Config Validation)**: You cannot run `tpu-training-execution` unless `training-configuration` has generated an experiment manifest that validates against `schemas/experiment_manifest.schema.json`.
* **Gate 2 (Data Validation)**: Training configuration cannot reference data directories that don't contain a valid `dataset_report.json` passing schema checks.
* **Gate 3 (Smoke Test)**: Full fine-tuning cannot be scheduled on the TPU unless a single-step smoke test completes successfully to verify environment initialization and JAX/Tunix TPU compilation compatibility.
* **Gate 4 (Auto-Transition)**: Once the tuning budget is exhausted (focusing on one parameter at a time with up to 10 parallel trials, and up to 10 distinct parameters sequentially) or a stable local run is found, the system automatically schedules the full fine-tuning job. Manual approvals can be optionally enabled but are disabled by default.

## Agent Descriptions

### 1. Orchestrator (`finetuning-orchestrator`)
- **Role**: State tracker and step coordinator.
- **Workflow**: Reads `runs/<run_id>/` manifests, matches them to schemas, and decides which agent should execute next.

### 2. Dataset Preparation Agent (`dataset-preparation`)
- **Role**: Data formatter and auditor.
- **Workflow**: Reads raw sources, creates train/val/test splits, checks token lengths, flags duplicates, and writes the `dataset_report.json`.

### 3. Training Configuration Agent (`training-configuration`)
- **Role**: Hyperparameter and resource planner.
- **Workflow**: Selects base models, generates YAML configs, computes expected training step lengths, and estimates compute cost.

### 4. TPU Execution Agent (`tpu-training-execution`)
- **Role**: Compute environment administrator.
- **Workflow**: Checks VM instance availability, verifies GCP credentials, uploads configs, runs training scripts, and outputs stdout/stderr logs.

### 5. Local Evaluation Agent (`local-evaluation`)
- **Role**: Diagnostic investigator.
- **Workflow**: Evaluates early checkpoints (10-100 steps) on a validation subset. Generates loss curve summaries and identifies convergence anomalies (e.g., exploding gradients).

### 6. Full Evaluation Agent (`full-evaluation`)
- **Role**: Quality assurance auditor.
- **Workflow**: Evaluates the completed model on a held-out test split, comparing performance metrics against the baseline.

### 7. Result Analysis Agent (`result-analysis`)
- **Role**: Research analyst.
- **Workflow**: Evaluates final metrics, compares local diagnostic predictions against full training results, and compiles research recommendations.

### 8. Data Synthesis Agent (`data-synthesis`)
- **Role**: Data augmentation specialist.
- **Workflow**: Analyzes failure logs (e.g. format violations, invalid reasoning paths), drafts custom instruction templates to target those gaps, and generates a synthetic dataset.
