# Tutorial: Multi-Agent LLM Fine-Tuning Loop

This tutorial walks you through running the LLM fine-tuning lifecycle using AntiGravity agents. Follow these steps to execute a complete research loop.

## Step 1: Inspect the Project
Before starting, familiarize yourself with the codebase layout:
- `configs/base/`: Contains the baseline configs for models, datasets, TPUs, etc.
- `schemas/`: Defines JSON schemas that serve as contracts between stages.
- `scripts/`: Python and Shell scripts to run various lifecycle stages.
- `.agents/skills/`: Custom agent instructions that AntiGravity loads to automate the steps.

## Step 2: Prepare the Dataset
The dataset agent curates and formats the training data.
Run:
```bash
python scripts/data/download_dataset.py --dataset dolly-sample
python scripts/data/prepare_dataset.py --input data/raw/dolly-sample.json --output-dir data/processed
python scripts/data/validate_dataset.py --data-dir data/processed
```
This produces `train.jsonl`, `valid.jsonl`, `test.jsonl` under `data/processed/`, as well as a `dataset_report.json` verifying the data is clean and split properly.

## Step 3: Generate Config
Configure the training hyperparameters and TPU options.
Run:
```bash
python scripts/training/build_training_config.py --experiment-id run_001
```
This merges default values with experiment-specific overrides, producing `configs/experiments/run_001.yaml` and validating it against `schemas/training_config.schema.json`.

## Step 4: Run Environment Checks
Ensure your machine and TPU instances are accessible and authenticated:
```bash
bash scripts/bootstrap/check_environment.sh
bash scripts/bootstrap/check_gcloud_tpu.sh
```

## Step 5: Run Smoke Training
Run a fast, single-step training smoke test to verify model loading, optimizer state initialization, and Tunix execution pipeline on JAX/TPU:
```bash
bash scripts/training/run_smoke_training.sh --config configs/experiments/run_001.yaml --run-id run_001
```
Verify that `runs/run_001/smoke_training.log` and `runs/run_001/training_run_manifest.json` are created.

## Step 6: Run Local Evaluation
Run a cheap local evaluation (e.g. 10-20 steps of Tunix diagnostic check) to inspect optimization trajectory:
```bash
python scripts/evaluation/run_local_eval.py --run-id run_001
```
This writes `runs/run_001/local_eval_report.json` showing training loss slope, validation accuracy estimate, and text sample outputs.

## Step 7: Local Hyperparameter Tuning Search
The system executes multiple 10-20 step diagnostic runs. To keep search tractable, you optimize **only one hyperparameter at a time** (sequential coordinate search).
- For the single target hyperparameter, you can launch up to **10 parallel trials** to compare different values.
- Up to **10 parameters** can be sequentially optimized during active research.
- The overall search loop budget is bounded by these parameters (up to 100 runs max). AntiGravity reviews metrics to select the best parameters.

## Step 8: Auto-Transition to Full Fine-Tuning
Once the best config is identified (or the local search budget is exhausted), the orchestrator **automatically proceeds** to full TPU fine-tuning by default. (If user confirmation gates are explicitly enabled in the configuration, the orchestrator will prompt you in chat first).

## Step 9: Run Full Evaluation
After full training completes, evaluate the final model against the baseline model:
```bash
python scripts/evaluation/run_full_eval.py --run-id run_001
python scripts/evaluation/compare_with_baseline.py --run-id run_001
```
This produces `runs/run_001/full_eval_report.json` and `reports/run_001_full_evaluation.md`.

## Step 10: Analyze Results
Summarize the findings and learn from any failure cases:
```bash
python scripts/analysis/summarize_run.py --run-id run_001
python scripts/analysis/propose_next_experiment.py --run-id run_001
```
This outputs `reports/run_001_analysis.md` and `runs/run_001/next_experiment.json`.

## Step 11: Propose Next Experiment
If the model failed on certain instructions (e.g. math or code formatting), the Orchestrator can trigger `data-synthesis` to generate targeting examples, beginning the next lifecycle iteration.
