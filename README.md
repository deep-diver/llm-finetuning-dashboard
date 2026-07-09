# Multi-Agent LLM Fine-Tuning Lifecycle Tutorial

Welcome to the **Multi-Agent LLM Fine-Tuning Lifecycle** tutorial. This project demonstrates how AntiGravity IDE can coordinate an autonomous, multi-agent workflow to handle the complete fine-tuning research loop of Large Language Models (LLMs).

## Project Rationale

Fine-tuning LLMs is a multi-disciplinary effort that involves data curation, hardware provisioning, config generation, training execution, evaluation, error analysis, and hyperparameter tuning. In production, this process is slow, costly, and error-prone. 

This tutorial project demonstrates a **minimal, complete, and safe** fine-tuning lifecycle. We use:
- A **small model** (e.g., 0.5B parameters) to enable fast fine-tuning on affordable hardware.
- A **small instruction-tuning dataset subset** (e.g., a sample from Dolly-15k) to keep data processing fast and local.
- **Short runs** (smoke tests and few-step diagnostic training) to perform local search and parameter validation.
- **Explicit artifact contracts** (JSON schemas) between agents to coordinate state transitions.

The core logic of optimization is driven by the AI agent (AntiGravity) analyzing output metrics and diagnosing failures, while the code itself remains deterministic, providing reports, errors, and validation results.

## Agent System Overview

The system is designed around an **Orchestrator** and specialized **Sub-Agents**, each defined by a Skill under `.agents/skills/`:

1. **`finetuning-orchestrator`**: Inspects run states, determines the next agent to run, enforces safety limits, and manages user approvals.
2. **`dataset-preparation`**: Curates the raw instruction-tuning data, splits it (train/validation/test), and checks formatting, leakage, and duplicates.
3. **`training-configuration`**: Selects hyperparameters, configures TPU v5e settings, validates paths, and estimates training costs.
4. **`tpu-training-execution`**: Validates the `gcloud` environment, executes smoke training, and runs full fine-tuning.
5. **`local-evaluation`**: Runs short-run diagnostic evaluations to check if training is moving in the right direction.
6. **`full-evaluation`**: Performs comprehensive evaluation on held-out datasets after training.
7. **`result-analysis`**: Compares baseline vs. fine-tuned performance and recommends the next iteration.
8. **`data-synthesis`**: Synthesizes custom training data to target specific failure cases identified during evaluation.

## Workflow at a Glance

```
[dataset-preparation]
        │
        ▼
[training-configuration]
        │
        ▼
[tpu-training-execution] ──(Smoke test training)
        │
        ▼
[local-evaluation] ────(Local Auto-Research Loop - hyperparameter tuning within budget)
        │
        ▼ (Auto-Transition / Best Config Selected)
[tpu-training-execution] ──(Full fine-tuning)
        │
        ▼
[full-evaluation]
        │
        ▼
[result-analysis]
        │
        ▼ (Optional)
[data-synthesis] ───────(Next experiment iteration)
```

## Quick Start

1. **Bootstrap the environment**:
   Run the environment validation scripts to check Python dependencies and gcloud/TPU credentials:
   ```bash
   bash scripts/bootstrap/check_environment.sh
   bash scripts/bootstrap/check_gcloud_tpu.sh
   ```

2. **See the Tutorial**:
   Open [TUTORIAL.md](file:///Users/deep-diver/Developers/llm-finetuning/TUTORIAL.md) for a step-by-step walkthrough of the lifecycle.

3. **Explore Agent Details**:
   Read [AGENT_WORKFLOW.md](file:///Users/deep-diver/Developers/llm-finetuning/AGENT_WORKFLOW.md) for architectural details on how the agents collaborate.
