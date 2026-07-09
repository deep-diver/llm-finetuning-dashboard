#!/usr/bin/env python3
"""
propose_next_experiment.py

Analyzes the loss trajectory and eval metrics of a completed run to
dynamically propose hyperparameter adjustments for the next iteration,
rather than returning hard-coded fixed values.
"""
import os
import sys
import json
import yaml
import argparse
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from finetuning_lifecycle.reporting import save_manifest


def load_search_space():
    """Load hyperparameter search space definition."""
    path = os.path.join("configs", "base", "search_space.yaml")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f).get("search", {})


def load_all_results(run_prefix="run"):
    """Load eval results from all completed runs to guide next proposal."""
    import glob
    results = []
    for run_dir in sorted(glob.glob(os.path.join("runs", f"{run_prefix}_*"))):
        run_id = os.path.basename(run_dir)
        # Try full eval report first, then local
        for report_name in ["full_eval_report.json", "local_eval_report.json"]:
            report_path = os.path.join(run_dir, report_name)
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report = json.load(f)
                config_path = os.path.join("configs", "experiments", f"{run_id}.yaml")
                config = {}
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                results.append({"run_id": run_id, "report": report, "config": config})
                break
    return results


def compute_rouge_score(report):
    """Extract average ROUGE-L score from eval report."""
    samples = report.get("samples", [])
    if samples:
        scores = [s.get("rouge_l", 0.0) for s in samples]
        return sum(scores) / len(scores) if scores else 0.0
    return report.get("avg_rouge_l", report.get("mean_rouge_l", 0.0))


def propose_next_params(space, all_results, current_run_report):
    """
    Propose next hyperparameter values using a simple adaptive strategy:
    - Among unexplored combinations, pick the one most different from worst runs.
    - Falls back to random sampling from the search space.
    """
    hp_space = space.get("hyperparameters", {})

    # Collect already-tried combinations
    tried = set()
    for r in all_results:
        hp = r["config"].get("hyperparameters", {})
        key = (
            hp.get("learning_rate"),
            hp.get("batch_size"),
            hp.get("epochs"),
            hp.get("warmup_steps"),
        )
        tried.add(key)

    # Build candidate grid
    candidates = []
    for lr in hp_space.get("learning_rate", {}).get("values", [2e-4]):
        for bs in hp_space.get("batch_size", {}).get("values", [1]):
            for ep in hp_space.get("epochs", {}).get("values", [1]):
                for ws in hp_space.get("warmup_steps", {}).get("values", [10]):
                    key = (lr, bs, ep, ws)
                    if key not in tried:
                        candidates.append(key)

    if candidates:
        # Prefer candidates with learning rates far from the worst-performing run
        if all_results:
            worst = min(all_results, key=lambda r: compute_rouge_score(r["report"]))
            worst_lr = worst["config"].get("hyperparameters", {}).get("learning_rate", 2e-4)
            candidates.sort(key=lambda c: abs(c[0] - worst_lr), reverse=True)
        chosen = candidates[0]
        rationale = "Adaptive selection: chose unexplored combination most distinct from worst-performing run."
    else:
        # All combinations tried – pick best-performing and perturb slightly
        if all_results:
            best = max(all_results, key=lambda r: compute_rouge_score(r["report"]))
            best_hp = best["config"].get("hyperparameters", {})
            lr_vals = hp_space.get("learning_rate", {}).get("values", [2e-4])
            bs_vals = hp_space.get("batch_size", {}).get("values", [1])
            ep_vals = hp_space.get("epochs", {}).get("values", [1])
            ws_vals = hp_space.get("warmup_steps", {}).get("values", [10])
            chosen = (
                random.choice(lr_vals),
                random.choice(bs_vals),
                random.choice(ep_vals),
                random.choice(ws_vals),
            )
            rationale = "All combinations exhausted. Re-sampling randomly to probe variance."
        else:
            chosen = (2e-4, 1, 1, 10)
            rationale = "No prior results available. Using default values."

    return {
        "learning_rate": chosen[0],
        "batch_size": chosen[1],
        "epochs": chosen[2],
        "warmup_steps": chosen[3],
    }, rationale


def main():
    parser = argparse.ArgumentParser(
        description="Dynamically propose hyperparameters for the next experiment iteration."
    )
    parser.add_argument("--run-id", type=str, default="run_001",
                        help="Current run ID to base the proposal on.")
    parser.add_argument("--run-prefix", type=str, default="run",
                        help="Prefix to use when scanning all previous runs.")
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.run_id)

    # Load current run eval report
    current_report = {}
    for report_name in ["full_eval_report.json", "local_eval_report.json"]:
        report_path = os.path.join(run_dir, report_name)
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                current_report = json.load(f)
            break

    # Load search space
    space = load_search_space() or {}
    suggest_synthesis = current_report.get("recommend_data_synthesis", False)

    # Load all historical results
    all_results = load_all_results(args.run_prefix)

    # Propose next hyperparameters
    next_hp, rationale = propose_next_params(space, all_results, current_report)

    # Increment run ID
    try:
        prefix, num = args.run_id.split("_")
        next_id = f"{prefix}_{int(num)+1:03d}"
    except ValueError:
        next_id = f"{args.run_id}_next"

    current_rouge = compute_rouge_score(current_report) if current_report else 0.0

    proposal = {
        "parent_run_id": args.run_id,
        "proposed_run_id": next_id,
        "rationale": rationale,
        "current_avg_rouge_l": current_rouge,
        "adjustments": {
            **next_hp,
            "data_mix": {
                "use_synthetic_data": suggest_synthesis,
                "synthetic_data_ratio": 0.2 if suggest_synthesis else 0.0,
            },
        },
        "focus_areas": ["loss_trajectory", "rouge_l_improvement"],
    }

    # Save proposal manifest
    out_path = os.path.join(run_dir, "next_experiment.json")
    save_manifest(proposal, out_path, "next_experiment.schema.json")
    print(f"✅ Generated next experiment proposal manifest: {out_path}")

    # Save research notes
    notes_path = os.path.join(run_dir, "research_notes.md")
    with open(notes_path, "w") as f:
        f.write(f"""# Research Notes: {args.run_id} → {next_id}

## Current Run Results
- **Average ROUGE-L**: {current_rouge:.4f}

## Proposed Next Hyperparameters
- **Learning Rate**: {next_hp['learning_rate']}
- **Batch Size**: {next_hp['batch_size']}
- **Epochs**: {next_hp['epochs']}
- **Warmup Steps**: {next_hp['warmup_steps']}

## Rationale
{rationale}

## Data Mix
- **Use Synthetic Data**: {suggest_synthesis}
- **Synthetic Ratio**: {0.2 if suggest_synthesis else 0.0}
""")
    print(f"✅ Generated research notes: {notes_path}")


if __name__ == "__main__":
    main()
