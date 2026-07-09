#!/usr/bin/env python3
"""
select_best_hyperparams.py

Scans all completed auto-research runs and selects the best experiment
based on the configured selection metric (avg_rouge_l, final_loss, etc.).
Outputs the best run ID and its hyperparameters to a JSON file.
"""
import os
import sys
import json
import yaml
import argparse
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))


def load_eval_result(run_id):
    """Load full_eval_report.json for a given run, return None if missing."""
    report_path = os.path.join("runs", run_id, "full_eval_report.json")
    if not os.path.exists(report_path):
        # Fallback: try local_eval_report.json (smoke-only runs)
        report_path = os.path.join("runs", run_id, "local_eval_report.json")
        if not os.path.exists(report_path):
            return None
    with open(report_path, "r") as f:
        return json.load(f)


def load_run_config(run_id):
    """Load the experiment YAML config for a given run."""
    config_path = os.path.join("configs", "experiments", f"{run_id}.yaml")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_metric(report, metric_name):
    """Extract the selection metric value from an evaluation report."""
    if metric_name == "avg_rouge_l":
        # Try full eval report format first
        samples = report.get("samples", [])
        if samples:
            scores = [s.get("rouge_l", 0.0) for s in samples]
            return sum(scores) / len(scores) if scores else 0.0
        # Fallback: top-level avg_rouge_l
        return report.get("avg_rouge_l", report.get("mean_rouge_l", 0.0))
    elif metric_name == "final_loss":
        # Lower is better - negate so we can use max()
        loss = report.get("final_loss", report.get("loss", float("inf")))
        return -loss
    elif metric_name == "compliance_rate":
        return report.get("compliance_rate", 0.0)
    else:
        return report.get(metric_name, 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="Select best hyperparameters from completed auto-research runs."
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="run",
        help="Prefix used for run IDs (e.g. 'run' matches run_001, run_002 ...)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/best_hyperparams.json",
        help="Output path for the best hyperparameter JSON file.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Override selection metric (avg_rouge_l / final_loss / compliance_rate).",
    )
    args = parser.parse_args()

    # Load search space config to get selection metric
    search_space_path = os.path.join("configs", "base", "search_space.yaml")
    selection_metric = args.metric or "avg_rouge_l"
    if os.path.exists(search_space_path):
        with open(search_space_path, "r") as f:
            space = yaml.safe_load(f)
        selection_metric = args.metric or space.get("search", {}).get(
            "selection_metric", "avg_rouge_l"
        )

    print(f"Selection metric: {selection_metric}")

    # Discover all completed run directories
    run_dirs = sorted(glob.glob(os.path.join("runs", f"{args.run_prefix}_*")))
    if not run_dirs:
        print("❌ No completed runs found under runs/")
        sys.exit(1)

    results = []
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        report = load_eval_result(run_id)
        config = load_run_config(run_id)
        if report is None or config is None:
            print(f"  ⚠️  Skipping {run_id} (missing eval report or config)")
            continue
        score = extract_metric(report, selection_metric)
        hp = config.get("hyperparameters", {})
        results.append(
            {
                "run_id": run_id,
                "score": score,
                "metric": selection_metric,
                "hyperparameters": {
                    "learning_rate": hp.get("learning_rate"),
                    "batch_size": hp.get("batch_size"),
                    "epochs": hp.get("epochs"),
                    "warmup_steps": hp.get("warmup_steps"),
                },
            }
        )
        print(f"  {run_id}: {selection_metric}={score:.4f}  lr={hp.get('learning_rate')}  bs={hp.get('batch_size')}")

    if not results:
        print("❌ No valid results found to compare.")
        sys.exit(1)

    # Select best (higher score = better for all metrics; final_loss is negated above)
    best = max(results, key=lambda r: r["score"])
    print(f"\n🏆 Best run: {best['run_id']}  ({selection_metric}={best['score']:.4f})")
    print(f"   Hyperparameters: {best['hyperparameters']}")

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "best_run_id": best["run_id"],
                "selection_metric": selection_metric,
                "best_score": best["score"],
                "best_hyperparameters": best["hyperparameters"],
                "all_results": results,
            },
            f,
            indent=2,
        )
    print(f"✅ Saved best hyperparameter selection to: {args.output}")


if __name__ == "__main__":
    main()
