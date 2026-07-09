#!/usr/bin/env python3
import os
import sys
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="Estimate training duration and compute cost.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Cost Rate Cards (Mock prices for TPU v5e)
    # v5e-4 pricing is approx $4.80/hour ($1.20 per chip-hour for spot/on-demand v5e)
    tpu_type = config.get("tpu", {}).get("accelerator_type", "v5e-4")
    hourly_rate = 4.80 if "v5e-4" in tpu_type else 1.20
    
    # Model & Data params
    batch_size = config.get("hyperparameters", {}).get("batch_size", 4)
    epochs = config.get("hyperparameters", {}).get("epochs", 1)
    
    # Check dataset report for size if available, otherwise fallback
    run_id = config.get("run_id", "run_001")
    report_path = f"runs/{run_id}/dataset_report.json"
    num_samples = 500
    if os.path.exists(report_path):
        import json
        with open(report_path, "r") as f:
            rep = json.load(f)
            num_samples = rep.get("splits", {}).get("train", {}).get("num_samples", 500)

    # Estimate training steps
    steps = (num_samples * epochs) // batch_size
    
    # Let's assume a step rate of 100ms per step on TPU v5e for a 0.5B model
    step_duration_sec = 0.1
    total_seconds = steps * step_duration_sec
    total_hours = total_seconds / 3600.0
    estimated_cost = total_hours * hourly_rate

    print(f"=== Compute Budget Estimation for {run_id} ===")
    print(f"TPU Type: {tpu_type} (Est. Hourly Rate: ${hourly_rate:.2f})")
    print(f"Dataset Size: {num_samples} samples")
    print(f"Total Steps: {steps}")
    print(f"Estimated Training Time: {total_seconds:.2f} seconds ({total_hours:.4f} hours)")
    print(f"Estimated Cost: ${estimated_cost:.4f}")
    print("==============================================")

if __name__ == "__main__":
    main()
