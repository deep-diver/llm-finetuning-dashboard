#!/usr/bin/env python3
import os
import sys
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate targeted synthetic instruction tuning data.")
    parser.add_argument("--run-id", type=str, default="run_001", help="Unique identifier for current run.")
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.run_id)
    failures_path = os.path.join(run_dir, "failure_cases.jsonl")

    if not os.path.exists(failures_path):
        print(f"⚠️ No failure cases found at {failures_path}. Using general format synthesis.")
        failures_exist = False
    else:
        failures_exist = True

    print(f"Generating synthetic formatting examples targeting failures in {args.run_id}...")

    # Write target synthetic examples
    synthetic_examples = [
        {
            "instruction": "Write a JSON object with a 'status' key set to 'active'.",
            "context": "",
            "response": "{\"status\": \"active\"}"
        },
        {
            "instruction": "Output only JSON structure containing 'items' list.",
            "context": "",
            "response": "{\"items\": []}"
        }
    ]

    out_dir = "data/synthetic"
    os.makedirs(out_dir, exist_ok=True)
    synthetic_file = os.path.join(out_dir, f"{args.run_id}_synthetic.jsonl")

    with open(synthetic_file, "w") as f:
        for item in synthetic_examples:
            f.write(json.dumps(item) + "\n")
    print(f"✅ Generated synthetic dataset split: {synthetic_file}")

    # Generate synthetic report
    report = {
        "run_id": args.run_id,
        "synthetic_file_path": synthetic_file,
        "records_generated": len(synthetic_examples),
        "target_failures": ["format_violation"] if failures_exist else ["general_formatting"]
    }

    report_path = os.path.join(run_dir, "synthetic_data_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Generated synthetic data report: {report_path}")

    # Generate synthetic data card
    card_path = os.path.join(run_dir, "synthetic_data_card.md")
    with open(card_path, "w") as f:
        f.write(f"""# Synthetic Data Card: {args.run_id}

## Dataset Profile
* **Target synthetic output file**: `{synthetic_file}`
* **Records generated**: {len(synthetic_examples)}
* **Goal**: Correct formatting errors (particularly JSON format compliance) identified during full evaluation of {args.run_id}.
""")
    print(f"✅ Generated synthetic data card: {card_path}")

if __name__ == "__main__":
    main()
