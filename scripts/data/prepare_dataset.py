#!/usr/bin/env python3
import os
import sys
import json
import argparse
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from finetuning_lifecycle.reporting import save_manifest, generate_dataset_card
from finetuning_lifecycle.dataset import check_data_leakage

def main():
    parser = argparse.ArgumentParser(description="Split and format raw instruction dataset.")
    parser.add_argument("--input", type=str, default="data/raw/dolly-sample.json", help="Path to raw json dataset.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Target processed directory.")
    parser.add_argument("--run-id", type=str, default="run_001", help="Experiment ID identifier.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Error: Input file {args.input} not found. Run download_dataset.py first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.join("runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(args.input, "r") as f:
        raw_data = json.load(f)

    # Process format and split
    formatted_data = []
    duplicate_count = 0
    seen = set()

    for item in raw_data:
        inst = item.get("instruction", "")
        resp = item.get("response", "")
        context = item.get("context", "")
        
        # Check formatting validation
        if not inst or not resp:
            continue
            
        if inst in seen:
            duplicate_count += 1
            continue
        seen.add(inst)
        
        formatted_data.append({
            "instruction": inst,
            "context": context,
            "response": resp
        })

    # Basic random split
    random.seed(42)
    random.shuffle(formatted_data)
    
    total = len(formatted_data)
    train_end = int(total * 0.6)
    valid_end = train_end + int(total * 0.2)
    
    train_split = formatted_data[:max(1, train_end)]
    valid_split = formatted_data[max(1, train_end):max(2, valid_end)]
    test_split = formatted_data[max(2, valid_end):]
    
    # Defaults in case dataset is extremely small
    if not test_split:
        test_split = valid_split

    splits = {
        "train": (train_split, "train.jsonl"),
        "valid": (valid_split, "valid.jsonl"),
        "test": (test_split, "test.jsonl")
    }

    for name, (data_list, filename) in splits.items():
        out_path = os.path.join(args.output_dir, filename)
        with open(out_path, "w") as f:
            for item in data_list:
                f.write(json.dumps(item) + "\n")
        print(f"Written {len(data_list)} items to {out_path}")

    # Build report manifest
    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    leakage = check_data_leakage(train_path, test_path)

    report = {
        "dataset_name": "dolly-sample",
        "source_url": "local-file://" + args.input,
        "splits": {
            "train": {
                "num_samples": len(train_split),
                "path": train_path
            },
            "valid": {
                "num_samples": len(valid_split),
                "path": os.path.join(args.output_dir, "valid.jsonl")
            },
            "test": {
                "num_samples": len(test_split),
                "path": test_path
            }
        },
        "statistics": {
            "total_records": total,
            "duplicate_count": duplicate_count,
            "format_error_count": len(raw_data) - total - duplicate_count,
            "leakage_detected": leakage,
            "mean_instruction_tokens": sum(len(x["instruction"].split()) for x in formatted_data) / total if total else 0,
            "mean_response_tokens": sum(len(x["response"].split()) for x in formatted_data) / total if total else 0
        }
    }

    # Save and validate
    report_path = os.path.join(run_dir, "dataset_report.json")
    save_manifest(report, report_path, "dataset_report.schema.json")
    print(f"✅ Generated dataset report: {report_path}")

    card_path = os.path.join(run_dir, "dataset_card.md")
    generate_dataset_card(args.run_id, report, card_path)
    print(f"✅ Generated dataset card: {card_path}")

if __name__ == "__main__":
    main()
