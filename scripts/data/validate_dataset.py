#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from finetuning_lifecycle.dataset import analyze_dataset_file, check_data_leakage

def main():
    parser = argparse.ArgumentParser(description="Audit and validate split datasets.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed splits directory.")
    args = parser.parse_args()

    splits = ["train.jsonl", "valid.jsonl", "test.jsonl"]
    all_ok = True

    print("Checking dataset splits...")
    for filename in splits:
        file_path = os.path.join(args.data_dir, filename)
        if not os.path.exists(file_path):
            print(f"❌ Missing split: {file_path}")
            all_ok = False
            continue
            
        stats = analyze_dataset_file(file_path)
        print(f"\nAudit for {filename}:")
        print(f"  - Count: {stats['record_count']}")
        print(f"  - Formatting Errors: {stats['format_error_count']}")
        print(f"  - Duplicate entries: {stats['duplicate_count']}")
        print(f"  - Avg Instruction Length (words): {stats['mean_instruction_tokens']:.2f}")
        print(f"  - Avg Response Length (words): {stats['mean_response_tokens']:.2f}")
        
        if stats["format_error_count"] > 0:
            print(f"⚠️ Warning: format errors detected in {filename}")
            
    train_path = os.path.join(args.data_dir, "train.jsonl")
    test_path = os.path.join(args.data_dir, "test.jsonl")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        leakage = check_data_leakage(train_path, test_path)
        if leakage:
            print("\n❌ Data leakage detected! Overlap found between train and test datasets.")
            all_ok = False
        else:
            print("\n✅ Leakage audit: No instruction overlap detected between train and test splits.")

    if all_ok:
        print("\n🎉 Dataset validation passed successfully!")
    else:
        print("\n❌ Dataset validation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
