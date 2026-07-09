#!/usr/bin/env python3
import os
import sys
import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Sub-sample dataset to speed up training smoke tests.")
    parser.add_argument("--input", type=str, required=True, help="Path to input jsonl file.")
    parser.add_argument("--output", type=str, required=True, help="Path to write sample jsonl.")
    parser.add_argument("--size", type=int, default=10, help="Number of records to sample.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r") as f:
        lines = f.readlines()

    sample_size = min(len(lines), args.size)
    sampled_lines = random.sample(lines, sample_size)

    with open(args.output, "w") as f:
        f.writelines(sampled_lines)

    print(f"✅ Sampled {sample_size} records from {args.input} to {args.output}")

if __name__ == "__main__":
    main()
