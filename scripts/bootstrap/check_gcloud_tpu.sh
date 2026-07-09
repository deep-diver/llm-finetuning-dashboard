#!/usr/bin/env bash
# check_gcloud_tpu.sh - Validates GCP credentials and TPU accessibility.

set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "========================================="
echo "Verifying gcloud and TPU cloud settings..."
echo "========================================="

python3 -c "
import sys
from finetuning_lifecycle.tpu import check_gcloud_auth
ok, msg = check_gcloud_auth()
print(f'Gcloud Auth Check: {\"✅\" if ok else \"❌\"} - {msg}')
if not ok:
    sys.exit(0) # Non-blocking for tutorial scaffold purposes
"

echo "Note: Running TPU describe commands requires base/tpu.yaml values."
echo "Ensure configs/base/tpu.yaml contains your target project and VM names."
echo "========================================="
