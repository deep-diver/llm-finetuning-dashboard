#!/usr/bin/env python3
"""
run_ratchet_experiment.py
Autonomous ML Research Agent ratchet loop.
Proposes code changes, runs experiments on TPU, evaluates, and decides whether to commit (ratchet) or revert.
"""
import os
import sys
import subprocess
import time
import json
import yaml

# Experiment configuration
RUN_ID = "run_006"
BASELINE_LOSS = 4.7500
ZONE = "us-west4-a"
PROJECT = "gcp-ml-172005"
ACCELERATOR = "v5litepod-8"
RUNTIME = "v2-alpha-tpuv5-lite"
TPU_NAME = "finetuning-tpu-vm"
QR_NAME = "finetuning-tpu-qr"

def run_cmd(args, shell=False):
    """Run system command and print output."""
    print(f"Executing: {' '.join(args) if not shell else args}")
    res = subprocess.run(args, shell=shell, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ Command failed: {res.stderr}")
        return False, res.stdout, res.stderr
    return True, res.stdout, res.stderr

def main():
    print("========================================================")
    # 1. Bootstrapping TPU VM
    print("--- STEP 1: Provisioning TPU VM ---")
    
    # Check if resource already exists
    exists_cmd = ["gcloud", "compute", "tpus", "queued-resources", "describe", QR_NAME, "--zone", ZONE, "--project", PROJECT]
    exists = subprocess.run(exists_cmd, capture_output=True).returncode == 0
    
    if exists:
        print(f"Queued resource {QR_NAME} already exists. Skipping creation.")
    else:
        print(f"Creating queued resource {QR_NAME}...")
        create_cmd = [
            "gcloud", "compute", "tpus", "queued-resources", "create", QR_NAME,
            "--zone", ZONE, "--accelerator-type", ACCELERATOR,
            "--runtime-version", RUNTIME, "--node-id", TPU_NAME,
            "--project", PROJECT, "--quiet"
        ]
        success, _, _ = run_cmd(create_cmd)
        if not success:
            print("❌ Failed to create TPU VM resource. Aborting.")
            sys.exit(1)
            
    # Wait for active status
    print("Waiting for TPU VM to become ACTIVE...")
    state = "UNKNOWN"
    for i in range(40):
        desc_cmd = ["gcloud", "compute", "tpus", "queued-resources", "describe", QR_NAME, "--zone", ZONE, "--project", PROJECT, "--format=value(state.state)"]
        res = subprocess.run(desc_cmd, capture_output=True, text=True)
        state = res.stdout.strip() if res.returncode == 0 else "UNKNOWN"
        print(f"  Status: {state} (attempt {i+1}/40)")
        if state == "ACTIVE":
            break
        time.sleep(15)
        
    if state != "ACTIVE":
        print("❌ TPU VM did not become ACTIVE. Deleting and aborting.")
        cleanup_resources()
        sys.exit(1)
        
    # Bootstrap dependencies
    print("Installing JAX/Flax dependencies on TPU VM...")
    bootstrap_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", TPU_NAME,
        "--zone", ZONE, "--project", PROJECT,
        "--command", "pip3 install --quiet --upgrade pip && pip3 install --quiet torch --index-url https://download.pytorch.org/whl/cpu && pip3 install --quiet jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip3 install --quiet optax flax 'transformers==4.45.0' pyyaml jsonschema rouge-score"
    ]
    run_cmd(bootstrap_cmd)
    
    # 2. Build configuration for experiment
    print("\n--- STEP 2: Creating run config ---")
    config_cmd = [
        "python3", "scripts/training/build_training_config.py",
        "--run-id", RUN_ID, "--lr", "5e-05", "--batch-size", "1", "--epochs", "1"
    ]
    success, _, _ = run_cmd(config_cmd)
    if not success:
        print("❌ Failed to build experiment config.")
        cleanup_resources()
        sys.exit(1)
        
    # 3. Synchronize modified workspace via tar stream
    print("\n--- STEP 3: Syncing workspace to TPU VM ---")
    sync_cmd = (
        f"tar -cf - configs data schemas scripts src | "
        f"gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone {ZONE} --project {PROJECT} "
        f"--command \"mkdir -p ~/llm-finetuning && tar -C ~/llm-finetuning -xf -\""
    )
    success, _, _ = run_cmd(sync_cmd, shell=True)
    if not success:
        print("❌ Failed to sync workspace.")
        cleanup_resources()
        sys.exit(1)
        
    # 4. Execute modified JAX training on TPU VM
    print("\n--- STEP 4: Executing modified training on TPU VM ---")
    hf_token = os.environ.get("HF_TOKEN", "")
    train_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", TPU_NAME,
        "--zone", ZONE, "--project", PROJECT,
        "--command", f"cd ~/llm-finetuning && export XLA_PYTHON_CLIENT_PREALLOCATE=false && export HF_TOKEN={hf_token} && export HUGGING_FACE_HUB_TOKEN={hf_token} && PYTHONPATH=src python3 scripts/training/train_jax.py --config configs/experiments/{RUN_ID}.yaml"
    ]
    # We run this command synchronously and pipe output to terminal
    print("Training is running...")
    proc = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(f"  [TPU] {line.strip()}")
    proc.wait()
    
    if proc.returncode != 0:
        print("❌ Training execution failed.")
        cleanup_resources()
        sys.exit(1)
        
    # 5. Execute remote validation evaluation on TPU VM
    print("\n--- STEP 5: Running remote evaluation ---")
    eval_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", TPU_NAME,
        "--zone", ZONE, "--project", PROJECT,
        "--command", f"cd ~/llm-finetuning && PYTHONPATH=src python3 scripts/evaluation/run_local_eval.py --run-id {RUN_ID}"
    ]
    success, _, _ = run_cmd(eval_cmd)
    
    # 6. Retrieve report
    print("\n--- STEP 6: Pulling evaluation report ---")
    pull_cmd = (
        f"gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone {ZONE} --project {PROJECT} "
        f"--command \"tar -C ~/llm-finetuning --exclude=\\\"runs/{RUN_ID}/checkpoints\\\" -cf - runs/{RUN_ID}\" | tar -xf -"
    )
    success, _, _ = run_cmd(pull_cmd, shell=True)
    
    # 7. Evaluate metric and decide (Ratchet Decision)
    print("\n--- STEP 7: Ratchet Decision (Hypothesis Validation) ---")
    report_path = f"runs/{RUN_ID}/local_eval_report.json"
    
    experiment_loss = 99.0
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
            experiment_loss = report.get("estimated_val_loss", 99.0)
            
    print(f"Baseline Loss   : {BASELINE_LOSS:.4f}")
    print(f"Experiment Loss : {experiment_loss:.4f}")
    
    if experiment_loss < BASELINE_LOSS:
        print(f"🎉 SUCCESS! Loss improved from {BASELINE_LOSS:.4f} to {experiment_loss:.4f}.")
        print("✅ RATCHET: Committing and keeping the optimizer modifications!")
    else:
        print(f"❌ FAILURE. Loss did not improve (Current: {experiment_loss:.4f} >= Baseline: {BASELINE_LOSS:.4f}).")
        print("🔄 REVERT: Restoring original train_jax.py file from git...")
        run_cmd(["git", "checkout", "scripts/training/train_jax.py"])
        
    # 8. Clean up GCP resources
    print("\n--- STEP 8: Cleaning up GCP resources ---")
    cleanup_resources()
    print("========================================================")

def cleanup_resources():
    """Delete VM and queued resources."""
    print("Deleting TPU VM...")
    subprocess.run(["gcloud", "compute", "tpus", "tpu-vm", "delete", TPU_NAME, "--zone", ZONE, "--project", PROJECT, "--quiet"])
    print("Deleting queued resource...")
    subprocess.run(["gcloud", "compute", "tpus", "queued-resources", "delete", QR_NAME, "--zone", ZONE, "--project", PROJECT, "--quiet"])

if __name__ == "__main__":
    main()
