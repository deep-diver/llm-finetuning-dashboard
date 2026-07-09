#!/usr/bin/env python3
"""
run_coordinate_ascent.py
Autonomous Coordinate Ascent Hyperparameter Search Orchestrator.
Sequentially tunes 10 hyperparameters with 10 values each (total 100 trials).
Then performs full training with the optimal hyperparameters.
"""
import os
import sys
import subprocess
import time
import json
import yaml

# Resource specs
ZONE = "us-west4-a"
PROJECT = "gcp-ml-172005"
ACCELERATOR = "v5litepod-8"
RUNTIME = "v2-alpha-tpuv5-lite"
TPU_NAME = "finetuning-tpu-vm"
QR_NAME = "finetuning-tpu-qr"

# Search state path
STATE_PATH = "runs/coordinate_search_state.json"

# Define the 10 parameters and their 10 values
PARAMETERS_SPACE = [
    {
        "name": "learning_rate",
        "arg": "--lr",
        "values": [1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4, 1e-3],
        "default": 5e-5
    },
    {
        "name": "warmup_steps",
        "arg": "--warmup-steps",
        "values": [0, 2, 5, 8, 10, 12, 15, 20, 25, 30],
        "default": 10
    },
    {
        "name": "weight_decay",
        "arg": "--weight-decay",
        "values": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
        "default": 0.01
    },
    {
        "name": "gradient_accumulation_steps",
        "arg": "--grad-accum",
        "values": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
        "default": 1
    },
    {
        "name": "lora_r",
        "arg": "--lora-r",
        "values": [2, 4, 6, 8, 12, 16, 24, 32, 48, 64],
        "default": 8
    },
    {
        "name": "lora_alpha",
        "arg": "--lora-alpha",
        "values": [4, 8, 16, 24, 32, 48, 64, 96, 128, 256],
        "default": 16
    },
    {
        "name": "lora_dropout",
        "arg": "--lora-dropout",
        "values": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15],
        "default": 0.05
    },
    {
        "name": "max_grad_norm",
        "arg": "--max-grad-norm",
        "values": [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0],
        "default": 1.0
    },
    {
        "name": "batch_size",
        "arg": "--batch-size",
        "values": [1, 1, 1, 1, 2, 2, 2, 2, 4, 4],  # Safe limits to avoid OOM
        "default": 1
    },
    {
        "name": "adam_beta1",
        "arg": "--adam-beta1",
        "values": [0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.98],
        "default": 0.9
    }
]


def run_cmd(args, shell=False, retries=2, stream=False, timeout=600):
    """Run a command with retries and optional streaming output."""
    for attempt in range(retries + 1):
        if attempt > 0:
            wait = 10 * attempt
            print(f"  ⟳ Retry {attempt}/{retries} (waiting {wait}s)...")
            time.sleep(wait)

        try:
            if stream:
                proc = subprocess.Popen(
                    args, shell=shell,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                output_lines = []
                for line in proc.stdout:
                    stripped = line.strip()
                    if stripped:
                        print(f"    [TPU] {stripped}")
                        output_lines.append(stripped)
                proc.wait(timeout=timeout)
                stdout = "\n".join(output_lines)
                if proc.returncode == 0:
                    return True, stdout, ""
                stderr = stdout  # stderr mixed into stdout when streaming
            else:
                res = subprocess.run(
                    args, shell=shell, capture_output=True, text=True, timeout=timeout
                )
                if res.returncode == 0:
                    return True, res.stdout, res.stderr
                stderr = res.stderr or res.stdout

            # Print error detail on failure
            print(f"  ❌ Command failed (attempt {attempt+1}): {stderr[-500:] if stderr else '(no output)'}")

        except subprocess.TimeoutExpired:
            print(f"  ⏱ Command timed out after {timeout}s")
            stderr = "timeout"

    return False, "", stderr


def ssh_cmd(remote_cmd, stream=False, retries=3, timeout=600):
    """Run a command on the remote TPU VM via gcloud SSH."""
    args = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", TPU_NAME,
        "--zone", ZONE, "--project", PROJECT,
        "--command", remote_cmd
    ]
    return run_cmd(args, stream=stream, retries=retries, timeout=timeout)


def kill_remote_jax():
    """Kill any lingering JAX/Python training processes and clean up temp files."""
    print("  🔪 Killing JAX processes and cleaning up disk...")
    # Kill process
    ssh_cmd("pkill -9 -f train_jax.py || true", retries=1, timeout=30)
    time.sleep(5)  # Give TPU time to fully release its lock
    # Clean up temp files that accumulate across trials
    ssh_cmd(
        "rm -rf /tmp/tpu_logs/* /tmp/jax_* /tmp/xla_* ~/.cache/pip/http* 2>/dev/null || true",
        retries=1, timeout=30
    )
    # Show remaining disk usage for visibility
    ok, out, _ = ssh_cmd("df -h / | tail -1", retries=1, timeout=15)
    if ok and out.strip():
        print(f"  💾 Remote disk: {out.strip()}")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)

    initial_best = {p["name"]: p["default"] for p in PARAMETERS_SPACE}
    return {
        "current_param_index": 0,
        "best_hyperparams": initial_best,
        "completed_trials": []
    }


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def cleanup_resources():
    print("🧹 Cleaning up remote GCP VM and queued resources...")
    subprocess.run(["gcloud", "compute", "tpus", "tpu-vm", "delete", TPU_NAME,
                    "--zone", ZONE, "--project", PROJECT, "--quiet"],
                   capture_output=True)
    subprocess.run(["gcloud", "compute", "tpus", "queued-resources", "delete", QR_NAME,
                    "--zone", ZONE, "--project", PROJECT, "--quiet"],
                   capture_output=True)


def sync_workspace():
    """Sync local code/configs to the remote TPU VM."""
    sync_cmd = (
        f"tar -cf - configs data schemas scripts src | "
        f"gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone {ZONE} --project {PROJECT} "
        f"--command \"mkdir -p ~/llm-finetuning && tar -C ~/llm-finetuning -xf -\""
    )
    ok, _, err = run_cmd(sync_cmd, shell=True, retries=2, timeout=120)
    if not ok:
        print(f"  ⚠ Workspace sync failed: {err}")
    return ok


def pull_trial_metrics(trial_id, retries=3):
    """Pull train_metrics.json from remote for a given trial. Returns final_loss or 999."""
    for attempt in range(retries):
        # Pull metrics only (exclude checkpoints to save bandwidth)
        pull_cmd = (
            f"gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone {ZONE} --project {PROJECT} "
            f"--command \"cd ~/llm-finetuning && "
            f"[ -f runs/{trial_id}/train_metrics.json ] && "
            f"tar --exclude='runs/{trial_id}/checkpoints' -cf - runs/{trial_id} || "
            f"echo 'MISSING'\" | tar -xf - 2>/dev/null || true"
        )
        run_cmd(pull_cmd, shell=True, retries=0, timeout=60)

        metrics_path = f"runs/{trial_id}/train_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            loss = m.get("final_loss", 999.0)
            print(f"  ✅ Metrics retrieved: final_loss={loss:.4f}, steps={m.get('total_steps', '?')}")

            # Delete remote run dir entirely to free disk space (metrics already pulled locally)
            print(f"  🗑 Freeing remote disk: removing runs/{trial_id} on TPU VM...")
            ssh_cmd(f"rm -rf ~/llm-finetuning/runs/{trial_id}", retries=1, timeout=30)

            return loss

        if attempt < retries - 1:
            print(f"  ⟳ train_metrics.json not found locally, retrying pull ({attempt+2}/{retries})...")
            time.sleep(8)

    print(f"  ⚠ Could not retrieve train_metrics.json for {trial_id} after {retries} attempts")
    # Still try to clean up even if pull failed
    ssh_cmd(f"rm -rf ~/llm-finetuning/runs/{trial_id}", retries=1, timeout=30)
    return 999.0


def run_trial(trial_id, param_name, val, overrides, hf_token):
    """Run a single training trial on the remote TPU. Returns final_loss."""
    print(f"\n  👉 Trial {trial_id} | {param_name}={val}")

    # Build config locally
    build_args = ["python3", "scripts/training/build_training_config.py", "--run-id", trial_id]
    for p in PARAMETERS_SPACE:
        build_args.extend([p["arg"], str(overrides[p["name"]])])
    ok, _, err = run_cmd(build_args, retries=1)
    if not ok:
        print(f"  ❌ Config build failed: {err}")
        return 999.0

    # Verify config was created and is valid
    config_path = f"configs/experiments/{trial_id}.yaml"
    if not os.path.exists(config_path):
        print(f"  ❌ Config file not found: {config_path}")
        return 999.0
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        print(f"  ❌ Config file parsed as None (empty YAML): {config_path}")
        return 999.0

    # Kill any leftover JAX process before syncing
    kill_remote_jax()

    # Sync workspace to remote
    sync_workspace()

    # Run training remotely with streaming output
    train_remote_cmd = (
        f"cd ~/llm-finetuning && "
        f"export XLA_PYTHON_CLIENT_PREALLOCATE=false && "
        f"export HF_TOKEN={hf_token} && "
        f"export HUGGING_FACE_HUB_TOKEN={hf_token} && "
        f"PYTHONPATH=src python3 scripts/training/train_jax.py "
        f"--config configs/experiments/{trial_id}.yaml"
    )
    ok, _, _ = ssh_cmd(train_remote_cmd, stream=True, retries=1, timeout=600)
    if not ok:
        print(f"  ⚠ Training command returned non-zero (may still have produced metrics)")

    # Pull and return metrics
    return pull_trial_metrics(trial_id)


def main():
    state = load_state()
    current_best = state["best_hyperparams"]
    hf_token = os.environ.get("HF_TOKEN", "")

    print("====================================================================")
    print("⚡ Starting Coordinate Ascent Hyperparameter Search Pipeline")
    print(f"  Resuming from param index: {state['current_param_index']}/10")
    print(f"  Current best: {json.dumps(current_best, indent=2)}")
    print("====================================================================")

    # --- STEP 1: Provision TPU VM ---
    print("\n--- STEP 1: Provisioning TPU VM ---")
    exists_res = subprocess.run(
        ["gcloud", "compute", "tpus", "queued-resources", "describe", QR_NAME,
         "--zone", ZONE, "--project", PROJECT],
        capture_output=True
    )
    if exists_res.returncode != 0:
        print(f"Creating queued resource {QR_NAME}...")
        create_cmd = [
            "gcloud", "compute", "tpus", "queued-resources", "create", QR_NAME,
            "--zone", ZONE, "--accelerator-type", ACCELERATOR,
            "--runtime-version", RUNTIME, "--node-id", TPU_NAME,
            "--project", PROJECT,
            "--quiet"
        ]
        ok, _, err = run_cmd(create_cmd)
        if not ok:
            print(f"❌ Failed to create TPU VM: {err}")
            sys.exit(1)

    print("Waiting for TPU VM to become ACTIVE...")
    tpu_state = "UNKNOWN"
    for i in range(40):
        desc_res = subprocess.run(
            ["gcloud", "compute", "tpus", "queued-resources", "describe", QR_NAME,
             "--zone", ZONE, "--project", PROJECT, "--format=value(state.state)"],
            capture_output=True, text=True
        )
        tpu_state = desc_res.stdout.strip() if desc_res.returncode == 0 else "UNKNOWN"
        print(f"  Status: {tpu_state} (attempt {i+1}/40)")
        if tpu_state == "ACTIVE":
            break
        time.sleep(15)

    if tpu_state != "ACTIVE":
        print("❌ TPU VM did not become ACTIVE. Cleaning up.")
        cleanup_resources()
        sys.exit(1)

    # --- STEP 2: Bootstrap dependencies ---
    print("\n--- STEP 2: Bootstrapping remote dependencies ---")
    bootstrap_cmd = (
        "pip3 install --quiet --upgrade pip && "
        "pip3 install --quiet torch --index-url https://download.pytorch.org/whl/cpu && "
        "pip3 install --quiet 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && "
        "pip3 install --quiet optax flax 'transformers==4.45.0' pyyaml jsonschema rouge-score"
    )
    ok, _, err = ssh_cmd(bootstrap_cmd, retries=2, timeout=300)
    if not ok:
        print(f"⚠ Bootstrap had issues: {err[-300:]}")

    # --- STEP 3: Coordinate Ascent Loop ---
    print("\n--- STEP 3: Coordinate Ascent Search (100 trials total) ---")

    while state["current_param_index"] < len(PARAMETERS_SPACE):
        param_idx = state["current_param_index"]
        param = PARAMETERS_SPACE[param_idx]
        param_name = param["name"]

        print(f"\n{'='*68}")
        print(f"🔍 Tuning Parameter {param_idx+1}/10: '{param_name}'")
        print(f"   Candidate values: {param['values']}")
        print(f"   Fixed context: { {k: v for k, v in current_best.items() if k != param_name} }")
        print(f"{'='*68}")

        best_val = current_best[param_name]
        best_loss = 999.0

        for val_idx, val in enumerate(param["values"]):
            trial_id = f"coord_p{param_idx+1}_t{val_idx+1}"

            # Skip already completed trials
            existing = next(
                (t for t in state["completed_trials"] if t["trial_id"] == trial_id), None
            )
            if existing:
                print(f"  ⏭ Skipping completed: {trial_id} (loss={existing['loss']:.4f})")
                if existing["loss"] < best_loss:
                    best_loss = existing["loss"]
                    best_val = val
                continue

            # Set up overrides: fix all other params at their current best
            overrides = current_best.copy()
            overrides[param_name] = val

            trial_loss = run_trial(trial_id, param_name, val, overrides, hf_token)

            print(f"  📊 Result: '{param_name}'={val} → Loss={trial_loss:.4f}"
                  + (" ✨ NEW BEST!" if trial_loss < best_loss else ""))

            if trial_loss < best_loss:
                best_loss = trial_loss
                best_val = val

            # Record and persist
            state["completed_trials"].append({
                "trial_id": trial_id,
                "param_name": param_name,
                "value": val,
                "loss": trial_loss
            })
            save_state(state)

        print(f"\n✅ Round {param_idx+1} done: best '{param_name}' = {best_val}  (loss={best_loss:.4f})")
        current_best[param_name] = best_val
        state["current_param_index"] += 1
        state["best_hyperparams"] = current_best
        save_state(state)

    # --- STEP 4: Final Full Training ---
    print("\n====================================================================")
    print("🎉 COORDINATE ASCENT COMPLETE!")
    print(f"Optimal Hyperparameters:\n{json.dumps(current_best, indent=2)}")
    print("====================================================================")
    print("\n--- STEP 4: Final Full Training Run ---")

    full_build_args = [
        "python3", "scripts/training/build_training_config.py",
        "--run-id", "run_full", "--epochs", "3"
    ]
    for p in PARAMETERS_SPACE:
        full_build_args.extend([p["arg"], str(current_best[p["name"]])])
    ok, _, err = run_cmd(full_build_args, retries=1)
    if not ok:
        print(f"❌ Full training config build failed: {err}")
        cleanup_resources()
        sys.exit(1)

    # Validate config before sending to TPU
    with open("configs/experiments/run_full.yaml") as f:
        full_cfg = yaml.safe_load(f)
    if full_cfg is None:
        print("❌ Full training config is empty/None!")
        cleanup_resources()
        sys.exit(1)
    print(f"✅ Full training config validated: run_id={full_cfg.get('run_id')}")

    kill_remote_jax()
    sync_workspace()

    full_remote_cmd = (
        f"cd ~/llm-finetuning && "
        f"export XLA_PYTHON_CLIENT_PREALLOCATE=false && "
        f"export HF_TOKEN={hf_token} && "
        f"export HUGGING_FACE_HUB_TOKEN={hf_token} && "
        f"PYTHONPATH=src python3 scripts/training/train_jax.py "
        f"--config configs/experiments/run_full.yaml"
    )
    print("Running full training (streaming output)...")
    ssh_cmd(full_remote_cmd, stream=True, retries=1, timeout=3600)

    # Pull checkpoints
    print("Pulling final checkpoints...")
    pull_full = (
        f"gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone {ZONE} --project {PROJECT} "
        f"--command \"cd ~/llm-finetuning && tar -cf - runs/run_full\" | tar -xf -"
    )
    run_cmd(pull_full, shell=True, retries=2, timeout=300)
    print("✅ Full training complete. Checkpoint in runs/run_full/")

    cleanup_resources()
    print("👍 All done!")


if __name__ == "__main__":
    main()
