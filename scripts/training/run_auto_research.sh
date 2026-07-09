#!/usr/bin/env bash
# =============================================================================
# run_auto_research.sh
#
# Real automated hyperparameter search pipeline for JAX/Flax LLM fine-tuning.
#
# Flow:
#   1. Provision TPU VM (once)
#   2. Loop (up to max_tuning_runs):
#        a. Generate config with new hyperparameter combination
#        b. Run smoke training on TPU VM
#        c. Pull checkpoint + run local evaluation
#        d. Record results; check early stopping
#   3. Select best hyperparameters (select_best_hyperparams.py)
#   4. Run full training with best config
#   5. Final evaluation + reports
#   6. Delete TPU VM and queued resource
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults & Argument Parsing
# ---------------------------------------------------------------------------
RUN_PREFIX="run"
MAX_TUNING_RUNS=20
EARLY_STOPPING_PATIENCE=5

usage() {
  echo "Usage: $0 [--max-runs N] [--prefix PREFIX]"
  echo "  --max-runs N      Max hyperparameter search iterations (default: 20)"
  echo "  --prefix PREFIX   Run ID prefix (default: run)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-runs)   MAX_TUNING_RUNS="$2"; shift 2 ;;
    --prefix)     RUN_PREFIX="$2";      shift 2 ;;
    --help|-h)    usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

# ---------------------------------------------------------------------------
# Environment & Config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_TPU="configs/base/tpu.yaml"
LOG_DIR="runs/auto_research_logs"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/auto_research_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"; }

# Read TPU settings from config
PROJECT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG_TPU}')); print(c['tpu']['project_id'])")
ZONE=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG_TPU}')); print(c['tpu']['zone'])")
ACCELERATOR=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG_TPU}')); print(c['tpu']['accelerator_type'])")
RUNTIME=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG_TPU}')); print(c['tpu']['runtime_version'])")
TPU_NAME=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG_TPU}')); print(c['tpu']['tpu_vm_name'])")
QR_NAME="finetuning-tpu-qr"

log "========================================================"
log "  Auto-Research Pipeline Starting"
log "  Max runs      : ${MAX_TUNING_RUNS}"
log "  Run prefix    : ${RUN_PREFIX}"
log "  TPU           : ${TPU_NAME} (${ACCELERATOR}) in ${ZONE}"
log "  Project       : ${PROJECT}"
log "========================================================"

# ---------------------------------------------------------------------------
# Helper: Generate next hyperparameter combination
# ---------------------------------------------------------------------------
# Reads search_space.yaml and returns values for run N (0-indexed)
# Strategy: iterate through grid in order, wrapping if exhausted
generate_hp_combo() {
  local run_index="$1"
  python3 - <<PYEOF
import yaml, itertools, sys

with open("configs/base/search_space.yaml") as f:
    space = yaml.safe_load(f)["search"]["hyperparameters"]

lr_vals  = space["learning_rate"]["values"]
bs_vals  = space["batch_size"]["values"]
ep_vals  = space["epochs"]["values"]
ws_vals  = space["warmup_steps"]["values"]

grid = list(itertools.product(lr_vals, bs_vals, ep_vals, ws_vals))
idx  = int(${run_index}) % len(grid)
lr, bs, ep, ws = grid[idx]
print(f"{lr} {bs} {ep} {ws}")
PYEOF
}

# ---------------------------------------------------------------------------
# Helper: Extract final loss from a run's local eval report
# ---------------------------------------------------------------------------
get_final_loss() {
  local run_id="$1"
  python3 - <<PYEOF
import json, os
for fname in ["full_eval_report.json", "local_eval_report.json"]:
    p = os.path.join("runs", "${run_id}", fname)
    if os.path.exists(p):
        d = json.load(open(p))
        print(d.get("final_loss", d.get("estimated_val_loss", d.get("loss", 99.0))))
        exit()
print(99.0)
PYEOF
}

# ---------------------------------------------------------------------------
# PHASE 1: Provision TPU VM
# ---------------------------------------------------------------------------
log "--- PHASE 1: Provisioning TPU VM ---"

log "Checking if queued resource ${QR_NAME} already exists..."
if gcloud compute tpus queued-resources describe "${QR_NAME}" --zone "${ZONE}" --project "${PROJECT}" >/dev/null 2>&1; then
  log "Queued resource ${QR_NAME} already exists. Skipping creation."
else
  log "Creating queued resource ${QR_NAME} (${ACCELERATOR}) in ${ZONE}..."
  gcloud compute tpus queued-resources create "${QR_NAME}" \
      --zone "${ZONE}" \
      --accelerator-type "${ACCELERATOR}" \
      --runtime-version "${RUNTIME}" \
      --node-id "${TPU_NAME}" \
      --project "${PROJECT}" 2>&1 | tee -a "${MAIN_LOG}"
fi

log "Waiting for TPU VM to become ACTIVE..."
for i in $(seq 1 40); do
  STATE=$(gcloud compute tpus queued-resources describe "${QR_NAME}" \
      --zone "${ZONE}" --project "${PROJECT}" \
      --format="value(state.state)" 2>/dev/null || echo "UNKNOWN")
  log "  Status: ${STATE} (attempt ${i}/40)"
  if [[ "${STATE}" == "ACTIVE" ]]; then break; fi
  sleep 15
done

if [[ "${STATE}" != "ACTIVE" ]]; then
  log "❌ TPU VM did not become ACTIVE after waiting. Aborting."
  exit 1
fi
log "✅ TPU VM is ACTIVE."

# Install dependencies on TPU VM (once)
log "Bootstrapping JAX dependencies on TPU VM..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" \
    --command "pip3 install --quiet --upgrade pip && \
               pip3 install --quiet torch --index-url https://download.pytorch.org/whl/cpu && \
               pip3 install --quiet jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
               pip3 install --quiet optax flax 'transformers==4.45.0' pyyaml jsonschema rouge-score" \
    2>&1 | tee -a "${MAIN_LOG}"
log "✅ Bootstrap complete."

# ---------------------------------------------------------------------------
# PHASE 2: Hyperparameter Search Loop
# ---------------------------------------------------------------------------
log "--- PHASE 2: Hyperparameter Search Loop (max ${MAX_TUNING_RUNS} runs) ---"

COMPLETED_RUNS=0
NO_IMPROVE_COUNT=0
BEST_LOSS=99.0
LAST_RUN_ID=""

for run_index in $(seq 0 $((MAX_TUNING_RUNS - 1))); do

  # Format run ID: run_001, run_002, ...
  RUN_ID=$(printf "%s_%03d" "${RUN_PREFIX}" $((run_index + 1)))
  LAST_RUN_ID="${RUN_ID}"
  log ""
  log "=== Tuning Run ${run_index+1}/${MAX_TUNING_RUNS}: ${RUN_ID} ==="

  # Generate hyperparameter combo for this iteration
  read -r LR BS EP WS <<< "$(generate_hp_combo ${run_index})"
  log "  lr=${LR}  batch_size=${BS}  epochs=${EP}  warmup_steps=${WS}"

  # Build experiment config
  python3 scripts/training/build_training_config.py \
      --run-id "${RUN_ID}" \
      --lr "${LR}" \
      --batch-size "${BS}" \
      --epochs "${EP}" \
      2>&1 | tee -a "${MAIN_LOG}"

  # Sync workspace to TPU VM
  log "  Syncing workspace to TPU VM..."
  tar -cf - configs data schemas scripts src | \
      gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
      --zone "${ZONE}" --project "${PROJECT}" \
      --command "mkdir -p ~/llm-finetuning && tar -C ~/llm-finetuning -xf -" 2>&1 | tee -a "${MAIN_LOG}"

  # Prepare smoke config (1 epoch, batch_size=1 for speed)
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
      --zone "${ZONE}" --project "${PROJECT}" \
      --command "python3 -c \"
import yaml
cfg = yaml.safe_load(open('llm-finetuning/configs/experiments/${RUN_ID}.yaml'))
cfg['hyperparameters']['epochs'] = 1
cfg['hyperparameters']['batch_size'] = 1
yaml.dump(cfg, open('llm-finetuning/configs/experiments/${RUN_ID}_smoke.yaml', 'w'))
\"" 2>&1 | tee -a "${MAIN_LOG}"

  # Run smoke training
  log "  Running smoke training..."
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
      --zone "${ZONE}" --project "${PROJECT}" \
      --command "cd ~/llm-finetuning && \
                 export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
                 export HF_TOKEN=${HF_TOKEN:-} && \
                 export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-} && \
                 PYTHONPATH=src python3 scripts/training/train_jax.py \
                     --config configs/experiments/${RUN_ID}_smoke.yaml" \
      2>&1 | tee -a "${MAIN_LOG}" || {
        log "  ⚠️  Smoke training failed for ${RUN_ID}, skipping."
        continue
      }

  # Run local evaluation directly on the TPU VM (saves 3.5GB bandwidth!)
  log "  Running local evaluation on TPU VM..."
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
      --zone "${ZONE}" --project "${PROJECT}" \
      --command "cd ~/llm-finetuning && PYTHONPATH=src python3 scripts/evaluation/run_local_eval.py --run-id ${RUN_ID}" 2>&1 | tee -a "${MAIN_LOG}"

  # Pull only the tiny evaluation reports back (excluding the 3.5GB checkpoints directory!)
  log "  Retrieving evaluation report from TPU VM..."
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
      --zone "${ZONE}" --project "${PROJECT}" \
      --command "tar -C ~/llm-finetuning --exclude=\"runs/${RUN_ID}/checkpoints\" -cf - runs/${RUN_ID}" | tar -xf - 2>&1 | tee -a "${MAIN_LOG}"

  # Extract loss and check early stopping
  CURRENT_LOSS=$(get_final_loss "${RUN_ID}")
  log "  Final loss: ${CURRENT_LOSS} (best so far: ${BEST_LOSS})"

  IMPROVED=$(python3 -c "print('yes' if float('${CURRENT_LOSS}') < float('${BEST_LOSS}') - 0.01 else 'no')")
  if [[ "${IMPROVED}" == "yes" ]]; then
    BEST_LOSS="${CURRENT_LOSS}"
    NO_IMPROVE_COUNT=0
    log "  ✅ Improvement detected! New best loss: ${BEST_LOSS}"
  else
    NO_IMPROVE_COUNT=$((NO_IMPROVE_COUNT + 1))
    log "  No improvement (${NO_IMPROVE_COUNT}/${EARLY_STOPPING_PATIENCE})"
  fi

  COMPLETED_RUNS=$((COMPLETED_RUNS + 1))

  # Early stopping check
  if [[ ${NO_IMPROVE_COUNT} -ge ${EARLY_STOPPING_PATIENCE} ]]; then
    log "Early stopping triggered after ${COMPLETED_RUNS} runs (no improvement for ${EARLY_STOPPING_PATIENCE} consecutive runs)."
    break
  fi

done

log ""
log "=== Hyperparameter search complete. Completed ${COMPLETED_RUNS} runs. ==="

# ---------------------------------------------------------------------------
# PHASE 3: Select Best Hyperparameters
# ---------------------------------------------------------------------------
log "--- PHASE 3: Selecting Best Hyperparameters ---"
python3 scripts/analysis/select_best_hyperparams.py \
    --run-prefix "${RUN_PREFIX}" \
    --output "runs/best_hyperparams.json" \
    2>&1 | tee -a "${MAIN_LOG}"

BEST_RUN_ID=$(python3 -c "import json; d=json.load(open('runs/best_hyperparams.json')); print(d['best_run_id'])")
BEST_LR=$(python3 -c "import json; d=json.load(open('runs/best_hyperparams.json')); print(d['best_hyperparameters']['learning_rate'])")
BEST_BS=$(python3 -c "import json; d=json.load(open('runs/best_hyperparams.json')); print(d['best_hyperparameters']['batch_size'])")
BEST_EP=$(python3 -c "import json; d=json.load(open('runs/best_hyperparams.json')); print(d['best_hyperparameters']['epochs'])")

log "🏆 Best run: ${BEST_RUN_ID}  (lr=${BEST_LR}, batch_size=${BEST_BS}, epochs=${BEST_EP})"

# ---------------------------------------------------------------------------
# PHASE 4: Full Training with Best Config
# ---------------------------------------------------------------------------
log "--- PHASE 4: Full Training with Best Hyperparameters ---"
FULL_RUN_ID="${RUN_PREFIX}_full"

python3 scripts/training/build_training_config.py \
    --run-id "${FULL_RUN_ID}" \
    --lr "${BEST_LR}" \
    --batch-size "${BEST_BS}" \
    --epochs "${BEST_EP}" \
    2>&1 | tee -a "${MAIN_LOG}"

# Sync updated workspace
tar -cf - configs data schemas scripts src | \
    gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" \
    --command "mkdir -p ~/llm-finetuning && tar -C ~/llm-finetuning -xf -" 2>&1 | tee -a "${MAIN_LOG}"

# Run full training
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" \
    --command "cd ~/llm-finetuning && \
               export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
               export HF_TOKEN=${HF_TOKEN:-} && \
               export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-} && \
               PYTHONPATH=src python3 scripts/training/train_jax.py \
                   --config configs/experiments/${FULL_RUN_ID}.yaml" \
    2>&1 | tee -a "${MAIN_LOG}"

# Pull full training artifacts
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" \
    --command "tar -C ~/llm-finetuning -cf - runs/${FULL_RUN_ID}" | tar -xf - 2>&1 | tee -a "${MAIN_LOG}"

log "✅ Full training complete."

# ---------------------------------------------------------------------------
# PHASE 5: Final Evaluation & Reports
# ---------------------------------------------------------------------------
log "--- PHASE 5: Final Evaluation & Reports ---"

python3 scripts/evaluation/run_full_eval.py --run-id "${FULL_RUN_ID}" \
    2>&1 | tee -a "${MAIN_LOG}"

python3 scripts/evaluation/compare_with_baseline.py --run-id "${FULL_RUN_ID}" \
    2>&1 | tee -a "${MAIN_LOG}"

python3 scripts/analysis/summarize_run.py --run-id "${FULL_RUN_ID}" \
    2>&1 | tee -a "${MAIN_LOG}"

python3 scripts/analysis/propose_next_experiment.py \
    --run-id "${FULL_RUN_ID}" \
    --run-prefix "${RUN_PREFIX}" \
    2>&1 | tee -a "${MAIN_LOG}"

python3 scripts/synthesis/synthesize_additional_data.py --run-id "${FULL_RUN_ID}" \
    2>&1 | tee -a "${MAIN_LOG}"

log "✅ All reports generated."

# ---------------------------------------------------------------------------
# PHASE 6: Cleanup TPU Resources
# ---------------------------------------------------------------------------
log "--- PHASE 6: Cleaning Up TPU Resources ---"

log "Deleting TPU VM: ${TPU_NAME}..."
gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" --quiet \
    2>&1 | tee -a "${MAIN_LOG}"

log "Deleting queued resource: ${QR_NAME}..."
for i in $(seq 1 20); do
  STATE=$(gcloud compute tpus queued-resources describe "${QR_NAME}" \
      --zone "${ZONE}" --project "${PROJECT}" \
      --format="value(state.state)" 2>/dev/null || echo "DELETED")
  if [[ "${STATE}" == "DELETED" || "${STATE}" == "SUSPENDED" || "${STATE}" == "SUSPENDING" ]]; then break; fi
  sleep 10
done
gcloud compute tpus queued-resources delete "${QR_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" --quiet \
    2>&1 | tee -a "${MAIN_LOG}" || true

log "✅ All TPU resources cleaned up."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "========================================================"
log "  Auto-Research Pipeline Completed!"
log "  Tuning runs completed : ${COMPLETED_RUNS}"
log "  Best run              : ${BEST_RUN_ID}"
log "  Best lr               : ${BEST_LR}"
log "  Best batch_size       : ${BEST_BS}"
log "  Best epochs           : ${BEST_EP}"
log "  Full training run     : ${FULL_RUN_ID}"
log "  Log file              : ${MAIN_LOG}"
log "========================================================"
