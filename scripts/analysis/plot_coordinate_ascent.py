#!/usr/bin/env python3
"""
plot_coordinate_ascent.py
Generates a Karpathy-style Autoresearch progress plot tracking coordinate ascent trials.
Plots kept improvements, discarded trials, running best validation loss, and annotations.
"""
import os
import sys
import json
import shutil

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib"])
    import matplotlib.pyplot as plt
    import numpy as np

STATE_PATH = "runs/coordinate_search_state.json"
OUTPUT_DIR = "reports"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "coordinate_ascent_progress.png")
ARTIFACT_DIR = "/Users/deep-diver/.gemini/antigravity/brain/9c0fc136-8be6-4471-bac4-5d024a1c789f"

PARAMS_SPACE = [
    "learning_rate", "warmup_steps", "weight_decay",
    "gradient_accumulation_steps", "lora_r", "lora_alpha",
    "lora_dropout", "max_grad_norm", "batch_size", "adam_beta1"
]

def main():
    import glob

    # Scan all local runs/coord_*/train_metrics.json for real final_loss values
    metrics_by_id = {}
    for metrics_path in sorted(glob.glob("runs/coord_*/train_metrics.json")):
        with open(metrics_path) as f:
            m = json.load(f)
        metrics_by_id[m["run_id"]] = m["final_loss"]

    # Load state file for param_name/value metadata
    state_trials = []
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
        state_trials = state.get("completed_trials", [])

    # Merge: real loss from train_metrics.json overrides state file
    trials_by_id = {t["trial_id"]: t.copy() for t in state_trials}
    for tid, real_loss in metrics_by_id.items():
        if tid in trials_by_id:
            trials_by_id[tid]["loss"] = real_loss
        else:
            parts = tid.split("_")
            if len(parts) == 3:
                p_idx = int(parts[1][1:]) - 1
                param_name = PARAMS_SPACE[p_idx] if p_idx < len(PARAMS_SPACE) else "unknown"
                trials_by_id[tid] = {
                    "trial_id": tid,
                    "param_name": param_name,
                    "value": "?",
                    "loss": real_loss
                }

    # Helper function to sort trials by parameter index and trial index numerically
    def get_sort_key(t):
        tid = t["trial_id"]
        parts = tid.split("_")
        try:
            p_idx = int(parts[1][1:]) # e.g. p1 -> 1
            t_idx = int(parts[2][1:]) # e.g. t10 -> 10
            return (p_idx, t_idx)
        except Exception:
            return (999, 999)

    trials = sorted(trials_by_id.values(), key=get_sort_key)


    if not trials:
        print("❌ No completed trials found yet.")
        sys.exit(0)

    # Extract data
    x = []
    y = []
    labels = []

    valid_trials = [t for t in trials if t["loss"] < 900.0]
    if not valid_trials:
        print("⚠ No valid trials yet.")
        sys.exit(0)

    for idx, t in enumerate(valid_trials):
        x.append(idx + 1)
        y.append(t["loss"])
        labels.append(f"{t['param_name']}={t['value']}")
        
    n_trials = len(x)
    
    # Calculate running best loss
    running_best = []
    kept_indices = []
    
    current_best = y[0]
    running_best.append(current_best)
    kept_indices.append(0) # First run (baseline) is kept
    
    for i in range(1, n_trials):
        if y[i] < current_best:
            current_best = y[i]
            kept_indices.append(i)
        running_best.append(current_best)
        
    # Split into kept vs discarded
    x_np = np.array(x)
    y_np = np.array(y)
    
    kept_x = x_np[kept_indices]
    kept_y = y_np[kept_indices]
    kept_labels = [labels[i] for i in kept_indices]
    
    discarded_indices = [i for i in range(n_trials) if i not in kept_indices]
    discarded_x = x_np[discarded_indices]
    discarded_y = y_np[discarded_indices]
    
    # Setup styling - Clean light/white theme similar to Karpathy's plot
    plt.figure(figsize=(15, 7.5))
    plt.grid(True, which='both', linestyle=':', color='lightgrey', alpha=0.7)
    
    # Plot Discarded runs (grey squares)
    if len(discarded_x) > 0:
        plt.scatter(discarded_x, discarded_y, color='#e5e7eb', edgecolors='#9ca3af', 
                    marker='s', s=35, alpha=0.5, label='Discarded', zorder=2)
                    
    # Plot Kept runs (green circles)
    plt.scatter(kept_x, kept_y, color='#10b981', edgecolors='#047857', 
                marker='o', s=70, label='Kept', zorder=4)
                
    # Plot Running Best line (stair step)
    # We pad the stair step line for visual continuous flow (where='post' matches step-down at improvement points)
    plt.step(x, running_best, where='post', color='#10b981', linewidth=2, label='Running best', zorder=3)

    
    # Annotations for kept runs (offset diagonally)
    for k_idx, k_x, k_y, k_lbl in zip(kept_indices, kept_x, kept_y, kept_labels):
        # Format label name to be shorter/prettier
        short_lbl = k_lbl.replace("learning_rate", "lr").replace("gradient_accumulation_steps", "grad_accum")
        
        # Decide label text: first is baseline, rest are modifications
        display_text = "baseline" if k_idx == 0 else f"{short_lbl}"
        
        plt.annotate(
            display_text, 
            xy=(k_x, k_y), 
            xytext=(3, 10), 
            textcoords='offset points', 
            rotation=25, 
            fontsize=8, 
            color='#047857',
            weight='semibold',
            ha='left',
            va='bottom'
        )
        
    # Title & Labels
    plt.title(f"Autoresearch Progress: {n_trials} Trials, {len(kept_indices)} Kept Improvements", 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Experiment #", fontsize=11, labelpad=8)
    plt.ylabel("Validation Loss (lower is better)", fontsize=11, labelpad=8)
    
    # Adjust axes limits
    margin_y = (max(y) - min(y)) * 0.1 if len(y) > 1 and max(y) != min(y) else 0.05
    plt.ylim(min(y) - margin_y, max(y) + margin_y)
    plt.xlim(0, n_trials + 2)
    
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='lightgrey')
    plt.tight_layout()
    
    # Save chart
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Matplotlib chart generated: {OUTPUT_PATH}")
    
    # Copy to artifacts directory
    if os.path.exists(ARTIFACT_DIR):
        dest_path = os.path.join(ARTIFACT_DIR, "coordinate_ascent_progress.png")
        shutil.copy(OUTPUT_PATH, dest_path)
        print(f"✅ Copied chart to sandbox artifact folder: {dest_path}")

if __name__ == "__main__":
    main()
