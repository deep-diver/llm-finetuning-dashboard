#!/usr/bin/env python3
"""
plot_tuning.py
Generates a static matplotlib visualization of the auto-research results
and saves it as an image for direct embedding.
"""
import os
import sys
import json
import yaml

# Check and install matplotlib if missing
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Installing matplotlib...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib"])
    import matplotlib.pyplot as plt
    import numpy as np

def load_run_data():
    runs = []
    losses = []
    compliances = []
    
    # Scan run_001 to run_005
    for i in range(1, 6):
        run_id = f"run_00{i}"
        report_path = os.path.join("runs", run_id, "local_eval_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                data = json.load(f)
                runs.append(run_id)
                losses.append(data.get("estimated_val_loss", 4.75))
                compliances.append(data.get("format_compliance_rate", 1.0) * 100)
                
    return runs, losses, compliances

def main():
    runs, losses, compliances = load_run_data()
    if not runs:
        print("❌ No run reports found to plot.")
        sys.exit(1)
        
    # Styling configuration for modern dark theme
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Estimated Validation Loss
    ax1.plot(runs, losses, marker='o', linewidth=2.5, color='#8b5cf6', markersize=8, markerfacecolor='#a78bfa')
    ax1.set_title("Tuning Runs: Estimated Validation Loss", fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel("Tuning Iterations", fontsize=10, labelpad=10)
    ax1.set_ylabel("Loss Value", fontsize=10, labelpad=10)
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.set_ylim(4.5, 5.0)
    
    # Plot 2: Format Compliance Rate
    bars = ax2.bar(runs, compliances, color='#10b981', alpha=0.8, width=0.4)
    ax2.set_title("Tuning Runs: Format Compliance Rate (%)", fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel("Tuning Iterations", fontsize=10, labelpad=10)
    ax2.set_ylabel("Compliance Rate (%)", fontsize=10, labelpad=10)
    ax2.grid(True, linestyle='--', alpha=0.2, axis='y')
    ax2.set_ylim(0, 110)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, height + 2, f'{height:.1f}%', ha='center', va='bottom', fontsize=9, color='#e5e7eb')
        
    plt.tight_layout()
    
    # Save image
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "auto_research_chart.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved matplotlib chart to: {output_path}")

    # Copy to artifacts directory if in antigravity sandbox env
    artifact_dir = "/Users/deep-diver/.gemini/antigravity/brain/9c0fc136-8be6-4471-bac4-5d024a1c789f"
    if os.path.exists(artifact_dir):
        dest_path = os.path.join(artifact_dir, "auto_research_chart.png")
        import shutil
        shutil.copy(output_path, dest_path)
        print(f"✅ Copied chart to artifacts directory: {dest_path}")

if __name__ == "__main__":
    main()
