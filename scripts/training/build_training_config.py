#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from finetuning_lifecycle.config import build_experiment_config, validate_training_config
from finetuning_lifecycle.reporting import save_manifest

def main():
    parser = argparse.ArgumentParser(description="Generate and validate experiment training configs with full coordinate overrides.")
    parser.add_argument("--run-id", type=str, default="run_001", help="Unique identifier for the run.")
    
    # 10 Coordinate Ascent Hyperparameters
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--warmup-steps", type=int, help="Override warmup steps.")
    parser.add_argument("--weight-decay", type=float, help="Override weight decay.")
    parser.add_argument("--grad-accum", type=int, help="Override gradient accumulation steps.")
    parser.add_argument("--lora-r", type=int, help="Override LoRA rank (r).")
    parser.add_argument("--lora-alpha", type=int, help="Override LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, help="Override LoRA dropout.")
    parser.add_argument("--max-grad-norm", type=float, help="Override max gradient norm clipping.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--adam-beta1", type=float, help="Override adam/adafactor beta1 momentum.")
    
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    args = parser.parse_args()

    # Build overrides dictionary
    overrides = {}
    hyperparams = {}
    peft = {}
    safety = {}
    
    if args.lr is not None:
        hyperparams["learning_rate"] = args.lr
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size
    if args.epochs is not None:
        hyperparams["epochs"] = args.epochs
    if args.warmup_steps is not None:
        hyperparams["warmup_steps"] = args.warmup_steps
    if args.weight_decay is not None:
        hyperparams["weight_decay"] = args.weight_decay
    if args.grad_accum is not None:
        hyperparams["gradient_accumulation_steps"] = args.grad_accum
        
    if args.lora_r is not None:
        peft["r"] = args.lora_r
    if args.lora_alpha is not None:
        peft["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        peft["lora_dropout"] = args.lora_dropout
        
    if args.max_grad_norm is not None:
        safety["max_grad_norm"] = args.max_grad_norm
        
    if args.adam_beta1 is not None:
        hyperparams["adam_beta1"] = args.adam_beta1

    if hyperparams:
        overrides["hyperparameters"] = hyperparams
    if peft:
        overrides["peft"] = peft
    if safety:
        overrides["safety"] = safety

    print(f"Building configuration for run: {args.run_id}...")
    try:
        config = build_experiment_config(args.run_id, overrides)
        
        # Validate Consolidated Configuration
        validate_training_config(config)
        print("✅ Consolidated configuration validated against schema.")
        
        # Save yaml configuration
        exp_dir = os.path.join("configs", "experiments")
        os.makedirs(exp_dir, exist_ok=True)
        config_yaml_path = os.path.join(exp_dir, f"{args.run_id}.yaml")
        
        with open(config_yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✅ Saved YAML config to {config_yaml_path}")

        # Save experiment manifest
        manifest = {
            "run_id": args.run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "configured",
            "config_path": config_yaml_path,
            "base_model": config["model"]["base_model_id"],
            "dataset_report_path": f"runs/{args.run_id}/dataset_report.json",
            "steps_completed": 0,
            "active_stage": "training-configuration"
        }
        
        run_dir = os.path.join("runs", args.run_id)
        os.makedirs(run_dir, exist_ok=True)
        manifest_path = os.path.join(run_dir, "experiment_manifest.json")
        save_manifest(manifest, manifest_path, "experiment_manifest.schema.json")
        print(f"✅ Saved experiment manifest: {manifest_path}")

        # Create training config summary markdown
        summary_path = os.path.join(run_dir, "training_config_summary.md")
        with open(summary_path, "w") as f:
            f.write(f"""# Training Configuration Summary: {args.run_id}
            
## Architecture Profile
* **Base Model**: {config['model']['base_model_id']}
* **LoRA Active**: {config['peft'].get('use_lora', False)}
* **LoRA Rank**: {config['peft'].get('r', 'N/A')}
* **LoRA Alpha**: {config['peft'].get('lora_alpha', 'N/A')}
* **LoRA Dropout**: {config['peft'].get('lora_dropout', 'N/A')}

## Hyperparameter Setpoints
* **Learning Rate**: {config['hyperparameters']['learning_rate']}
* **Batch Size**: {config['hyperparameters']['batch_size']}
* **Epochs**: {config['hyperparameters']['epochs']}
* **Warmup Steps**: {config['hyperparameters'].get('warmup_steps', 'N/A')}
* **Weight Decay**: {config['hyperparameters'].get('weight_decay', 'N/A')}
* **Gradient Accumulation**: {config['hyperparameters'].get('gradient_accumulation_steps', 'N/A')}

## Hardware Context
* **Accelerator target**: {config['tpu']['accelerator_type']}
* **Target zone**: {config['tpu']['zone']}
""")
        print(f"✅ Saved human readable summary: {summary_path}")

    except Exception as e:
        print(f"❌ Failed to build or validate config: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
