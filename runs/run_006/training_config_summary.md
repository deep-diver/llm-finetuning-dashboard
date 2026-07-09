# Training Configuration Summary: run_006
            
## Architecture Profile
* **Base Model**: meta-llama/Llama-3.2-1B-Instruct
* **LoRA Active**: True
* **LoRA Rank**: 8
* **LoRA Alpha**: 16

## Hyperparameter Setpoints
* **Learning Rate**: 5e-05
* **Batch Size**: 1
* **Epochs**: 1
* **Optimizer**: optax.adamw

## Hardware Context
* **Accelerator target**: v5litepod-8
* **Target zone**: us-west4-a
