#!/usr/bin/env python3
import os
import sys
import json
import argparse
import time
import yaml
import jax
import jax.numpy as jnp
import optax
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer

def load_data(file_path):
    """Loads JSONL instruction dataset."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def format_prompt(record):
    """Formats dolly record into Qwen template."""
    inst = record.get("instruction", "")
    ctx = record.get("context", "")
    resp = record.get("response", "")
    
    prompt = f"<|im_start|>user\n{inst}\n{ctx}<|im_end|>\n<|im_start|>assistant\n"
    full_text = f"{prompt}{resp}<|im_end|>"
    return prompt, full_text

def prepare_batches(data, tokenizer, max_length=512):
    """Tokenizes dataset and yields batches."""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for record in data:
        _, full_text = format_prompt(record)
        enc = tokenizer(
            full_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        
        # In causal LM, labels are shifted input_ids
        labels = input_ids.copy()
        # Set padding tokens to -100 so they are ignored in cross entropy loss
        labels[attention_mask == 0] = -100
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
        
    return {
        "input_ids": jnp.array(input_ids_list),
        "attention_mask": jnp.array(attention_mask_list),
        "labels": jnp.array(labels_list)
    }

def main():
    parser = argparse.ArgumentParser(description="JAX TPU Causal LM Fine-Tuning Execution Script.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        print(f"❌ FATAL: Config file '{args.config}' is empty or invalid YAML. Aborting.")
        sys.exit(1)

    run_id = config.get("run_id", "run_001")
    model_id = config["model"]["base_model_id"]
    lr = float(config["hyperparameters"]["learning_rate"])
    batch_size = int(config["hyperparameters"]["batch_size"])
    epochs = int(config["hyperparameters"].get("epochs", 1))
    max_len = int(config["dataset"].get("max_seq_length", 512))
    
    print(f"JAX Devices detected: {jax.devices()}")
    print(f"TPU Cores: {jax.device_count()}")
    
    # Load tokenizer and model
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading Flax Model Configuration: {model_id}")
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_id)
    # Override context length to prevent massive OOM causal mask allocation (128k -> 2k)
    model_config.max_position_embeddings = 2048
    
    print(f"Loading Flax Model: {model_id}")
    model = FlaxAutoModelForCausalLM.from_pretrained(model_id, config=model_config, dtype=jnp.bfloat16)
    params = model.params
    
    # Load dataset
    train_path = config["dataset"]["train_path"]
    print(f"Loading dataset: {train_path}")
    train_data = load_data(train_path)
    print(f"Total training samples: {len(train_data)}")
    
    # Prepare batches
    batches = prepare_batches(train_data, tokenizer, max_length=max_len)
    
    # Setup learning rate schedule with warmup
    warmup_steps = int(config["hyperparameters"].get("warmup_steps", 0))
    if warmup_steps > 0:
        warmup_sched = optax.linear_schedule(0.0, lr, warmup_steps)
        constant_sched = optax.constant_schedule(lr)
        lr_schedule = optax.join_schedules([warmup_sched, constant_sched], [warmup_steps])
    else:
        lr_schedule = lr

    # Build optimizer chain
    tx_list = []
    
    # 1. Gradient clipping
    max_grad_norm = float(config["safety"].get("max_grad_norm", 1.0))
    tx_list.append(optax.clip_by_global_norm(max_grad_norm))
    
    # 2. Weight decay
    weight_decay = float(config["hyperparameters"].get("weight_decay", 0.0))
    if weight_decay > 0.0:
        tx_list.append(optax.add_decayed_weights(weight_decay))
        
    # 3. Base optimizer (Adafactor)
    tx_list.append(optax.adafactor(learning_rate=lr_schedule))
    
    tx = optax.chain(*tx_list)
    
    # 4. Gradient Accumulation setup (manual, no MultiSteps)
    grad_accum = int(config["hyperparameters"].get("gradient_accumulation_steps", 1))

    print(f"Initializing JAX Optax Optimizer (Adafactor Chain: Warmup={warmup_steps}, WD={weight_decay}, Clip={max_grad_norm}, Accum={grad_accum})")
    opt_state = tx.init(params)
    
    # Loss function
    def loss_fn(model_params, input_ids, attention_mask, labels):
        # Forward pass
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=model_params,
            train=True
        ).logits
        
        # Shift logits and labels for causal LM loss
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        
        # Compute cross entropy loss ignoring padding (-100 labels)
        mask = (shift_labels != -100)
        
        # One-hot representation of labels
        vocab_size = shift_logits.shape[-1]
        flat_logits = jnp.reshape(shift_logits, (-1, vocab_size))
        flat_labels = jnp.reshape(shift_labels, (-1,))
        flat_mask = jnp.reshape(mask, (-1,))
        
        # Softmax cross entropy
        loss = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_labels)
        loss = jnp.sum(loss * flat_mask) / jnp.maximum(jnp.sum(flat_mask), 1)
        return loss

    # JIT-compiled train step (single micro-step)
    @jax.jit
    def train_step(model_params, optimizer_state, input_ids, attention_mask, labels):
        loss_val, grads = jax.value_and_grad(loss_fn)(model_params, input_ids, attention_mask, labels)
        updates, next_opt_state = tx.update(grads, optimizer_state, model_params)
        next_params = optax.apply_updates(model_params, updates)
        return next_params, next_opt_state, loss_val

    # JIT-compiled accumulation step (accumulates grads without updating)
    @jax.jit
    def accum_step(model_params, acc_grads, input_ids, attention_mask, labels):
        loss_val, grads = jax.value_and_grad(loss_fn)(model_params, input_ids, attention_mask, labels)
        acc_grads = jax.tree_util.tree_map(lambda a, g: a + g, acc_grads, grads)
        return acc_grads, loss_val

    # JIT-compiled apply step (applies accumulated grads)
    @jax.jit
    def apply_accum(model_params, optimizer_state, acc_grads, n):
        avg_grads = jax.tree_util.tree_map(lambda g: g / n, acc_grads)
        updates, next_opt_state = tx.update(avg_grads, optimizer_state, model_params)
        next_params = optax.apply_updates(model_params, updates)
        return next_params, next_opt_state

    # Training execution loop
    print("Beginning JIT compilation and training loop...")
    num_samples = len(train_data)
    step = 0
    loss_history = []
    start_time = time.time()
    micro_losses = []
    acc_grads = None
    micro_step = 0

    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1} ---")
        for i in range(0, num_samples, batch_size):
            # Slice batch
            batch_input_ids = batches["input_ids"][i:i+batch_size]
            batch_att_mask = batches["attention_mask"][i:i+batch_size]
            batch_labels = batches["labels"][i:i+batch_size]

            if len(batch_input_ids) < batch_size:
                continue  # Skip partial batches

            if grad_accum <= 1:
                # Simple single-step update
                params, opt_state, loss_val = train_step(
                    params, opt_state, batch_input_ids, batch_att_mask, batch_labels
                )
                loss_val.block_until_ready()
                current_loss = float(loss_val)
                step += 1
            else:
                # Manual gradient accumulation
                if acc_grads is None:
                    # Initialize zero grads using params structure (same pytree shape as grads)
                    acc_grads = jax.tree_util.tree_map(jnp.zeros_like, params)

                acc_grads, loss_val = accum_step(
                    params, acc_grads, batch_input_ids, batch_att_mask, batch_labels
                )
                micro_losses.append(float(loss_val))
                micro_step += 1

                if micro_step >= grad_accum:
                    params, opt_state = apply_accum(params, opt_state, acc_grads, grad_accum)
                    current_loss = sum(micro_losses) / len(micro_losses)
                    acc_grads = None
                    micro_step = 0
                    micro_losses = []
                    step += 1
                else:
                    continue  # Keep accumulating, don't log yet

            step_start = time.time()
            loss_history.append(current_loss)
            print(f"Step {step} - Loss: {current_loss:.4f}")

            
    total_duration = time.time() - start_time
    final_loss = loss_history[-1] if loss_history else 999.0
    print(f"Training completed successfully in {total_duration:.2f}s (steps completed: {step}, final_loss: {final_loss:.4f})")
    
    # Save train metrics JSON (used by orchestrator for HP comparison)
    metrics_dir = f"runs/{run_id}"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics = {
        "run_id": run_id,
        "final_loss": final_loss,
        "initial_loss": loss_history[0] if loss_history else 999.0,
        "loss_history": loss_history,
        "total_steps": step,
        "duration_seconds": total_duration
    }
    metrics_path = os.path.join(metrics_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved train metrics: {metrics_path}")
    
    # Save parameters
    out_dir = f"runs/{run_id}/checkpoints/final"
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "flax_model.msgpack")
    print(f"Saving parameter dict checkpoint to: {checkpoint_path}")
    
    # Save JAX parameters using Flax msgpack serialization
    from flax import serialization
    bytes_data = serialization.to_bytes(params)
    with open(checkpoint_path, "wb") as f:
        f.write(bytes_data)
        
    print("✅ Model checkpoint saved.")

if __name__ == "__main__":
    main()
