import matplotlib.pyplot as plt

loss_history = [
    13.8125, 13.25, 12.25, 11.4375, 7.6875, 12.1875, 11.5, 9.5625, 10.4375, 10.4375, 
    8.5625, 9.75, 9.8125, 7.09375, 9.25, 5.0, 9.75, 5.96875, 8.3125, 8.9375, 
    8.3125, 9.75, 8.625, 8.4375, 7.6875, 7.65625, 5.0, 7.90625, 8.5, 4.6875, 
    7.875, 7.78125, 6.75, 5.84375, 3.390625, 7.40625, 6.46875, 5.78125, 6.09375, 7.59375, 
    5.5, 7.59375, 7.0625, 4.59375, 6.53125, 3.21875, 7.53125, 3.421875, 5.6875, 6.1875, 
    6.15625, 7.96875, 6.25, 6.78125, 5.84375, 5.75, 2.78125, 5.40625, 6.15625, 2.34375, 
    5.1875, 5.46875, 3.890625, 2.578125, 1.234375, 5.5, 3.796875, 2.6875, 3.171875, 4.9375, 
    2.921875, 5.5625, 4.65625, 2.375, 4.71875, 0.9921875, 4.9375, 1.0390625, 2.8125, 3.609375, 
    3.6875, 6.40625, 3.234375, 4.75, 3.234375, 3.09375, 0.84765625, 2.75, 3.125, 0.5390625
]
steps = list(range(1, len(loss_history) + 1))

# Calculate Exponential Moving Average (EMA) for smoothing
smoothed_loss = []
ema = loss_history[0]
smoothing_factor = 0.8 # 80% weight to history, 20% to current
smoothed_loss.append(ema)

for loss in loss_history[1:]:
    ema = ema * smoothing_factor + loss * (1 - smoothing_factor)
    smoothed_loss.append(ema)

plt.figure(figsize=(10, 5))
# Plot raw loss with low alpha
plt.plot(steps, loss_history, color='#1f77b4', alpha=0.25, linestyle='-', label='Raw Loss')
# Plot smoothed loss with solid line
plt.plot(steps, smoothed_loss, color='#0f4c81', linewidth=2.5, label='Smoothed Loss (EMA 0.8)')

plt.title('LLaMA 3.2-1B Full Fine-Tuning Convergence', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(x=30, color='r', linestyle=':', label='Epoch 1')
plt.axvline(x=60, color='g', linestyle=':', label='Epoch 2')
plt.legend(fontsize=11)
plt.tight_layout()

# Save locally
plt.savefig('runs/run_full/full_training_loss.png', dpi=150)
print("✅ Smoothed Full training loss plot generated dynamically.")
