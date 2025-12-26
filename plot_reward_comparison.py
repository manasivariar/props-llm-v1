import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the log file
log_file = '/home/mrajanva/props-llm-v1/logs/reverse_rl_cosine_simm/cartpole/overall_log.txt'
df = pd.read_csv(log_file)

# Create a figure with multiple subplots for better comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Scatter plot - True Reward vs Predicted Reward
ax1 = axes[0, 0]
ax1.scatter(df[' True Reward'], df[' Predicted Reward'], alpha=0.6, s=50, color='steelblue')
# Add diagonal reference line (where true == predicted)
min_val = min(df[' True Reward'].min(), df[' Predicted Reward'].min())
max_val = max(df[' True Reward'].max(), df[' Predicted Reward'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('True Reward', fontsize=11)
ax1.set_ylabel('Predicted Reward', fontsize=11)
ax1.set_title('Scatter: True Reward vs Predicted Reward', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Prediction Error over iterations
ax2 = axes[0, 1]
error = df[' Predicted Reward'] - df[' True Reward']
ax2.plot(df['Iteration'], error, color='darkred', linewidth=1.5, marker='o', markersize=3, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.fill_between(df['Iteration'], error, 0, alpha=0.2, color='darkred')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Error (Predicted - True)', fontsize=11)
ax2.set_title('Prediction Error Over Iterations', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Absolute Error over iterations
ax3 = axes[1, 0]
abs_error = np.abs(error)
ax3.plot(df['Iteration'], abs_error, color='orange', linewidth=1.5, marker='s', markersize=3, alpha=0.7)
ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_ylabel('Absolute Error |Predicted - True|', fontsize=11)
ax3.set_title('Absolute Prediction Error Over Iterations', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
# Add mean line
mean_abs_error = abs_error.mean()
ax3.axhline(y=mean_abs_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_abs_error:.2f}')
ax3.legend()

# Plot 4: Bar-like comparison - sampled iterations
ax4 = axes[1, 1]
# Sample every 10th iteration for clarity
sample_indices = np.arange(0, len(df), 10)
x_pos = np.arange(len(sample_indices))
width = 0.35
ax4.bar(x_pos - width/2, df[' True Reward'].iloc[sample_indices], width, label='True Reward', color='steelblue', alpha=0.8)
ax4.bar(x_pos + width/2, df[' Predicted Reward'].iloc[sample_indices], width, label='Predicted Reward', color='orange', alpha=0.8)
ax4.set_xlabel('Iteration (sampled every 10th)', fontsize=11)
ax4.set_ylabel('Reward', fontsize=11)
ax4.set_title('Bar Comparison: True vs Predicted (Sampled)', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos[::3])
ax4.set_xticklabels(df['Iteration'].iloc[sample_indices][::3].astype(int))
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = '/home/mrajanva/props-llm-v1/plots/reward_comparison_detailed.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Detailed comparison plot saved to {output_path}")

# Print some statistics
print(f"\n=== Prediction Statistics ===")
print(f"Mean Absolute Error: {mean_abs_error:.2f}")
print(f"RMSE: {np.sqrt((error**2).mean()):.2f}")
print(f"Mean Error: {error.mean():.2f}")
print(f"Std Error: {error.std():.2f}")
print(f"Correlation: {df[' True Reward'].corr(df[' Predicted Reward']):.4f}")

plt.show()
