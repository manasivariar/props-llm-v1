import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Base path for propsR logs
base_path = '/home/mrajanva/props-llm-v1/propsR-log/cartpole_propsR'
trials = ['trial_1', 'trial_2', 'trial_3', 'trial_4', 'trial_5']

# Load data from all trials
data = {}
for trial in trials:
    log_file = os.path.join(base_path, trial, 'overall_log.txt')
    if os.path.exists(log_file):
        data[trial] = pd.read_csv(log_file)
    else:
        print(f"Warning: {log_file} not found")

print(f"Loaded {len(data)} trials")

# === PLOT 1: Individual trial plots (4 subplots) ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

trial_list = list(data.keys())[:4]
for idx, trial in enumerate(trial_list):
    df = data[trial]
    ax = axes[idx]
    
    # Scatter plot with reference line
    ax.scatter(df[' True Reward'], df[' Predicted Reward'], alpha=0.6, s=50, color='steelblue')
    min_val = min(df[' True Reward'].min(), df[' Predicted Reward'].min())
    max_val = max(df[' True Reward'].max(), df[' Predicted Reward'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate correlation
    corr = df[' True Reward'].corr(df[' Predicted Reward'])
    mae = np.abs(df[' Predicted Reward'] - df[' True Reward']).mean()
    
    ax.set_xlabel('True Reward', fontsize=10)
    ax.set_ylabel('Predicted Reward', fontsize=10)
    ax.set_title(f'{trial} - Scatter Plot\nCorr: {corr:.3f}, MAE: {mae:.2f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
output_path1 = '/home/mrajanva/props-llm-v1/plots/propsR_scatter_comparison.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"Scatter comparison plot saved to {output_path1}")

# === PLOT 2: Absolute Error comparison across trials ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, trial in enumerate(trial_list):
    df = data[trial]
    ax = axes[idx]
    
    abs_error = np.abs(df[' Predicted Reward'] - df[' True Reward'])
    ax.plot(df['Iteration'], abs_error, color='orange', linewidth=1.5, marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('Absolute Error |Predicted - True|', fontsize=10)
    
    mean_abs_error = abs_error.mean()
    ax.axhline(y=mean_abs_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_abs_error:.2f}')
    ax.set_title(f'{trial} - Absolute Prediction Error', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
output_path2 = '/home/mrajanva/props-llm-v1/plots/propsR_error_comparison.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Error comparison plot saved to {output_path2}")

# === PLOT 3: Combined statistics across all trials ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Prepare statistics for all trials
stats = {}
for trial, df in data.items():
    error = df[' Predicted Reward'] - df[' True Reward']
    abs_error = np.abs(error)
    stats[trial] = {
        'MAE': abs_error.mean(),
        'RMSE': np.sqrt((error**2).mean()),
        'Correlation': df[' True Reward'].corr(df[' Predicted Reward']),
        'Mean Error': error.mean(),
        'Std Error': error.std()
    }

# Plot 3a: MAE and RMSE comparison
ax1 = axes[0]
trial_names = list(stats.keys())
mae_values = [stats[t]['MAE'] for t in trial_names]
rmse_values = [stats[t]['RMSE'] for t in trial_names]
x_pos = np.arange(len(trial_names))
width = 0.35

ax1.bar(x_pos - width/2, mae_values, width, label='MAE', color='steelblue', alpha=0.8)
ax1.bar(x_pos + width/2, rmse_values, width, label='RMSE', color='orange', alpha=0.8)
ax1.set_xlabel('Trial', fontsize=11)
ax1.set_ylabel('Error', fontsize=11)
ax1.set_title('Mean Absolute Error vs RMSE Across Trials', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(trial_names)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 3b: Correlation comparison
ax2 = axes[1]
corr_values = [stats[t]['Correlation'] for t in trial_names]
colors = ['green' if c > 0.5 else 'orange' if c > 0 else 'red' for c in corr_values]
bars = ax2.bar(trial_names, corr_values, color=colors, alpha=0.8)
ax2.set_ylabel('Correlation Coefficient', fontsize=11)
ax2.set_title('Correlation: True vs Predicted Across Trials', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylim([min(corr_values) - 0.1, 1])
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, corr_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
output_path3 = '/home/mrajanva/props-llm-v1/plots/propsR_statistics_summary.png'
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"Statistics summary plot saved to {output_path3}")

# === PRINT DETAILED STATISTICS ===
print("\n" + "="*70)
print("DETAILED STATISTICS FOR EACH TRIAL")
print("="*70)
for trial in trial_names:
    print(f"\n{trial}:")
    print(f"  Mean Absolute Error (MAE):  {stats[trial]['MAE']:.2f}")
    print(f"  Root Mean Square Error:     {stats[trial]['RMSE']:.2f}")
    print(f"  Correlation:                {stats[trial]['Correlation']:.4f}")
    print(f"  Mean Error (bias):          {stats[trial]['Mean Error']:.2f}")
    print(f"  Std Error:                  {stats[trial]['Std Error']:.2f}")

# === PLOT 4: Overlay comparison of absolute errors for all trials (trial_5 if available) ===
if 'trial_5' in data:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for trial, df in data.items():
        abs_error = np.abs(df[' Predicted Reward'] - df[' True Reward'])
        ax.plot(df['Iteration'], abs_error, linewidth=1.5, marker='o', markersize=2, alpha=0.6, label=trial)
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Absolute Error |Predicted - True|', fontsize=11)
    ax.set_title('Absolute Prediction Error - All Trials Overlay', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_path4 = '/home/mrajanva/props-llm-v1/plots/propsR_error_overlay.png'
    plt.savefig(output_path4, dpi=150, bbox_inches='tight')
    print(f"\nError overlay plot saved to {output_path4}")

print("\n" + "="*70)
print("All plots saved successfully!")
print("="*70)
