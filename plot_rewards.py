import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read the log file
log_file = '/home/mrajanva/props-llm-v1/logs/reverse_rl_cosine_simm/cartpole/overall_log.txt'

# Read CSV file
df = pd.read_csv(log_file)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot True Reward and Predicted Reward vs Iteration
plt.plot(df['Iteration'], df[' True Reward'], label='True Reward', marker='o', linewidth=2, markersize=4)
plt.plot(df['Iteration'], df[' Predicted Reward'], label='Predicted Reward', marker='s', linewidth=2, markersize=4)

# Labels and title
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.title('Iteration vs True Reward and Predicted Reward', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Save the plot
output_path = '/home/mrajanva/props-llm-v1/plots/reward_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {output_path}")

# Also display the plot
plt.show()
