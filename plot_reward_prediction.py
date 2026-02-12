import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the base directory where logs are stored
LOG_BASE_DIR = 'logs/reward_prediction'

def process_environment(env_path):
    env_name = os.path.basename(env_path)
    overlog_file = os.path.join(env_path, 'overlog.txt')
    
    if not os.path.exists(overlog_file):
        print(f"Skipping {env_name}: overlog.txt not found.")
        return

    print(f"Processing {env_name}...")
    
    data = []
    try:
        with open(overlog_file, 'r') as f:
            lines = f.readlines()
            # Skip header if it exists
            start_idx = 1 if len(lines) > 0 and "Iteration" in lines[0] else 0
            
            for line in lines[start_idx:]:
                parts = line.split('|')
                if len(parts) >= 4:
                    try:
                        iteration = int(parts[0].strip())
                        actual_reward = float(parts[2].strip())
                        predicted_reward = float(parts[3].strip())
                        data.append({
                            'Iteration': iteration, 
                            'Actual Reward': actual_reward, 
                            'Predicted Reward': predicted_reward
                        })
                    except ValueError:
                        continue # Skip malformed lines
    except Exception as e:
        print(f"Error reading {overlog_file}: {e}")
        return

    if not data:
        print(f"No valid data found in {env_name}.")
        return

    df = pd.DataFrame(data)

    # --- 1. Calculate Statistics ---
    act_mean = df['Actual Reward'].mean()
    act_std = df['Actual Reward'].std()
    pred_mean = df['Predicted Reward'].mean()
    pred_std = df['Predicted Reward'].std()
    
    # Calculate Residuals (Errors)
    df['Error'] = df['Actual Reward'] - df['Predicted Reward']
    mae = df['Error'].abs().mean()
    rmse = np.sqrt((df['Error']**2).mean())

    # Print to console
    print(f"--- Stats for {env_name} ---")
    print(f"Actual Reward:    Mean = {act_mean:.2f}, Std Dev = {act_std:.2f}")
    print(f"Predicted Reward: Mean = {pred_mean:.2f}, Std Dev = {pred_std:.2f}")
    print(f"Prediction Error: MAE  = {mae:.2f}, RMSE = {rmse:.2f}")
    print("-" * 30)

    # --- 2. Plotting ---
    
    # Plot 1: Line plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(df['Iteration'], df['Actual Reward'], label='Actual Reward', marker='o', alpha=0.7)
    plt.plot(df['Iteration'], df['Predicted Reward'], label='Predicted Reward', marker='x', alpha=0.7)
    
    # Add stats box
    stats_text = (f"Actual: $\mu$={act_mean:.1f}, $\sigma$={act_std:.1f}\n"
                  f"Pred:   $\mu$={pred_mean:.1f}, $\sigma$={pred_std:.1f}")
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f'Actual vs Predicted Reward - {env_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save in the environment folder
    plot1_path = os.path.join(env_path, 'reward_comparison_line.png')
    plt.savefig(plot1_path)
    plt.close() # Close figure to free memory

    # Plot 2: Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(df['Actual Reward'], df['Predicted Reward'], alpha=0.6, c='blue', edgecolors='k')
    
    # Add y=x line (Ideal)
    min_val = min(df['Actual Reward'].min(), df['Predicted Reward'].min())
    max_val = max(df['Actual Reward'].max(), df['Predicted Reward'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction (y=x)')
    
    # Add error stats box
    error_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}"
    plt.gca().text(0.05, 0.95, error_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f'Actual vs Predicted Scatter - {env_name}')
    plt.xlabel('Actual Reward')
    plt.ylabel('Predicted Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal') 
    
    # Save in the environment folder
    plot2_path = os.path.join(env_path, 'reward_comparison_scatter.png')
    plt.savefig(plot2_path)
    plt.close()

    print(f"Saved plots to {env_path}\n")

def main():
    # Find all subdirectories in the log base directory
    if not os.path.exists(LOG_BASE_DIR):
        print(f"Directory {LOG_BASE_DIR} does not exist.")
        return

    # Get list of all folders in logs/reward_prediction
    env_folders = [f.path for f in os.scandir(LOG_BASE_DIR) if f.is_dir()]
    
    if not env_folders:
        print("No environment folders found.")
        return

    for env_folder in env_folders:
        process_environment(env_folder)

    print("All experiments processed.")

if __name__ == "__main__":
    main()