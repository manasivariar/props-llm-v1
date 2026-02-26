import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def load_dataset(dataset_path):
    data = []
    try:
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "|" in line:
                    parts = line.strip().split('|')
                    try:
                        r = float(parts[1].strip())
                        data.append(r)
                    except: continue
    except Exception as e:
        print(f"Could not load dataset: {e}")
    return data

def simulate_uniform_sampling(rewards, n_samples=200):
    if not rewards: return []
    min_r, max_r = min(rewards), max(rewards)
    bins = np.linspace(min_r, max_r, 11) 
    binned_data = {i: [] for i in range(10)}
    
    for r in rewards:
        bin_idx = np.digitize(r, bins) - 1
        bin_idx = min(max(bin_idx, 0), 9)
        binned_data[bin_idx].append(r)
        
    sampled_rewards = []
    k_per_bin = max(1, n_samples // 10)
    for i in range(10):
        available = len(binned_data[i])
        if available > 0:
            k = min(available, k_per_bin)
            sampled_rewards.extend(random.sample(binned_data[i], k))
    
    return sampled_rewards

def plot_sampling_distribution(env_name, dataset_path, save_dir):
    rewards = load_dataset(dataset_path)
    if not rewards:
        print(f"Skipping sampling plot: No data found at {dataset_path}")
        return

    sampled_rewards = simulate_uniform_sampling(rewards, n_samples=200)

    plt.figure(figsize=(14, 6))
    
    # Plot 1: Original Distribution
    plt.subplot(1, 2, 1)
    plt.hist(rewards, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Original Dataset Distribution ({len(rewards)} policies)")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)

    # Plot 2: Uniformly Sampled Distribution
    plt.subplot(1, 2, 2)
    plt.hist(sampled_rewards, bins=10, color='lightgreen', edgecolor='black')
    plt.title(f"Uniformly Sampled Context ({len(sampled_rewards)} policies)")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)

    plt.suptitle(f"{env_name.capitalize()} - Context Sampling Visualization", fontsize=16)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{env_name}_sampling_distribution.png")
    plt.savefig(plot_path)
    print(f"Saved sampling plot to: {plot_path}")
    plt.close()

def plot_predictions(env_name, overlog_path, save_dir):
    if not os.path.exists(overlog_path):
        print(f"Skipping prediction plot: No overlog found at {overlog_path}")
        return

    # Parse the overlog safely
    data = []
    with open(overlog_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]: # Skip header
            parts = line.split('|')
            if len(parts) >= 4:
                try:
                    iteration = int(parts[0].strip())
                    actual = float(parts[2].strip())
                    predicted = float(parts[3].strip())
                    data.append((iteration, actual, predicted))
                except:
                    continue

    if not data:
        print("No valid data found in overlog to plot.")
        return

    df = pd.DataFrame(data, columns=['Iteration', 'Actual Reward', 'Predicted Reward'])

    # Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(df['Actual Reward'], df['Predicted Reward'], color='purple', alpha=0.7, edgecolors='k', s=80)
    
    # Line of best fit / Ideal line
    min_val = min(df['Actual Reward'].min(), df['Predicted Reward'].min())
    max_val = max(df['Actual Reward'].max(), df['Predicted Reward'].max())
    
    # Plot perfect prediction line (y = x)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y = x)')
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(df['Actual Reward'] - df['Predicted Reward']))
    
    plt.title(f"{env_name.capitalize()} - Actual vs Predicted Reward")
    plt.xlabel("Actual Simulated Reward")
    plt.ylabel("LLM Predicted Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add text box with error stats
    plt.text(0.05, 0.95, f"Mean Absolute Error: {mae:.2f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plot_path = os.path.join(save_dir, f"{env_name}_prediction_scatter.png")
    plt.savefig(plot_path)
    print(f"Saved prediction scatter plot to: {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Short name of the environment (e.g., hopper, cliffwalking)")
    args = parser.parse_args()

    env_name = args.env
    dataset_path = f"final_dataset/{env_name}_dataset.txt"
    save_dir = f"logs/reward_prediction_uniformly_sampled/{env_name}"
    overlog_path = os.path.join(save_dir, "overlog.txt")

    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating plots for {env_name}...")
    plot_sampling_distribution(env_name, dataset_path, save_dir)
    plot_predictions(env_name, overlog_path, save_dir)
    print("Done!")

if __name__ == "__main__":
    main()