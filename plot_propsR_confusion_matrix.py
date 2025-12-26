import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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

# Define reward categories based on CartPole max reward (500)
# Categories: Very Low [0-100], Low [100-250], Medium [250-400], High [400-500]
def categorize_reward(reward):
    if reward < 100:
        return 'Very Low'
    elif reward < 250:
        return 'Low'
    elif reward < 400:
        return 'Medium'
    else:
        return 'High'

categories = ['Very Low', 'Low', 'Medium', 'High']

# === PLOT 1: Confusion matrices for first 4 trials ===
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

trial_list = list(data.keys())[:4]
for idx, trial in enumerate(trial_list):
    df = data[trial]
    ax = axes[idx]
    
    # Categorize rewards
    true_categories = df[' True Reward'].apply(categorize_reward)
    pred_categories = df[' Predicted Reward'].apply(categorize_reward)
    
    # Create confusion matrix
    cm = confusion_matrix(true_categories, pred_categories, labels=categories)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=categories, yticklabels=categories, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Category', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Category', fontsize=11, fontweight='bold')
    ax.set_title(f'{trial} - Confusion Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
output_path1 = '/home/mrajanva/props-llm-v1/plots/propsR_confusion_matrices.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"Confusion matrices plot saved to {output_path1}")

# === PLOT 2: Confusion matrix for trial_5 (larger) ===
if 'trial_5' in data:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    df = data['trial_5']
    true_categories = df[' True Reward'].apply(categorize_reward)
    pred_categories = df[' Predicted Reward'].apply(categorize_reward)
    
    cm = confusion_matrix(true_categories, pred_categories, labels=categories)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                xticklabels=categories, yticklabels=categories, 
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    ax.set_xlabel('Predicted Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Category', fontsize=12, fontweight='bold')
    ax.set_title('Trial 5 - Confusion Matrix (Detailed)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path2 = '/home/mrajanva/props-llm-v1/plots/propsR_trial5_confusion_matrix.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Trial 5 confusion matrix plot saved to {output_path2}")

# === PLOT 3: Normalized confusion matrices (percentage) ===
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, trial in enumerate(trial_list):
    df = data[trial]
    ax = axes[idx]
    
    # Categorize rewards
    true_categories = df[' True Reward'].apply(categorize_reward)
    pred_categories = df[' Predicted Reward'].apply(categorize_reward)
    
    # Create confusion matrix
    cm = confusion_matrix(true_categories, pred_categories, labels=categories)
    
    # Normalize by row (true category)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=categories, yticklabels=categories, 
                cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
    ax.set_xlabel('Predicted Category', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Category', fontsize=11, fontweight='bold')
    ax.set_title(f'{trial} - Normalized Confusion Matrix (%)', fontsize=12, fontweight='bold')

plt.tight_layout()
output_path3 = '/home/mrajanva/props-llm-v1/plots/propsR_confusion_matrices_normalized.png'
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"Normalized confusion matrices plot saved to {output_path3}")

# === PRINT DETAILED CLASSIFICATION REPORTS ===
print("\n" + "="*70)
print("CLASSIFICATION REPORTS FOR ALL TRIALS")
print("="*70)

for trial in list(data.keys()):
    df = data[trial]
    true_categories = df[' True Reward'].apply(categorize_reward)
    pred_categories = df[' Predicted Reward'].apply(categorize_reward)
    
    print(f"\n{'='*70}")
    print(f"{trial}")
    print(f"{'='*70}")
    print(classification_report(true_categories, pred_categories, labels=categories, zero_division=0))
    
    # Print confusion matrix counts
    cm = confusion_matrix(true_categories, pred_categories, labels=categories)
    print(f"\nConfusion Matrix Counts:")
    print(f"{'':12} | Predicted: Very Low | Low    | Medium | High")
    print(f"{'-'*60}")
    for i, true_cat in enumerate(categories):
        print(f"True {true_cat:6} | {cm[i][0]:19} | {cm[i][1]:6} | {cm[i][2]:6} | {cm[i][3]:4}")

# === SUMMARY TABLE ===
print("\n" + "="*70)
print("ACCURACY SUMMARY FOR ALL TRIALS")
print("="*70)
print(f"{'Trial':<10} | {'Accuracy':<10} | {'Precision (Avg)':<16} | {'Recall (Avg)':<14}")
print("-"*60)

for trial in list(data.keys()):
    df = data[trial]
    true_categories = df[' True Reward'].apply(categorize_reward)
    pred_categories = df[' Predicted Reward'].apply(categorize_reward)
    
    cm = confusion_matrix(true_categories, pred_categories, labels=categories)
    accuracy = np.trace(cm) / cm.sum() * 100
    
    # Calculate weighted average precision and recall
    report = classification_report(true_categories, pred_categories, labels=categories, 
                                 output_dict=True, zero_division=0)
    precision_avg = report['weighted avg']['precision'] * 100
    recall_avg = report['weighted avg']['recall'] * 100
    
    print(f"{trial:<10} | {accuracy:>8.2f}% | {precision_avg:>14.2f}% | {recall_avg:>12.2f}%")

print("\n" + "="*70)
print("All confusion matrix plots saved successfully!")
print("="*70)
