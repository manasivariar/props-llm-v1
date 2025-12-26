# path = "propsR-log/mountaincar/mountaincar_propsR/trial_3/overall_log.txt"
# path = "propsR-log/cartpole/cartpole_propsR/trial_4/overall_log.txt"
path = "propsR-log/inverted_double_pendulum/idp_propsR/trial_4/overall_log.txt"

# load the csv into a dataframe and split it into 4 equal part and store then in a dictionary
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import io

data = {}
with open(path, 'r') as f:
    lines = f.readlines()
    num_lines = len(lines)
    part_size = num_lines // 4
    for i in range(4):
        part_lines = lines[i*part_size : (i+1)*part_size] if i < 3 else lines[i*part_size :]
        df = pd.read_csv(io.StringIO(''.join(part_lines)), header=None)
        df.columns = [' Step', ' Time Elapsed', ' Cumulative Reward', ' Total Steps', ' Total API Calls', ' True Reward', ' Predicted Reward', ' Confidence Score']
        # Convert numeric columns to float
        df[' True Reward'] = pd.to_numeric(df[' True Reward'], errors='coerce')
        df[' Predicted Reward'] = pd.to_numeric(df[' Predicted Reward'], errors='coerce')
        df[' Confidence Score'] = pd.to_numeric(df[' Confidence Score'], errors='coerce')
        data[f'trial_{i+1}'] = df

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"{'Trial':<10} | {'MAE':<8} | {'RMSE':<8} | {'Corr':<8} | {'Avg Conf':<10} | {'Accuracy':<10}")
print("-"*70)
for trial in list(data.keys()):
    df = data[trial]
    error = df[' Predicted Reward'] - df[' True Reward']
    abs_error = np.abs(error)
    
    mae = abs_error.mean()
    rmse = np.sqrt((error**2).mean())
    corr = df[' True Reward'].corr(df[' Predicted Reward'])
    avg_conf = df[' Confidence Score'].mean()
    
    # Calculate accuracy within a tolerance of 10
    tolerance = 50
    accurate_predictions = np.abs(error) <= tolerance
    accuracy = accurate_predictions.sum() / len(df) * 100  # percentage
    
    print(f"{trial:<10} | {mae:<8.2f} | {rmse:<8.2f} | {corr:<8.3f} | {avg_conf:<10.3f} | {accuracy:<10.1f}%")