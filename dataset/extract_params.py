import os
import random
import re
import numpy as np

# Extract and calculate average reward from training_rollout.txt files


for i in [1, 2, 3, 4, 5]:
    file = f"ablation-logs/idp-logs/B5R65/trial_{i}"
    # reward_file_lines = open(f"logs/cartpole/cartpole_propsp_10_trials/trial_{i}/overall_log.txt").readlines()[1:]

    with open("cp-params.txt", "a") as param_file:
        for i, filename in enumerate(os.listdir(file)):
            all_rewards = []
            if filename.startswith("episode"):
                with open(os.path.join(file, filename, "training_rollout.txt"), "r") as f:
                    try:
                        lines = f.readlines()
                        # Extract all "Total reward:" values
                        for line in lines:
                            if "Total reward:" in line:
                                # print("hi")
                                match = re.search(r'Total reward:\s*([-\d.]+)', line.strip())
                                # print(match.group(1))
                                if match:
                                    all_rewards.append(float(match.group(1)))
                        param_file.write(f"{lines[0].strip()} | {np.mean(all_rewards)}\n")
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")
                        continue

# The above code extracts parameters and their corresponding rewards from log files and writes them to a new file.
# Want to categorize based on reward ranges? 0-25%, 25-50%, 50-75%, 75-100%
with open("cp-params-categorized.txt", "w") as categorized_file:
    rewards = []
    with open("cp-params.txt", "r") as param_file:
        for line in param_file:
            parts = line.strip().split(" | ")
            if len(parts) == 2:
                param, reward_str = parts
                try:
                    reward = float(reward_str)
                    rewards.append((param, reward))
                except ValueError:
                    continue

    if not rewards:
        print("No valid rewards found.")
        exit(1)

    min_reward = min(r[1] for r in rewards)
    max_reward = max(r[1] for r in rewards)
    range_size = (max_reward - min_reward) / 4

    categories = {'1': [], '2': [], '3': [], '4': []}
    for param, reward in rewards:
        if reward <= min_reward + range_size:
            category = "1"
        elif reward <= min_reward + 2 * range_size:
            category = "2"
        elif reward <= min_reward + 3 * range_size:
            category = "3"
        else:
            category = "4"
        categorized_file.write(f"{param} | {reward} | {category}\n")
        categories[category].append((param, reward))

# Sample 100 params from each category
sampled_params = []
samples_per_category = 130

for cat, items in categories.items():
    if len(items) >= samples_per_category:
        sampled_params.extend(random.sample(items, samples_per_category))
    else:
        print(f"Warning: Category {cat} has only {len(items)} items, taking all.")
        sampled_params.extend(items)

with open("cp-params-sampled.txt", "w") as sampled_file:
    for param, reward in sampled_params:
        sampled_file.write(f"{param} | {reward}\n")


