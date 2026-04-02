import os
import numpy as np
import random
from agent.reward_prediction_agent import RewardPredictionAgent

class RewardPredictionRunner:
    def __init__(self, config):
        self.config = config
        self.env_name = config['gym_env_name'].split('-')[0].lower()
        
        # Handle naming discrepancies
        if self.env_name == "inverteddoublependulum": 
            self.env_name = "inverted_double_pendulum"
        if self.env_name == "gym_navigation:navigationtrack": 
            self.env_name = "nav"
            
        self.dataset_path = f"FINAL_DATASET/{self.env_name}_dataset.txt"
        self.log_base_dir = f"logs/baseline_vs_llm/{self.env_name}"
        
        os.makedirs(self.log_base_dir, exist_ok=True)
        
        self.agent = RewardPredictionAgent(config, self.log_base_dir)

    def load_dataset(self):
        data = []
        try:
            with open(self.dataset_path, 'r') as f:
                lines = f.readlines()
                
                # Temporary list to hold all parsed lines
                temp_data = []
                lengths = {}
                
                for line in lines:
                    if "|" in line:
                        parts = line.strip().split('|')
                        params_str = parts[0].strip().replace('[', '').replace(']', '')
                        # Robust splitting: handle extra spaces or empty strings
                        params = [float(x) for x in params_str.split(',') if x.strip()]
                        reward = float(parts[1].strip())
                        
                        temp_data.append((params, reward))
                        
                        # Count frequency of parameter lengths
                        l = len(params)
                        lengths[l] = lengths.get(l, 0) + 1

                if not lengths:
                    print(f"Error: No valid data found in {self.dataset_path}")
                    return []

                # 1. Determine the "Correct" Rank (The length that appears most often)
                expected_length = max(lengths, key=lengths.get)
                print(f"Inferred Policy Rank: {expected_length}")
                print(f"Skipping {len(temp_data) - lengths[expected_length]} corrupted lines.")

                # 2. Filter the data
                for params, reward in temp_data:
                    if len(params) == expected_length:
                        data.append((params, reward))
                        
        except FileNotFoundError:
            print(f"Dataset not found at {self.dataset_path}")
            exit(1)
            
        return data

    def run(self):
        full_dataset = self.load_dataset()
        print(f"Loaded {len(full_dataset)} policies from {self.dataset_path}")
        
        # 1. Shuffle to fix distribution shift
        random.seed(42)
        random.shuffle(full_dataset)
        
        # 2. Split 70/30
        split_idx = int(len(full_dataset) * 0.7)
        train_data = full_dataset[:split_idx]
        test_data = full_dataset[split_idx:]
        
        print(f"Training Samples (Context): {len(train_data)}")
        print(f"Total Test Candidates: {len(test_data)}")

        # 3. --- LIMIT TO MAX 300 ITERATIONS ---
        max_iterations = 300
        if len(test_data) > max_iterations:
            print(f"Limiting test set to first {max_iterations} samples.")
            test_data = test_data[:max_iterations]
        # --------------------------------------

        overlog_path = os.path.join(self.log_base_dir, "overlog.txt")
        if not os.path.exists(overlog_path):
            with open(overlog_path, "w") as f:
                f.write("Iteration | Parameters | Actual Reward | Predicted Reward\n")

        # 4. Loop
        for i, (query_params, actual_reward) in enumerate(test_data):
            iteration = i + 1
            print(f"--- Iteration {iteration}/{len(test_data)} ---")

            # Predict
            prompt, raw_response, thinking, predicted_reward = self.agent.predict_reward(train_data, query_params, iteration, actual_reward)

            # Logging
            iter_dir = os.path.join(self.log_base_dir, f"iteration{iteration}")
            os.makedirs(iter_dir, exist_ok=True)

            with open(os.path.join(iter_dir, "reward_reasoning.txt"), "w") as f:
                f.write("SYSTEM PROMPT:\n")
                f.write(prompt)
                f.write("\n\n")
                f.write("LLM RESPONSE:\n")
                f.write("\n\n")
                f.write("Thinking:\n")
                f.write(thinking)
                f.write("\n\n")
                f.write(raw_response)

            with open(os.path.join(iter_dir, "parameters.txt"), "w") as f:
                f.write(str(query_params))

            with open(overlog_path, "a") as f:
                f.write(f"{iteration} | {query_params} | {actual_reward} | {predicted_reward}\n")
            
            print(f"Actual: {actual_reward:.4f} | Predicted: {predicted_reward:.4f}")