import os
import numpy as np
import random
import gymnasium as gym
import time
from agent.reward_prediction_agent import RewardPredictionAgent

class RewardPredictionRunner:
    def __init__(self, config):
        self.config = config
        self.env_name = config['gym_env_name']
        self.short_name = self.env_name.split('-')[0].lower()
        
        # Handle naming discrepancies
        if self.short_name == "inverteddoublependulum": 
            self.short_name = "inverted_double_pendulum"
        
        # Setup Logging Directory
        self.log_dir = f"logs/reward_prediction_uniformly_sampled/{self.short_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.agent = RewardPredictionAgent(config)
        self.dataset_path = f"final_dataset/{self.short_name}_dataset.txt"

    def load_dataset(self):
        # Robust loading logic
        data = []
        try:
            with open(self.dataset_path, 'r') as f:
                lines = f.readlines()
                temp = []
                lengths = {}
                for line in lines:
                    if "|" in line:
                        parts = line.strip().split('|')
                        try:
                            p_str = parts[0].strip().replace('[','').replace(']','')
                            p = [float(x) for x in p_str.split(',') if x.strip()]
                            r = float(parts[1].strip())
                            temp.append((p, r))
                            lengths[len(p)] = lengths.get(len(p), 0) + 1
                        except: continue
                
                if not lengths: return []
                mode_len = max(lengths, key=lengths.get)
                data = [x for x in temp if len(x[0]) == mode_len]
        except:
            pass
        return data

    def execute_policy(self, env_name, params, n_runs=20):
        """
        Creates environment, runs the policy n_runs times, 
        prints all rewards, and returns the average reward.
        Supports Continuous, Discrete, Linear, and Tabular spaces.
        """
        try:
            env = gym.make(env_name)
            
            # 1. Safely get Observation Dimension
            if isinstance(env.observation_space, gym.spaces.Discrete):
                obs_dim = env.observation_space.n
                discrete_obs = True
            else:
                obs_dim = env.observation_space.shape[0]
                discrete_obs = False
                
            # 2. Safely get Action Dimension
            if isinstance(env.action_space, gym.spaces.Discrete):
                act_dim = env.action_space.n
                discrete_act = True
            else:
                act_dim = env.action_space.shape[0]
                discrete_act = False

            w_size = obs_dim * act_dim
            is_tabular = False
            
            # 3. Dynamic Dimension Check
            if len(params) == w_size or len(params) == (w_size + act_dim):
                W = np.array(params[:w_size]).reshape(obs_dim, act_dim)
                b = np.array(params[w_size:]) if len(params) > w_size else np.zeros(act_dim)
            elif discrete_obs and len(params) == obs_dim:
                is_tabular = True
            else:
                 print(f"  [Error] Dimension mismatch. Params length: {len(params)}")
                 env.close()
                 return -9999.0

            total_rewards = []
            print(f"  [Executing Policy in {env_name} for {n_runs} episodes]")
            
            for run_idx in range(n_runs):
                obs, _ = env.reset()
                episode_reward = 0
                done = False
                truncated = False
                
                step_count = 0
                max_steps = 500 # <--- ADDED: Hard timeout limit to prevent infinite loops
                
                while not (done or truncated) and step_count < max_steps:
                    
                    if is_tabular:
                        # Tabular: Map parameter directly to action
                        # FIX: We scale the parameter so that random floats span all actions
                        # Assuming params are standard normally around [-2, 2], map them to [0, act_dim-1]
                        raw_action = params[int(obs)]
                        if discrete_act:
                            # Safely map to integer actions
                            action = int(np.clip(np.round(raw_action), 0, act_dim - 1))
                        else:
                            action = raw_action
                    else:
                        # Linear: Matrix Multiplication
                        if discrete_obs:
                            obs_vec = np.zeros(obs_dim)
                            obs_vec[int(obs)] = 1.0
                        else:
                            obs_vec = obs

                        action_out = W.T @ obs_vec + b
                        
                        if discrete_act:
                            action = int(np.argmax(action_out))
                        else:
                            action = action_out

                    obs, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    step_count += 1 # <--- Increment safety counter
                
                total_rewards.append(episode_reward)
                # Print out step_count to let you know if it got stuck and timed out
                status = " (Timed Out)" if step_count >= max_steps else ""
                print(f"    Run {run_idx + 1:02d}: {episode_reward:.4f} - {step_count} steps{status}")
            
            env.close()
            
            avg_reward = np.mean(total_rewards)
            formatted_rewards = [round(r, 4) for r in total_rewards]
            print(f"  -> All Rewards: {formatted_rewards}")
            print(f"  -> Averaged Reward: {avg_reward:.4f}")
            
            return avg_reward
            
        except Exception as e:
            print(f"Execution Error: {e}")
            import traceback
            traceback.print_exc()
            return -9999.0

    def run(self):
        # 1. Load Data
        full_dataset = self.load_dataset()
        print(f"Loaded {len(full_dataset)} historical policies.")
        
        if not full_dataset:
            print("No dataset found. Exiting.")
            return

        # 2. Build Uniform Context (200 Samples)
        context = self.agent.get_uniform_context(full_dataset, n_samples=200)
        max_context_reward = max([r for _, r in context])
        print(f"Context Max Reward: {max_context_reward:.2f}")

        # 3. Setup Overlog
        overlog_path = os.path.join(self.log_dir, "overlog.txt")
        if not os.path.exists(overlog_path):
            with open(overlog_path, "w") as f:
                f.write("Iteration | Parameters | Actual Reward | Predicted Reward\n")

        # 4. Active Loop
        param_dim = len(context[0][0])
        n_iterations = 20  # You can increase this number if you want more tests
        
        for i in range(n_iterations):
            iteration = i + 1
            print(f"--- Iteration {iteration}/{n_iterations} ---")
            
            # A. Generate Random Parameter Candidate
            # Expanded range to allow integer rounding up to 3 for discrete grids
            target_params = np.random.uniform(-1.0, 3.5, size=param_dim).tolist()
            
            # B. Predict Reward (LLM)
            prompt, response, pred_reward, thinking = self.agent.predict(context, target_params)
            
            # C. Always Verify (Run 20 times and average)
            print(f"  > Verifying in {self.env_name} (20 runs)...")
            actual_reward = self.execute_policy(self.env_name, target_params, n_runs=20)
            
            print(f"  Predicted: {pred_reward:.2f} | Actual: {actual_reward:.2f}")

            # D. Logging
            # 1. Create Iteration Directory
            iter_dir = os.path.join(self.log_dir, f"iteration{iteration}")
            os.makedirs(iter_dir, exist_ok=True)

            # 2. Write parameters.txt
            with open(os.path.join(iter_dir, "parameters.txt"), "w") as f:
                f.write(str(target_params))

            # 3. Write reward_reasoning.txt
            with open(os.path.join(iter_dir, "reward_reasoning.txt"), "w") as f:
                f.write("SYSTEM PROMPT:\n")
                f.write(prompt)
                f.write("\n\n")
                # We assume 'thinking' is part of response or handled by brain; 
                # if brain returns tuple (prompt, raw_response, thinking), update agent/active_reward_prediction_agent.py to return it too.
                # Currently agent returns (prompt, raw_response, predicted, max).
                # We will just write the raw response here which usually contains thinking for newer models.
                f.write("LLM RESPONSE:\n")
                f.write(response)
                f.write("\n\n")
                f.write("LLM Thinking:\n")
                f.write(thinking)

            # 4. Update Overlog
            with open(overlog_path, "a") as f:
                f.write(f"{iteration} | {target_params} | {actual_reward:.4f} | {pred_reward:.4f}\n")