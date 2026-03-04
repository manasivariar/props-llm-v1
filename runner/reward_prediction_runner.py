import os
import numpy as np
import random
import gymnasium as gym
import time
import json
from agent.policy.linear_policy import LinearPolicy
from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.reward_prediction_agent import StickyRewardPredictionAgent

class StickyRewardPredictionRunner:
    def __init__(self, config):
        self.config = config
        self.env_name = config['gym_env_name']
        self.short_name = self.env_name.split('-')[0].lower()
        if self.short_name == "inverteddoublependulum": 
            self.short_name = "inverted_double_pendulum"
        
        self.log_dir = config.get("logdir", f"logs/sticky_prediction/{self.short_name}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.agent = StickyRewardPredictionAgent(config)
        self.dataset_path = f"final_dataset/{self.short_name}_dataset.txt"
        
        self.dim_actions = config.get("dim_actions", 3)  
        self.dim_states = config.get("dim_states", 11)   
        self.bias = config.get("bias", False)
        
        # State: The Note Memory
        self.sticky_notes = []

    def load_dataset(self):
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
        except Exception as e: 
            print(f"Could not load dataset: {e}")
        return data

    def execute_policy(self, env_name, params, n_runs=20, rollout_log_path=None, params_log_path=None):
        try:
            env = gym.make(env_name)
            if isinstance(env.observation_space, gym.spaces.Discrete):
                obs_dim = env.observation_space.n; discrete_obs = True
            else:
                obs_dim = env.observation_space.shape[0]; discrete_obs = False
                
            if isinstance(env.action_space, gym.spaces.Discrete):
                act_dim = env.action_space.n; discrete_act = True
            else:
                act_dim = env.action_space.shape[0]; discrete_act = False

            w_size = obs_dim * act_dim
            is_tabular = False
            
            if len(params) == w_size or len(params) == (w_size + act_dim):
                W = np.array(params[:w_size]).reshape(obs_dim, act_dim)
                b = np.array(params[w_size:]) if len(params) > w_size else np.zeros(act_dim)
            elif discrete_obs and len(params) == obs_dim:
                is_tabular = True
            else:
                 env.close()
                 return -9999.0

            # Log nicely formatted parameters
            if params_log_path:
                if self.bias:
                    policy = LinearPolicy(dim_actions=act_dim, dim_states=obs_dim)
                else:
                    policy = LinearPolicyNoBias(dim_actions=act_dim, dim_states=obs_dim)
                policy.update_policy(params)
                with open(params_log_path, "w") as p_file:
                    p_file.write(str(policy))

            total_rewards = []
            f = open(rollout_log_path, "w") if rollout_log_path else None
            print(f"  [Executing {env_name} | {n_runs} episodes]")
            
            for run_idx in range(n_runs):
                obs, _ = env.reset()
                episode_reward = 0
                done = False
                truncated = False
                step_count = 0
                max_steps = 1000
                
                if f:
                    f.write(f"--- Run {run_idx + 1}/{n_runs} ---\nstate | action | reward\n")
                
                while not (done or truncated) and step_count < max_steps:
                    if is_tabular:
                        raw_action = params[int(obs)]
                        action = int(np.clip(np.round(raw_action), 0, act_dim - 1)) if discrete_act else raw_action
                    else:
                        if discrete_obs:
                            obs_vec = np.zeros(obs_dim); obs_vec[int(obs)] = 1.0
                        else:
                            obs_vec = obs
                        action_out = W.T @ obs_vec + b
                        action = int(np.argmax(action_out)) if discrete_act else action_out

                    # Safe step mapping continuous logic
                    if not discrete_act:
                        action = np.tanh(action)
                    
                    obs, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    if f:
                        s_str = np.array2string(obs_vec, precision=4, separator=', ', suppress_small=True)
                        a_str = np.array2string(np.array(action).reshape(-1), precision=4, separator=', ', suppress_small=True)
                        f.write(f"{s_str} | {a_str} | {reward:.4f}\n")
                
                total_rewards.append(episode_reward)
                if f: f.write(f"Total reward for Run {run_idx + 1}: {episode_reward:.4f}\n\n")
            
            env.close()
            if f:
                f.write(f"=== AVERAGED REWARD ACROSS {n_runs} RUNS: {np.mean(total_rewards):.4f} ===\n")
                f.close()
                
            return np.mean(total_rewards)
            
        except Exception as e:
            print(f"Exec Error: {e}")
            return -9999.0

    def run(self):
        full_dataset = self.load_dataset()
        if not full_dataset:
            print("No dataset found. Exiting.")
            return
            
        param_dim = len(full_dataset[0][0])
        print(f"Loaded {len(full_dataset)} historical policies.")

        context_data = self.agent.get_uniform_context(full_dataset, n_samples=200)
        
        overlog_path = os.path.join(self.log_dir, "overlog.txt")
        if not os.path.exists(overlog_path):
            with open(overlog_path, "w") as f:
                f.write("Iteration | Actual | Predicted | Error | Notes_Count | Actions\n")

        n_iterations = self.config.get("num_episodes", 20)
        for i in range(n_iterations):
            iteration = i + 1
            print(f"\n==================================================")
            print(f"--- Sticky Iteration {iteration}/{n_iterations} ---")
            print(f"  Active Sticky Notes: {len(self.sticky_notes)}")
            
            iter_dir = os.path.join(self.log_dir, f"iteration_{iteration}")
            os.makedirs(iter_dir, exist_ok=True)
            
            target_params = np.round(np.random.uniform(-1.0, 3.5, size=param_dim), 2).tolist()
            
            # 1. Predict Phase
            pred_prompt, pred_response, pred_reward, _, thinking = self.agent.predict(context_data, target_params, self.sticky_notes)
            
            with open(os.path.join(iter_dir, "reward_reasoning.txt"), "w") as f:
                f.write("=== SYSTEM PROMPT ===\n"); f.write(pred_prompt)
                f.write("\n\n=== LLM RESPONSE ===\n"); f.write(pred_response)
                f.write("\n\n=== LLM THINKING ===\n"); f.write(thinking)
            
            # 2. Execute Phase
            rollout_file = os.path.join(iter_dir, "training_rollout.txt")
            params_file = os.path.join(iter_dir, "parameters.txt")
            actual_reward = self.execute_policy(self.env_name, target_params, n_runs=20, rollout_log_path=rollout_file, params_log_path=params_file)
            
            error = abs(pred_reward - actual_reward)
            print(f"  Predicted: {pred_reward:.2f} | Actual: {actual_reward:.2f} | Error: {error:.2f}")

            # 3. Reflection Trigger Phase
            action_log = "Maintained"
            error_threshold = max(0.20 * abs(actual_reward), 5.0)
            
            if error > error_threshold and actual_reward != -9999.0:
                print(f"  [!] High Error Detected. Triggering Memory Reconsolidation...")
                
                ref_prompt, ref_response, operations = self.agent.reflect(
                    target_params, pred_reward, actual_reward, pred_response, self.sticky_notes
                )
                
                with open(os.path.join(iter_dir, "sticky_note.txt"), "w") as f:
                    f.write("=== SYSTEM PROMPT ===\n"); f.write(ref_prompt)
                    f.write("\n\n=== LLM TOOL CALL RESPONSE ===\n"); f.write(ref_response)
                    f.write("\n\n=== PARSED OPERATIONS ===\n"); f.write(json.dumps(operations, indent=2))
                
                if operations:
                    action_log = "Updated"
                    for op in operations:
                        op_action = op.get("action", op.get("type", "")).upper()
                        op_idx = op.get("target_id", op.get("index"))
                        
                        if op_action == "ADD":
                            self.sticky_notes.append(op["content"])
                            print(f"  [+] ADDED Note: {op['content']}")
                        elif op_action == "UPDATE":
                            if op_idx is not None and 0 <= op_idx < len(self.sticky_notes):
                                print(f"  [~] UPDATED Note {op_idx}: {op['content']}")
                                self.sticky_notes[op_idx] = op["content"]
                    
                    # Sort deletions descending so indices don't shift
                    deletions = sorted(
                        [op.get("target_id", op.get("index")) for op in operations if op.get("action", op.get("type", "")).upper() == "DELETE" and op.get("target_id", op.get("index")) is not None], 
                        reverse=True
                    )
                    for idx in deletions:
                        if 0 <= idx < len(self.sticky_notes):
                            print(f"  [-] DELETED Note {idx}")
                            self.sticky_notes.pop(idx)
            else:
                print(f"  [✓] Prediction within acceptable bounds.")

            # Append Target to context for future iterations
            context_data.append((target_params, round(actual_reward, 2)))
            
            with open(overlog_path, "a") as f:
                f.write(f"{iteration} | {actual_reward:.4f} | {pred_reward:.4f} | {error:.4f} | {len(self.sticky_notes)} | {action_log}\n")