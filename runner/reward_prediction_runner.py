import os
import numpy as np
import gymnasium as gym
import traceback
from agent.policy.linear_policy import LinearPolicy
from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from agent.reward_prediction_agent import SymbolicRewardAgent

class SymbolicRewardRunner:
    def __init__(self, config):
        self.config = config
        self.env_name = config["gym_env_name"]
        self.short_name = self.env_name.split('-')[0].lower()
        
        self.log_dir = config.get("logdir", f"logs/symbolic_reward_{self.short_name}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Environment Dims
        self.dim_actions = config["dim_actions"]
        self.dim_states = config["dim_states"]
        self.bias = config.get("bias", False)
        self.param_count = self.dim_actions * self.dim_states + (self.dim_actions if self.bias else 0)
        
        self.world = ContinualSpaceGeneralWorld(
            gym_env_name=self.env_name,
            render_mode=config.get("render_mode", None),  # <-- Add this line
            max_traj_length=config.get("max_traj_length", 1000)
        )
        
        self.agent = SymbolicRewardAgent(config)
        self.dataset_path = f"finalDataset/{self.short_name}_dataset.txt"
        
        # Memory
        self.current_reward_function = ""

    def load_and_stratify_dataset(self, max_examples=400):
        """Loads dataset and samples uniformly across 20 reward bins to maximize coverage."""
        print(f"Loading dataset from {self.dataset_path}...")
        raw_data = []
        try:
            with open(self.dataset_path, 'r') as f:
                for line in f:
                    if "|" in line:
                        parts = line.strip().split('|')
                        try:
                            p_str = parts[0].strip().replace('[','').replace(']','')
                            p = [round(float(x), 3) for x in p_str.split(',') if x.strip()] # Round to save tokens
                            r = round(float(parts[1].strip()), 2)
                            if len(p) == self.param_count:
                                raw_data.append((p, r))
                        except: continue
        except Exception as e: print(f"File error: {e}")
        
        if not raw_data: return []

        # Stratified sampling
        rewards = [r for _, r in raw_data]
        min_r, max_r = min(rewards), max(rewards)
        num_bins = 20
        bins = np.linspace(min_r, max_r, num_bins + 1)
        binned_data = {i: [] for i in range(num_bins)}
        
        for p, r in raw_data:
            b_idx = min(max(np.digitize(r, bins) - 1, 0), num_bins - 1)
            binned_data[b_idx].append((p, r))
            
        k_per_bin = max(1, max_examples // num_bins)
        final_sample = []
        
        for i in range(num_bins):
            import random
            if binned_data[i]:
                k = min(len(binned_data[i]), k_per_bin)
                final_sample.extend(random.sample(binned_data[i], k))
        
        # Sort by reward for easier LLM reading
        final_sample.sort(key=lambda x: x[1])
        print(f"Sampled {len(final_sample)} examples covering reward range [{min_r:.2f}, {max_r:.2f}].")
        return final_sample

    def execute_in_gym(self, params, n_runs=20, rollout_path=None):
        """Executes actual Gymnasium simulator 20 times and averages."""
        if self.bias:
            policy = LinearPolicy(dim_actions=self.dim_actions, dim_states=self.dim_states)
        else:
            policy = LinearPolicyNoBias(dim_actions=self.dim_actions, dim_states=self.dim_states)
        policy.update_policy(params)
        
        run_rewards = []
        f = open(rollout_path, "w") if rollout_path else None
        
        for r_idx in range(n_runs):
            state = self.world.reset(new_reward=False)
            state = np.expand_dims(state, axis=0)
            done = False
            accu_reward = 0
            
            while not done:
                action = policy.get_action(state.T)
                action = np.reshape(action, (1, self.dim_actions))
                action = np.tanh(action)
                state, reward, done = self.world.step(action)
                accu_reward += reward
                
            run_rewards.append(accu_reward)
            if f: f.write(f"Run {r_idx+1} Reward: {accu_reward:.4f}\n")
            
        avg_reward = np.mean(run_rewards)
        if f: 
            f.write(f"Average: {avg_reward:.4f}\n")
            f.close()
            
        return avg_reward

    def evaluate_proxy_model(self, params):
        """Compiles and executes the LLM's generated Python code securely."""
        local_scope = {}
        try:
            exec(self.current_reward_function, {}, local_scope)
            if 'predict_reward' in local_scope:
                pred = local_scope['predict_reward'](params)
                return float(pred), None
            else:
                return -9999.0, "Function 'predict_reward(params)' not found in code."
        except Exception as e:
            err = traceback.format_exc()
            return -9999.0, str(err)

    def run(self):
        # 1. Bootstrap Phase (Iteration 0)
        print("\n=== PHASE 1: BOOTSTRAPPING SYMBOLIC REWARD MODEL ===")
        dataset = self.load_and_stratify_dataset(max_examples=400)
        
        bs_prompt, bs_response, bs_code = self.agent.bootstrap(dataset)
        self.current_reward_function = bs_code
        
        with open(os.path.join(self.log_dir, "bootstrap_prompt_and_response.txt"), "w") as f:
            f.write("=== BOOTSTRAP PROMPT ===\n")
            f.write(bs_prompt)
            f.write("\n\n=== BOOTSTRAP RESPONSE ===\n")
            f.write(bs_response)
            
        print("Generated Initial Reward Function:")
        print("--------------------------------------------------")
        print(self.current_reward_function)
        print("--------------------------------------------------")
        
        overlog = open(os.path.join(self.log_dir, "overall_log.txt"), "w")
        overlog.write("Iteration, Actual, Predicted, Error, Code_Updated\n")
        
        # 2. Continuous Refinement Loop
        n_iters = self.config.get("num_episodes", 100)
        for i in range(1, n_iters + 1):
            print(f"\n=== Iteration {i}/{n_iters} ===")
            iter_dir = os.path.join(self.log_dir, f"iteration_{i}")
            os.makedirs(iter_dir, exist_ok=True)
            
            # Save current code state before evaluating
            with open(os.path.join(iter_dir, "current_model.py"), "w") as f:
                f.write(self.current_reward_function)
            
            # Generate Target
            target_params = np.round(np.random.uniform(-3.0, 3.0, size=self.param_count), 3).tolist()
            with open(os.path.join(iter_dir, "parameters.txt"), "w") as f:
                f.write(str(target_params))
            
            # Execute Proxy Model (The Python String)
            pred_reward, trace_err = self.evaluate_proxy_model(target_params)
            
            # Execute True Simulator (Gymnasium)
            print("  Evaluating in Gymnasium (20 runs)...")
            actual_reward = self.execute_in_gym(
                target_params, 
                n_runs=20, 
                rollout_path=os.path.join(iter_dir, "training_rollout.txt")
            )
            
            if trace_err:
                print(f"  [!] Code Crash Detected.")
                error = float('inf')
            else:
                print(f"  Predicted: {pred_reward:.2f} | Actual: {actual_reward:.2f}")

            # Reflection & Update Phase
            print("  Passing results to LLM for code analysis...")
            ref_prompt, ref_response, new_code = self.agent.refine(
                self.current_reward_function, target_params, pred_reward, actual_reward, trace_err
            )
            
            with open(os.path.join(iter_dir, "prompt_and_response.txt"), "w") as f:
                f.write("=== PROMPT ===\n"); f.write(ref_prompt)
                f.write("\n\n=== RESPONSE ===\n"); f.write(ref_response)
                
            code_updated = (new_code.strip() != self.current_reward_function.strip())
            if code_updated:
                print("  [✓] LLM patched the code.")
                self.current_reward_function = new_code
            else:
                print("  [-] LLM kept code identical.")
                
            overlog.write(f"{i}, {actual_reward:.2f}, {pred_reward:.2f}, {error:.2f}, {code_updated}\n")
            overlog.flush()

        overlog.close()