import os
import numpy as np
import random
import gymnasium as gym
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from agent.reward_prediction_agent import RewardPredictionAgent

class RewardPredictionRunner:
    def __init__(self, config):
        self.config = config
        self.env_name = config['gym_env_name']
        self.short_name = self.env_name.split('-')[0].lower()
        if self.short_name == "inverteddoublependulum": 
            self.short_name = "inverted_double_pendulum"
        
        self.log_dir = f"logs/reward_prediction_using_PCA/{self.short_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.agent = RewardPredictionAgent(config)
        self.dataset_path = f"final_dataset/{self.short_name}_dataset.txt"
        
        # PCA Components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=5) # <--- Compress down to Top 5 Drivers of Variance

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
        except: pass
        return data

    def execute_policy(self, env_name, params, n_runs=20):
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
                 print(f"  [Error] Dimension mismatch.")
                 env.close()
                 return -9999.0

            total_rewards = []
            print(f"  [Executing {env_name} | {n_runs} episodes]")
            
            for run_idx in range(n_runs):
                obs, _ = env.reset()
                episode_reward = 0
                done = False
                truncated = False
                step_count = 0
                max_steps = 500 # Prevent infinite loops
                
                while not (done or truncated) and step_count < max_steps:
                    if is_tabular:
                        raw_action = params[int(obs)]
                        action = int(np.clip(np.round(raw_action), 0, act_dim - 1)) if discrete_act else raw_action
                    else:
                        # FIX: Cleanly separated logic so int(obs) is never called on continuous arrays
                        if discrete_obs:
                            obs_vec = np.zeros(obs_dim)
                            obs_vec[int(obs)] = 1.0
                        else:
                            obs_vec = obs

                        action_out = W.T @ obs_vec + b
                        action = int(np.argmax(action_out)) if discrete_act else action_out

                    obs, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    step_count += 1
                
                total_rewards.append(episode_reward)
                status = " (Timed Out)" if step_count >= max_steps else ""
                print(f"    Run {run_idx + 1:02d}: {episode_reward:.4f}{status}")
            
            env.close()
            avg_reward = np.mean(total_rewards)
            print(f"  -> Averaged Reward: {avg_reward:.4f}")
            return avg_reward
            
        except Exception as e:
            print(f"Execution Error: {e}")
            import traceback
            traceback.print_exc()
            return -9999.0

    def run(self):
        # 1. Load Original High-Dimensional Data
        full_dataset = self.load_dataset()
        if not full_dataset:
            print("No dataset found. Exiting.")
            return
            
        param_dim_original = len(full_dataset[0][0])
        print(f"Loaded {len(full_dataset)} historical policies (Dimension: {param_dim_original}).")

        # 2. Fit PCA (The Translation Layer)
        X = np.array([p for p, r in full_dataset])
        rewards = [r for _, r in full_dataset]
        
        # Scale data to mean=0, variance=1 before PCA
        X_scaled = self.scaler.fit_transform(X)
        # Transform 36D -> 5D
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Rebuild dataset with PCA parameters
        pca_dataset = [(X_pca[i].tolist(), rewards[i]) for i in range(len(rewards))]
        
        # 3. Build Uniform Context (Using PCA Data)
        context = self.agent.get_uniform_context(pca_dataset, n_samples=200)
        
        overlog_path = os.path.join(self.log_dir, "overlog.txt")
        if not os.path.exists(overlog_path):
            with open(overlog_path, "w") as f:
                f.write("Iteration | Params(Original) | Params(PCA) | Actual Reward | Predicted Reward\n")

        # 4. Active Prediction Loop
        n_iterations = 20
        
        for i in range(n_iterations):
            iteration = i + 1
            print(f"\n--- DICL Iteration {iteration}/{n_iterations} ---")
            
            # A. Generate Random Target in Original Space (e.g. 36D)
            target_params_nd = np.random.uniform(-1.0, 3.5, size=param_dim_original).tolist()
            
            # B. Translate Target to PCA Space (5D)
            target_scaled = self.scaler.transform([target_params_nd])
            target_params_pca = self.pca.transform(target_scaled)[0].tolist()
            
            # Formatted for clean console printing
            clean_pca = [round(x, 2) for x in target_params_pca]
            print(f"  Translating Target: {param_dim_original}D -> 5D {clean_pca}")
            
            # C. Predict Reward using 5D Target
            prompt, response, pred_reward, _ = self.agent.predict(context, target_params_pca)
            
            # D. Execute using Original 36D Target
            actual_reward = self.execute_policy(self.env_name, target_params_nd, n_runs=20)
            print(f"  Prediction: {pred_reward:.2f} | Actual: {actual_reward:.2f}")

            # E. Logging
            iter_dir = os.path.join(self.log_dir, f"iteration{iteration}")
            os.makedirs(iter_dir, exist_ok=True)

            with open(os.path.join(iter_dir, "parameters_original.txt"), "w") as f:
                f.write(str(target_params_nd))
                
            with open(os.path.join(iter_dir, "parameters_pca.txt"), "w") as f:
                f.write(str(target_params_pca))

            with open(os.path.join(iter_dir, "reward_reasoning.txt"), "w") as f:
                f.write("SYSTEM PROMPT:\n"); f.write(prompt); f.write("\n\n")
                f.write("LLM RESPONSE:\n"); f.write(response)

            with open(overlog_path, "a") as f:
                f.write(f"{iteration} | {target_params_nd} | {clean_pca} | {actual_reward:.4f} | {pred_reward:.4f}\n")