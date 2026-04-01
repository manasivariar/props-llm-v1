import os
import numpy as np
from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from agent.policy.linear_policy import LinearPolicy
from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.factsheet_agent import StaticFactsheetAgent

class StaticFactsheetRunner:
    def __init__(self, config):
        self.config = config
        self.env_name = config["gym_env_name"]
        self.short_name = self.env_name.split('-')[0].lower()
        
        self.log_dir = config.get("logdir", f"logs/static_factsheet_{self.short_name}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.dim_actions = config["dim_actions"]
        self.dim_states = config["dim_states"]
        self.bias = config.get("bias", False)
        self.param_count = self.dim_actions * self.dim_states + (self.dim_actions if self.bias else 0)
        
        self.world = ContinualSpaceGeneralWorld(
            gym_env_name=self.env_name,
            render_mode=config.get("render_mode", None),
            max_traj_length=config.get("max_traj_length", 1000)
        )
        
        self.agent = StaticFactsheetAgent(config)
        self.factsheet_path = config.get("initial_factsheet_path", "logs/rlm_hopper_4/factsheet.md")
        self.static_factsheet = ""
        
        # In-Context Memory Buffer
        self.recent_experiences = []
        self.max_experience_buffer = 100 # Keep only the last 100 to manage context length

    def load_factsheet(self):
        if not os.path.exists(self.factsheet_path):
            raise FileNotFoundError(f"Could not find {self.factsheet_path}.")
        with open(self.factsheet_path, "r", encoding="utf-8") as f:
            self.static_factsheet = f.read()
        print(f"Successfully loaded Static Factsheet ({len(self.static_factsheet)} chars).")

    def execute_in_gym(self, params, n_runs=20, rollout_path=None):
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
            if f: f.write(f"Run {r_idx+1}: {accu_reward:.4f}\n")
            
        avg_reward = np.mean(run_rewards)
        if f: 
            f.write(f"Average: {avg_reward:.4f}\n")
            f.close()
            
        return avg_reward

    def run(self):
        print("\n=== STARTING STATIC FACTSHEET PREDICTION PIPELINE ===")
        self.load_factsheet()
        
        overlog_path = os.path.join(self.log_dir, "overall_log.txt")
        with open(overlog_path, "w") as f:
            f.write("Iteration, Actual, Predicted\n")
            
        n_iters = self.config.get("num_episodes", 100)
        
        for i in range(1, n_iters + 1):
            print(f"\n==============================================")
            print(f"=== Iteration {i}/{n_iters} ===")
            print(f"  Experience Buffer Size: {len(self.recent_experiences)}")
            
            iter_dir = os.path.join(self.log_dir, f"iteration_{i}")
            os.makedirs(iter_dir, exist_ok=True)
                
            # Generate Unseen Target Params
            target_params = np.round(np.random.uniform(-3.0, 3.0, size=self.param_count), 3).tolist()
            with open(os.path.join(iter_dir, "parameters.txt"), "w") as f:
                f.write(str(target_params))
                
            # 1. Prediction Phase (Factsheet + Improvisation Buffer)
            pred_prompt, pred_resp, pred_reward, analysis_text, reasoning = self.agent.predict_reward(
                self.static_factsheet, 
                self.recent_experiences, 
                target_params
            )
            
            # Save the full raw prompt and response
            with open(os.path.join(iter_dir, "prediction_log.txt"), "w", encoding="utf-8") as f:
                f.write("=== PROMPT ===\n"); f.write(pred_prompt)
                f.write("\n\n=== RESPONSE ===\n"); f.write(pred_resp)
                
            # NEW: Save the isolated Analysis block!
            with open(os.path.join(iter_dir, "analysis.txt"), "w", encoding="utf-8") as f:
                f.write(analysis_text)
                f.write("\n\n=== REASONING ===\n")
                f.write(reasoning)
                
            # 2. Execution Phase (Ground Truth Reality Check)
            print("  Evaluating in Gymnasium (20 runs)...")
            actual_reward = self.execute_in_gym(
                target_params, 
                n_runs=20, 
                rollout_path=os.path.join(iter_dir, "training_rollout.txt")
            )
            print(f"  Predicted: {pred_reward:.2f} | Actual: {actual_reward:.2f}")

            # 3. Memory Update Phase
            experience_entry = {
                "params": [round(p, 2) for p in target_params],
                "predicted": round(pred_reward, 2),
                "actual": round(actual_reward, 2),
                "error": round(actual_reward - pred_reward, 2)
            }
            self.recent_experiences.append(experience_entry)
            
            if len(self.recent_experiences) > self.max_experience_buffer:
                self.recent_experiences.pop(0)
                
            with open(overlog_path, "a") as f:
                f.write(f"{i}, {actual_reward:.2f}, {pred_reward:.2f}\n")