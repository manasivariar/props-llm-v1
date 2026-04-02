from agent.policy.reward_prediction_brain import RewardPredictionBrain
from agent.policy.nn_brain_reward import NNBrainReward
import re
import numpy as np

class RewardPredictionAgent:
    def __init__(self, config, log_base_dir):
        self.brain = RewardPredictionBrain(
            llm_model_name=config['llm_model_name'],
            template_dir=config['template_dir'],
            template_name=config['template_name']
        )
        
        raw_env_name = config['gym_env_name'].split('-')[0].lower()
        if "inverteddouble" in raw_env_name:
             self.env_desc_file = "env_descriptions/inverteddoublependulum.j2"
        elif "continuous" in raw_env_name:
             self.env_desc_file = f"env_descriptions/{raw_env_name}.j2" 
        else:
             self.env_desc_file = f"env_descriptions/{raw_env_name}.j2"
             
        self.nn_shadow = NNBrainReward(input_dim=config['rank'])
        self.candidate_accumulation_buffer = [] # To store k*10 items
        self.shadow_log_path = f"{log_base_dir}/shadow_regressor_comparison.csv"
        
        # Initialize the CSV log header
        with open(self.shadow_log_path, "w") as f:
            f.write("Iteration,True_Reward,LLM_Predicted_Reward,NN_Predicted_Reward\n")

    def predict_reward(self, train_data, query_params, training_episodes, actual_reward):
        rank = len(query_params)
        
        # --- RAG STEP ---
        query_vec = np.array(query_params)
        scored_examples = []
        
        for params, reward in train_data:
            params_vec = np.array(params)
            dist = np.linalg.norm(query_vec - params_vec)
            scored_examples.append((dist, (params, reward)))

        scored_examples.sort(key=lambda x: x[0])

        top_k = 500
        relevant_train_data = [x[1] for x in scored_examples[:top_k]]
        # ----------------
        
        current_k_context = []
        for weights, reward in relevant_train_data:
            current_k_context.append((weights, reward))
        
        # Accumulate these k candidates into our batch buffer
        self.candidate_accumulation_buffer.extend(current_k_context)
        
        if training_episodes > 1 and training_episodes % 10 == 0:
            print(f"Shadow Regressor: Training on {len(self.candidate_accumulation_buffer)} candidates...")
            self.nn_shadow.train_on_batch(self.candidate_accumulation_buffer)
            # Clear buffer after training to start collecting for the next 10 steps
            self.candidate_accumulation_buffer = []

        prompt, raw_response, thinking = self.brain.predict(
            env_desc_file=self.env_desc_file,
            rank=rank,
            train_data=relevant_train_data,
            query_params=query_params
        )
        
        nn_pred_reward = self.nn_shadow.predict(query_params)
        
        predicted_reward = 0.0
        try:
            # --- FIX: SANITIZE RESPONSE ---
            # Replace En-dash (–) and Em-dash (—) with standard hyphen (-)
            clean_response = raw_response.replace('–', '-').replace('—', '-')
            
            # Updated Regex:
            # 1. Matches "**Predicted Reward:**" or just "Predicted Reward:"
            # 2. Handles spaces
            # 3. Captures the number (including negative signs)
            match = re.search(r"Predicted Reward:?\**\s*([-+]?[\d,]*\.?\d+)", clean_response, re.IGNORECASE)
            
            if match:
                # Remove commas (e.g., if LLM writes -1,200.5)
                num_str = match.group(1).replace(',', '')
                predicted_reward = float(num_str)
            else:
                print(f"Could not parse numeric reward. Raw excerpt: {raw_response[-50:]}")
        except Exception as e:
            print(f"Error parsing reward: {e}")
            
        with open(self.shadow_log_path, "a") as f:
            f.write(f"{training_episodes},{actual_reward},{predicted_reward},{nn_pred_reward}\n")

        return prompt, raw_response, thinking, predicted_reward