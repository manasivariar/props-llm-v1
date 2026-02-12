from agent.policy.reward_prediction_brain import RewardPredictionBrain
import re
import numpy as np

class RewardPredictionAgent:
    def __init__(self, config):
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

    def predict_reward(self, train_data, query_params):
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

        prompt, raw_response, thinking = self.brain.predict(
            env_desc_file=self.env_desc_file,
            rank=rank,
            train_data=relevant_train_data,
            query_params=query_params
        )
        
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

        return prompt, raw_response, thinking, predicted_reward