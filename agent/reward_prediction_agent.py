from agent.policy.reward_prediction_brain import RewardPredictionBrain
import re
import numpy as np
import random

class RewardPredictionAgent:
    def __init__(self, config):
        self.brain = RewardPredictionBrain(
            llm_model_name=config['llm_model_name'],
            template_dir=config['template_dir'],
            template_name="reward_prediction.j2" # <--- Uses the new DICL template
        )
        
        raw_env_name = config['gym_env_name'].split('-')[0].lower()
        if "inverteddouble" in raw_env_name:
             self.env_desc_file = "env_descriptions/inverteddoublependulum.j2"
        elif "continuous" in raw_env_name:
             self.env_desc_file = f"env_descriptions/{raw_env_name}.j2"
        elif "navigation" in raw_env_name or "nav" in raw_env_name:
             self.env_desc_file = "env_descriptions/nav.j2"
        else:
             self.env_desc_file = f"env_descriptions/{raw_env_name}.j2"

    def get_uniform_context(self, dataset, n_samples=200):
        """
        Samples the dataset uniformly across the reward range.
        Expects `dataset` to already be transformed into PCA space.
        """
        if not dataset: return []

        rewards = [r for _, r in dataset]
        min_r, max_r = min(rewards), max(rewards)
        
        bins = np.linspace(min_r, max_r, 11) 
        binned_data = {i: [] for i in range(10)}
        
        for params, r in dataset:
            bin_idx = np.digitize(r, bins) - 1
            bin_idx = min(max(bin_idx, 0), 9)
            binned_data[bin_idx].append((params, r))
            
        context = []
        k_per_bin = max(1, n_samples // 10)
        
        for i in range(10):
            available = len(binned_data[i])
            if available > 0:
                k = min(available, k_per_bin)
                context.extend(random.sample(binned_data[i], k))
        
        random.shuffle(context)
        return context

    def predict(self, context_data, target_params):
        rank = len(target_params) # This will now be the PCA rank (e.g., 5)
        max_context_reward = max([r for _, r in context_data])
        
        prompt, raw_response, thinking = self.brain.predict(
            env_desc_file=self.env_desc_file,
            rank=rank,
            context_data=context_data, 
            target_params=target_params,
            max_context_reward=max_context_reward
        )
        
        predicted_reward = -float('inf')
        try:
            clean_response = raw_response.replace('–', '-').replace('—', '-').replace('*', '')
            match = re.search(r"Predicted Reward:?\s*([-+]?[\d,]*\.?\d+)", clean_response, re.IGNORECASE)
            if match:
                predicted_reward = float(match.group(1).replace(',', ''))
            else:
                print(f"  [Warning] Could not parse reward: {clean_response[-50:]}")
        except Exception as e:
            print(f"Error parsing: {e}")

        return prompt, raw_response, predicted_reward, max_context_reward