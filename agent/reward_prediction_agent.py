from agent.policy.reward_prediction_brain import RewardPredictionBrain
import re
import numpy as np
import random

class RewardPredictionAgent:
    def __init__(self, config):
        self.brain = RewardPredictionBrain(
            llm_model_name=config['llm_model_name'],
            template_dir=config['template_dir'],
            template_name=config['template_name']
        )
        
        # Robust Environment Name Mapping
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
        Creates 10 bins from Min to Max reward and samples ~50 items from each.
        """
        if not dataset:
            return []

        rewards = [r for _, r in dataset]
        min_r, max_r = min(rewards), max(rewards)
        
        # 1. Create 10 Bins
        bins = np.linspace(min_r, max_r, 11) 
        binned_data = {i: [] for i in range(10)}
        
        for params, r in dataset:
            # -1 because digitize returns 1-based index usually, we want 0-based bucket
            bin_idx = np.digitize(r, bins) - 1
            bin_idx = min(bin_idx, 9) # Clip max index to 9
            bin_idx = max(bin_idx, 0) # Clip min index to 0
            binned_data[bin_idx].append((params, r))
            
        # 2. Sample Uniformly (50 per bin if n_samples=500)
        context = []
        k_per_bin = max(1, n_samples // 10)
        
        print(f"Sampling Strategy: Aiming for {k_per_bin} examples per bin (Total Target: {n_samples})")

        for i in range(10):
            available = len(binned_data[i])
            if available > 0:
                # If bin has fewer than k, take all of them. Otherwise take k random.
                k = min(available, k_per_bin)
                sampled_batch = random.sample(binned_data[i], k)
                context.extend(sampled_batch)
                # Optional: Debug print to see distribution
                # print(f"  Bin {i} ({bins[i]:.1f} - {bins[i+1]:.1f}): Took {k}/{available}")
        
        # 3. Shuffle context so LLM doesn't see ordered list (Low -> High)
        random.shuffle(context)
        print(f"Final Context Size: {len(context)}")
        return context

    def predict(self, context_data, target_params):
        rank = len(target_params)
        max_context_reward = max([r for _, r in context_data])
        
        prompt, raw_response, thinking = self.brain.predict(
            env_desc_file=self.env_desc_file,
            rank=rank,
            context_data=context_data, 
            target_params=target_params,
        )
        
        predicted_reward = -float('inf')
        try:
            # Sanitize: Replace dashes and strip ALL markdown stars
            clean_response = raw_response.replace('–', '-').replace('—', '-').replace('*', '')
            
            # Parse the number
            match = re.search(r"Predicted Reward:?\s*([-+]?[\d,]*\.?\d+)", clean_response, re.IGNORECASE)
            
            if match:
                predicted_reward = float(match.group(1).replace(',', ''))
            else:
                print(f"  [Warning] Could not parse reward. Raw string: {clean_response[-50:]}")
        except Exception as e:
            print(f"Error parsing: {e}")

        return prompt, raw_response, predicted_reward, thinking