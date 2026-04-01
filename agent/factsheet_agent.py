import re
from agent.policy.reward_prediction_brain import StaticFactsheetBrain

class StaticFactsheetAgent:
    def __init__(self, config):
        self.brain = StaticFactsheetBrain(
            llm_model_name=config['llm_model_name'],
            template_dir=config['template_dir'],
            template_name="factsheet_prompt.j2"
        )
        # Pull the environment description from config
        self.env_desc_file = config.get("env_desc_file", "env_descriptions/hopper.j2")

    def predict_reward(self, factsheet, recent_experiences, target_params):
        prompt, response, reasoning = self.brain.predict(
            env_description=self.env_desc_file,
            factsheet=factsheet,
            recent_experiences=recent_experiences,
            target_params=target_params
        )
        
        predicted_reward = -9999.0
        analysis_text = "Analysis not provided by LLM."
        
        try:
            # 1. Parse the Float
            clean_response = response.replace('–', '-').replace('—', '-').replace('*', '')
            reward_match = re.search(r"Predicted Reward:?\s*([-+]?[\d,]*\.?\d+)", clean_response, re.IGNORECASE)
            if reward_match:
                predicted_reward = float(reward_match.group(1).replace(',', ''))
                
            # 2. Parse the Analysis Block
            analysis_match = re.search(r"Analysis:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
            if analysis_match:
                analysis_text = analysis_match.group(1).strip()
                
        except Exception as e:
            print(f"  [Parse Error]: {e}")

        # Return 5 variables now, including the parsed analysis and reasoning
        return prompt, response, predicted_reward, analysis_text, reasoning