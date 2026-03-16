import re
from agent.policy.reward_prediction_brain import SymbolicRewardBrain

class SymbolicRewardAgent:
    def __init__(self, config):
        self.env_desc_file = config.get("env_desc_file", "env_descriptions/hopper.j2")
        
        self.brain_bootstrap = SymbolicRewardBrain(
            llm_model_name=config['llm_model_name'],
            template_dir=config['template_dir'],
            template_name="bootstrap_symbolic.j2"
        )
        
        self.brain_refine = SymbolicRewardBrain(
            llm_model_name=config['llm_model_name'],
            template_dir=config['template_dir'],
            template_name="refine_symbolic.j2"
        )

    def extract_code(self, raw_response):
        match = re.search(r"```python(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback if the LLM forgets tags
        if "def predict_reward" in raw_response:
            lines = raw_response.split('\n')
            code_lines = [l for l in lines if l.startswith(' ') or l.startswith('def') or l.startswith('\t')]
            return '\n'.join(code_lines).strip()
        
        return "def predict_reward(params):\n    return 0.0"

    def bootstrap(self, dataset):
        prompt, response = self.brain_bootstrap.predict(
            env_description=self.env_desc_file,
            dataset=dataset
        )
        code = self.extract_code(response)
        return prompt, response, code

    def refine(self, current_code, target_params, predicted_reward, actual_reward, traceback_error=None):
        prompt, response = self.brain_refine.predict(
            env_description=self.env_desc_file,
            current_code=current_code,
            target_params=target_params,
            predicted_reward=predicted_reward,
            actual_reward=actual_reward,
            traceback_error=traceback_error
        )
        new_code = self.extract_code(response)
        return prompt, response, new_code