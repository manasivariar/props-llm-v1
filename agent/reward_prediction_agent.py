from agent.policy.reward_prediction_brain import RewardPredictionBrain
import re
import json
import numpy as np
import random

class StickyRewardPredictionAgent:
    def __init__(self, config):
        self.brain_predict = RewardPredictionBrain(
            llm_model_name=config['llm_model_name'],
            template_dir=config['template_dir'],
            template_name=config.get('template_name', "reward_prediction.j2")
        )
        
        # Tools schema defined perfectly
        self.reflection_tools = [{
            "type": "function",
            "function": {
                "name": "manage_sticky_notes",
                "description": "Update the memory board of sticky notes based on your reflection of the prediction error.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operations": {
                            "type": "array",
                            "description": "List of operations to perform on the memory board.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string", 
                                        "enum": ["ADD", "UPDATE", "DELETE"]
                                    },
                                    "target_id": {
                                        "type": "integer",
                                        "description": "The ID of the note to UPDATE or DELETE. Use -1 for ADD."
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The new rule text (for ADD) or updated text (for UPDATE). Empty for DELETE."
                                    }
                                },
                                "required": ["action", "target_id", "content"]
                            }
                        }
                    },
                    "required": ["operations"],
                    "additionalProperties": False
                }
            }
        }]
        
        self.brain_reflect = RewardPredictionBrain(
            llm_model_name=config['llm_model_name'],
            template_dir=config['template_dir'],
            template_name="sticky_note.j2"
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
        if not dataset: return []
        rewards = [r for _, r in dataset]
        min_r, max_r = min(rewards), max(rewards)
        
        bins = np.linspace(min_r, max_r, 11) 
        binned_data = {i: [] for i in range(10)}
        
        for params, r in dataset:
            bin_idx = min(max(np.digitize(r, bins) - 1, 0), 9)
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

    def predict(self, context_data, target_params, sticky_notes):
        max_context_reward = max([r for _, r in context_data])
        
        prompt, raw_response, thinking, _ = self.brain_predict.predict(
            env_desc_file=self.env_desc_file,
            rank=len(target_params),
            context_data=context_data, 
            target_params=target_params,
            sticky_notes=sticky_notes,
            max_context_reward=max_context_reward
        )
        
        predicted_reward = -float('inf')
        try:
            clean_response = raw_response.replace('–', '-').replace('—', '-').replace('*', '')
            match = re.search(r"Predicted Reward:?\s*([-+]?[\d,]*\.?\d+)", clean_response, re.IGNORECASE)
            if match:
                predicted_reward = float(match.group(1).replace(',', ''))
        except Exception as e:
            print(f"  [Parse Error]: {e}")

        return prompt, raw_response, predicted_reward, max_context_reward, thinking

    def reflect(self, target_params, predicted_reward, actual_reward, previous_reasoning, current_notes):
        prompt, raw_response, thinking, tool_calls = self.brain_reflect.predict(
            env_desc_file=self.env_desc_file,
            rank=len(target_params),
            target_params=target_params,
            predicted_reward=predicted_reward,
            actual_reward=actual_reward,
            previous_reasoning=previous_reasoning,
            sticky_notes=current_notes,
            tools=self.reflection_tools,
            tool_choice={"type": "function", "function": {"name": "manage_sticky_notes"}}
        )
        
        operations = []
        try:
            # 1. Primary: Extract from Tool Calls
            if tool_calls:
                for tc in tool_calls:
                    if tc.function.name == "manage_sticky_notes":
                        args = json.loads(tc.function.arguments)
                        operations = args.get("operations", [])
            # 2. Fallback: If model ignored tool choice and just output text
            elif "**Operations:**" in raw_response:
                ops_text = raw_response.split("**Operations:**")[-1].strip()
                for line in ops_text.split('\n'):
                    line = line.strip().replace('*', '')
                    if line.startswith("ADD:"):
                        operations.append({"action": "ADD", "content": line[4:].strip()})
                    elif line.startswith("UPDATE"):
                        match = re.match(r"UPDATE\s+(\d+):(.*)", line, re.IGNORECASE)
                        if match:
                            operations.append({"action": "UPDATE", "target_id": int(match.group(1)), "content": match.group(2).strip()})
                    elif line.startswith("DELETE"):
                        match = re.match(r"DELETE\s+(\d+)", line, re.IGNORECASE)
                        if match:
                            operations.append({"action": "DELETE", "target_id": int(match.group(1))})
        except Exception as e:
            print(f"  [Parse Error in Reflection]: {e}")
            
        return prompt, raw_response, operations