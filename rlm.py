import os
import re
import io
import contextlib
import traceback
import socket
from openai import OpenAI

# ==========================================
# 1. The Persistent REPL Environment
# ==========================================
class PersistentREPL:
    def __init__(self):
        self.namespace = {}
        self.execute("import numpy as np\nimport math\nimport collections")

    def execute(self, code: str) -> str:
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(code, self.namespace)
        except Exception:
            traceback.print_exc(file=output)
            
        res = output.getvalue()
        if not res.strip():
            res = "Code executed successfully with no output."
            
        # Truncation ONLY applies to REPL execution output to protect context window
        if len(res) > 5000:
            return res[:4980] + "\n...[TRUNCATED TO 5000 CHARS]"
        return res

# ==========================================
# 2. RLM Agent Class
# ==========================================
class RLMAgent:
    def __init__(self, dataset_path: str, llm_model_name: str = "gpt-oss:120b", log_dir: str = "logs/rlm_analysis"):
        self.dataset_path = dataset_path
        self.llm_model_name = llm_model_name
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup Connection
        if self.llm_model_name == 'gpt-oss:120b':
            host_node = socket.gethostname()
            asurite_id = "mrajanva" 
            self.client = OpenAI(
                base_url=f"http://{asurite_id}@{host_node}:11434/v1",
                api_key="ollama"              
            )
        else:
            self.client = OpenAI()

        self.repl = PersistentREPL()
        self.history = []
        
        self._load_dataset_to_repl()

    def _load_dataset_to_repl(self):
        print(f"Loading dataset from {self.dataset_path}...")
        dataset = []
        try:
            with open(self.dataset_path, 'r') as f:
                for line in f:
                    if "|" in line:
                        parts = line.strip().split('|')
                        try:
                            p_str = parts[0].strip().replace('[','').replace(']','')
                            params = [float(x) for x in p_str.split(',') if x.strip()]
                            reward = float(parts[1].strip())
                            dataset.append((params, reward))
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Error loading dataset: {e}")
            
        if not dataset:
            raise ValueError("Dataset is empty or could not be parsed.")
            
        self.repl.namespace['DATASET'] = dataset
        self.param_dim = len(dataset[0][0])
        self.dataset_length = len(dataset)
        print(f"Loaded {self.dataset_length} records. Param dimension: {self.param_dim}.")

    def get_system_prompt(self) -> str:
        return f"""You are a Recursive Language Model (RLM) acting as an Autonomous Knowledge Extractor.
Your goal is to analyze a Reinforcement Learning dataset and produce a highly factual, grounded factsheet.

CRITICAL API OVERRIDE:
You are communicating over an API that will crash if you attempt to use native function calling, `<tool_call>`, `<|python|>`, or any built-in tool tokens.
You MUST NOT use any internal tool or function-calling mechanisms. You must output your response as pure, standard text.

CRITICAL AUDIENCE CONSTRAINT:
The final reader of your factsheet is another standard LLM that CANNOT execute code, CANNOT run regressions, and HAS NO software tools. The reader must be able to calculate the exact reward for an unseen parameter array JUST by reading the mathematical rules and logic in your factsheet.

Therefore, your factsheet MUST NOT contain instructions like "train a Random Forest" or "use OLS". 
Instead, YOU must do the modeling now using your REPL, and extract the explicit mathematical formulas, exact weights, and IF/THEN logic rules into the factsheet.

# The Environment
You have access to a persistent Python REPL.
A variable named `DATASET` is loaded in your environment.
- `DATASET` is a Python list of tuples: `[(params_list, reward_float), ...]`
- Total samples: {self.dataset_length}
- Number of parameters per sample: {self.param_dim}
- `numpy` is pre-imported as `np`. (You can import sklearn or scipy if needed).

# What the Factsheet MUST Contain:
1. **Explicit Linear Formulas:** E.g., `Reward = 12.5 + (4.2 * params[0]) - (1.1 * params[15])`. You must calculate these exact top weights in your REPL and provide the concrete algebraic equation.
2. **Critical Thresholds (Decision Logic):** E.g., `IF params[4] < -0.5 THEN reward caps at 10.0`. You must write code to find these exact split points and write them as plain-text rules.
3. **Dominant Features:** Tell the reader exactly which indices control the reward and which indices can be safely ignored.
4. **Step-by-Step Prediction Guide:** A sequential, text-based algorithmic guide instructing the reader exactly how to do the math in their head to arrive at the final scalar reward.

# Available Actions
You MUST respond with exactly ONE of the following action blocks per turn:

1. To run analysis code:
ACTION: execute_python
```python
# Code here to run linear regressions, decision trees, or statistical checks.
# print() the weights, thresholds, and equations so you can read them.
```
2. To ask a sub-instance of yourself a conceptual question:
ACTION: call_sub_rlm
PROMPT: [Your question here]

3. When you have extracted the exact formulas and rules:
ACTION: final
<markdown>
[Factsheet Content with explicit math formulas and IF/THEN rules]
</markdown>

First, think about what code you need to write to extract concrete formulas and decision splits. Write your thoughts as plain text, then end your response with exactly one ACTION block.
"""

    def call_sub_rlm(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": "You are a sub-module of a Data Science AI. Answer the prompt logically and concisely."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"SubRLM Error: {str(e)}"

    def parse_action(self, text: str):
        if "ACTION: execute_python" in text:
            match = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
            return "execute_python", match.group(1).strip() if match else ""
            
        elif "ACTION: call_sub_rlm" in text:
            match = re.search(r"PROMPT:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
            return "call_sub_rlm", match.group(1).strip() if match else ""
            
        elif "ACTION: final" in text:
            # Using specific XML tags for flawless extraction of the markdown content
            match = re.search(r"<markdown>(.*?)</markdown>", text, re.DOTALL | re.IGNORECASE)
            return "final", match.group(1).strip() if match else ""
            
        return "error", "Could not parse action. Ensure you use the exact ACTION blocks specified."

    def run(self, max_turns=15):
        self.history = [{"role": "system", "content": self.get_system_prompt()}]
        self.history.append({"role": "user", "content": "Begin your rule extraction. Write Python code to find the explicit mathematical formulas and thresholds governing this dataset."})
        
        print(f"\n=== Starting RLM Autonomous Analysis ===")
        print(f"Logging all sub-iterations to: {self.log_dir}/")
        
        for turn in range(1, max_turns + 1):
            print(f"\n--- Turn {turn} ---")
            
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=self.history
                )
                message = completion.choices[0].message
                response_text = message.content or ""
                reasoning_text = getattr(message, 'reasoning', '') 
            except Exception as e:
                print(f"API Error: {e}")
                break
                
            self.history.append({"role": "assistant", "content": response_text})
            
            action_type, payload = self.parse_action(response_text)
            
            observation = ""
            if action_type == "execute_python":
                print("[Action] Executing Python Code...")
                observation = self.repl.execute(payload)
                self.history.append({"role": "user", "content": f"Execution Output:\n```\n{observation}\n```\nWhat is your next action?"})
                
            elif action_type == "call_sub_rlm":
                print("[Action] Calling Sub-RLM...")
                observation = self.call_sub_rlm(payload)
                self.history.append({"role": "user", "content": f"SubRLM Output:\n```\n{observation}\n```\nWhat is your next action?"})
                
            elif action_type == "final":
                print("\n[Action] Factsheet Generated!")
                observation = "Factsheet saved to disk."
                # Payload contains the untruncated string inside the <markdown> tags
                with open(os.path.join(self.log_dir, "factsheet.md"), "w", encoding="utf-8") as f:
                    f.write(payload)
                
            else:
                print("[Action] Parse Error. Asking to correct format.")
                observation = payload
                self.history.append({"role": "user", "content": payload})

            log_path = os.path.join(self.log_dir, f"turn_{turn:02d}.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=========================================\n")
                f.write(f"              TURN {turn:02d} LOG\n")
                f.write(f"=========================================\n\n")
                if reasoning_text:
                    f.write("=== EXPLICIT REASONING TOKENS ===\n")
                    f.write(reasoning_text.strip() + "\n\n")
                f.write("=== LLM RESPONSE (Thoughts & Action) ===\n")
                f.write(response_text.strip() + "\n\n")
                f.write(f"=== PARSED ACTION: {action_type} ===\n")
                f.write("=== PAYLOAD (Code / Prompt / Markdown) ===\n")
                f.write(payload.strip() + "\n\n")
                f.write("=== EXECUTION OUTPUT ===\n")
                f.write(observation.strip() + "\n")

            if action_type == "final":
                print(f"Analysis complete. See {log_path} for final details.")
                break
                
            if turn == max_turns:
                print("\n[!] Max turns reached. Forcing termination.")
                
if __name__ == "__main__":
    DATASET_FILE = "finalDataset/hopper_dataset.txt"

    if not os.path.exists(DATASET_FILE):
        print(f"Error: Could not find {DATASET_FILE}.")
    else:
        agent = RLMAgent(
            dataset_path=DATASET_FILE, 
            llm_model_name="gpt-oss:120b",
            log_dir="logs/rlm_hopper_4"
        )
        agent.run(max_turns=15)