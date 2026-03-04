import os
import socket
from openai import OpenAI
import google.generativeai as genai
import anthropic
from jinja2 import Environment, FileSystemLoader

class RewardPredictionBrain:
    def __init__(self, llm_model_name: str, template_dir: str, template_name: str):
        self.llm_model_name = llm_model_name
        
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template(template_name)

        if "gemini" in llm_model_name:
            self.model_group = "gemini"
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        elif "claude" in llm_model_name:
            self.model_group = "anthropic"
            self.client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        else:
            self.model_group = "openai"
            if self.llm_model_name == 'gpt-oss:120b':
                host_node = socket.gethostname()
                asurite_id = "mrajanva"
                self.client = OpenAI(
                    base_url=f"http://{asurite_id}@{host_node}:11434/v1",
                    api_key="ollama"              
                )
            else:
                self.client = OpenAI()
    
    def add_llm_conversation(self, text, role):
        if self.model_group in ["openai", "anthropic"]:
            self.llm_conversation.append({"role": role, "content": text})
        else:
            self.llm_conversation.append({"role": role, "parts": text})

    def predict(self, env_desc_file, rank, tools=None, tool_choice=None, **kwargs):
        prompt = self.template.render(
            env_description=env_desc_file,
            rank=rank,
            **kwargs 
        )
        self.llm_conversation = []
        self.add_llm_conversation(prompt, "user")
        
        response_text = ""
        thinking = ""
        tool_calls = None
        
        try:
            if self.model_group == "openai":
                api_kwargs = {
                    "model": self.llm_model_name,
                    "messages": self.llm_conversation,
                }
                # Dynamically pass tool schemas if the Agent provides them
                if tools:
                    api_kwargs["tools"] = tools
                if tool_choice:
                    api_kwargs["tool_choice"] = tool_choice
                    
                completion = self.client.chat.completions.create(**api_kwargs)
                message = completion.choices[0].message
                
                response_text = message.content or ""
                thinking = getattr(message, 'reasoning', '')
                tool_calls = getattr(message, 'tool_calls', None)
                
            elif self.model_group == "anthropic":
                message = self.client.messages.create(
                    model=self.llm_model_name,
                    messages=self.llm_conversation,
                    max_tokens=4096,
                    temperature=0.0
                )
                response_text = message.content[0].text
                
            elif self.model_group == "gemini":
                model = genai.GenerativeModel(model_name=self.llm_model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(temperature=0.0)
                )
                response_text = response.text

        except Exception as e:
            print(f"LLM Error: {e}")
            return prompt, "Error", "Error", None

        # Now returning 4 values to perfectly bridge tools back to the Agent
        return prompt, response_text, thinking, tool_calls