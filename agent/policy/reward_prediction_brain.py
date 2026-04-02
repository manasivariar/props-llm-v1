import os
import time
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

        # Initialize Client
        if "gemini" in llm_model_name:
            self.model_group = "gemini"
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        elif "claude" in llm_model_name:
            self.model_group = "anthropic"
            self.client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
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
    
    # Helper for history (if needed in future)
    def add_llm_conversation(self, text, role):
        if self.model_group == "openai":
            self.llm_conversation.append({"role": role, "content": text})
        elif self.model_group == "anthropic":
            self.llm_conversation.append({"role": role, "content": text})
        else:
            self.llm_conversation.append({"role": role, "parts": text})

    def predict(self, env_desc_file, rank, train_data, query_params):
        # 1. Render Prompt
        prompt = self.template.render(
            env_description=env_desc_file,
            rank=rank,
            train_data=train_data,
            query_params=query_params
        )

        self.llm_conversation = []
        self.add_llm_conversation(prompt, "user")
        
        # 2. Call LLM
        response_text = ""
        try:
            if self.model_group == "openai":
                completion = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=self.llm_conversation,
                )
                response_text = completion.choices[0].message.content
                thinking = completion.choices[0].message.reasoning
                
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
            return prompt, "Error", "Error"

        return prompt, response_text, thinking