import os
import time
import socket
from openai import OpenAI
import google.generativeai as genai
import anthropic
from jinja2 import Environment, FileSystemLoader

class SymbolicRewardBrain:
    def __init__(self, llm_model_name: str, template_dir: str, template_name: str):
        self.llm_model_name = llm_model_name
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template(template_name)
        self.llm_conversation = []

        # Secure Connection Initialization
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
                asurite_id = "mrajanva"  # Replace with dynamic retrieval if needed
                self.client = OpenAI(
                    base_url=f"http://{asurite_id}@{host_node}:11434/v1",
                    api_key="ollama"              
                )
            else:
                self.client = OpenAI()

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        if self.model_group == "openai" or self.model_group == "anthropic":
            self.llm_conversation.append({"role": role, "content": text})
        else:
            self.llm_conversation.append({"role": role, "parts": text})

    def predict(self, **kwargs):
        prompt = self.template.render(**kwargs)
        self.reset_llm_conversation()
        self.add_llm_conversation(prompt, "user")
        
        response_text = ""
        for attempt in range(5):
            try:
                if self.model_group == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                    )
                    response_text = completion.choices[0].message.content
                elif self.model_group == "anthropic":
                    message = self.client.messages.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        max_tokens=4096,
                    )
                    response_text = message.content[0].text
                elif self.model_group == "gemini":
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    response = model.generate_content(prompt)
                    response_text = response.text
                break # Success
            except Exception as e:
                print(f"LLM API Error: {e}. Retrying in 10s...")
                time.sleep(10)

        return prompt, response_text