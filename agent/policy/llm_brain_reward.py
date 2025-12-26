import time
from jinja2 import Template
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
import google.generativeai as genai
import anthropic
import socket
import os
import json
import numpy as np

class PromptOutput(BaseModel):
    predicted_reward: float = Field(description="A float number representing your estimation of the episodic reward, the policy will achieve in the invereted double pendulum environment, based on the policy's parameters.")
    confidence_score: float = Field(description="A float number representing your confidence score in your predicted_reward.")
    reasoning: Optional[str] = Field(description="A detailed explanation of your reward prediction logic.", default="")
    
class LLMBrainReward:
    def __init__(
        self,
        llm_si_template: Template,
        llm_model_name: str,
    ):
        self.llm_si_template = llm_si_template
        self.llm_conversation = []
        self.llm_model_name = llm_model_name
        self.output_schema = PromptOutput.model_json_schema()
        
        # Client Setup
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
                ollama_port = os.environ.get("OLLAMA_PORT", "11434")
                print(f"Connecting to Ollama on {host_node}:{ollama_port}")
                self.client = OpenAI(
                    base_url=f"http://{asurite_id}@{host_node}:{ollama_port}/v1",
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

    def query_llm(self):
        response = ""
        thinking = ""
        for attempt in range(5):
            try:
                if self.model_group == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                    )
                    response = completion.choices[0].message.content
                    thinking = completion.choices[0].message.to_dict().get("reasoning", "")
                    # print("LLM Thinking:\n", thinking)   
                elif self.model_group == "anthropic":
                    message = self.client.messages.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        max_tokens=1024,
                    )
                    response = message.content[0].text
                else:
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    chat_session = model.start_chat(history=self.llm_conversation[:-1])
                    response = chat_session.send_message(self.llm_conversation[-1]["parts"])
                    response = response.text
                
                # Success
                if self.model_group == "openai":
                    self.add_llm_conversation(response, "assistant")
                else:
                    self.add_llm_conversation(response, "model")
                    
                
                return response, thinking

            except Exception as e:
                print(f"Error attempt {attempt+1}/5: {e}")
                if attempt == 4:
                    print("Max retries reached. Returning failure placeholder.")
                    return "predicted_reward: 0.0, confidence: 0.0\nReasoning: CONNECTION FAILED."
                else:
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)
        return ""

    def llm_predict_reward(
        self,
        history_string,
        target_params,
        optimum,
        env_desc_file,
        rank,
        parse_prediction_func,
        step_number
    ):
        self.reset_llm_conversation()

        target_params_str = ", ".join((target_params).astype(str).tolist())
        # print(f"LLMBrainReward: Predicting reward for target parameters: {target_params_str}")
        # print(f"LLMBrainReward: History string: {history_string}")

        system_prompt = self.llm_si_template.render({
            "env_description": env_desc_file,
            "history_string": history_string,
            "target_parameters": target_params_str,
            "rank": rank,
            "optimum": optimum,
            "output_schema": json.dumps(self.output_schema, indent=2),
            "step_number": step_number
        })
        
        # print(system_prompt)

        self.add_llm_conversation(system_prompt, "user")
        
        api_start_time = time.time()
        response, thinking = self.query_llm()
        api_time = time.time() - api_start_time

        # Parse values using the function passed from Agent
        predicted_reward, confidence, clean_reasoning = parse_prediction_func(response)

        # Create Full Transcript for the text file
        full_transcript = "system:\n" + system_prompt + \
            "\n\n\nLLM:\n" + response \
            + "\n\n\nThinking:\n" \
            + thinking

        return predicted_reward, confidence, clean_reasoning, full_transcript, api_time