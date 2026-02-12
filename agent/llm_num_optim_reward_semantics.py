import os
import random
import argparse
import yaml
import numpy as np
from jinja2 import Environment, FileSystemLoader
from agent.policy.llm_brain_linear_policy import LLMBrain
import re

class LLMNumOptimRewardSemanticAgent:
    def __init__(
        self,
        logdir,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        env_desc_file=None,
    ):
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.replay_buffer = []
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.env_desc_file = env_desc_file

def load_dataset(self, dataset_file):
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('|')
            param_str = parts[0].strip()
            reward_str = parts[1].strip()
            
            param_values = [float(x) for x in param_str.split(',')]
            reward_value = float(reward_str)
            
            self.replay_buffer.add(np.array(param_values).reshape(-1), reward_value)
    print(f"Loaded dataset from {dataset_file} with {len(self.replay_buffer.buffer)} entries.")
    
def split_dataset(self, train_ratio=0.7):
    total_size = len(self.replay_buffer.buffer)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_size = int(total_size * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_buffer = EpisodeRewardBufferNoBias(max_size=train_size)
    test_buffer = EpisodeRewardBufferNoBias(max_size=total_size - train_size)
    
    for idx in train_indices:
        train_buffer.add(self.replay_buffer.buffer[idx], self.replay_buffer.rewards[idx])
    
    for idx in test_indices:
        test_buffer.add(self.replay_buffer.buffer[idx], self.replay_buffer.rewards[idx])
    
    return train_buffer, test_buffer