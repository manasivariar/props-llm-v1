from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.linear_policy import LinearPolicy
from agent.policy.llm_brain_reward import LLMBrainReward
from agent.policy.llm_brain_reward import PromptOutput
from agent.policy.replay_buffer import PredictionBuffer
from world.base_world import BaseWorld
from pydantic import ValidationError
import numpy as np
import time
import os
import re
import json
import random

class LLMNumOptimRewardAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        llm_si_template,
        llm_model_name,
        warmup_episodes,
        num_evaluation_episodes,
        bias,
        optimum,
        env_desc_file,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0
        self.total_episodes = 0
        self.total_steps = 0 
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.bias = bias
        self.env_desc_file = env_desc_file
        self.num_evaluation_episodes = num_evaluation_episodes
        self.logdir = logdir
        self.optimum = optimum

        if not self.bias:
            param_count = dim_action * dim_state
            self.policy = LinearPolicyNoBias(dim_actions=dim_action, dim_states=dim_state)
        else:
            param_count = dim_action * dim_state + dim_action
            self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
        
        self.rank = param_count
        self.replay_buffer = PredictionBuffer(max_size=max_traj_count)
        
        with open('dataset/idp-params-sampled.txt', 'r') as f:
            self.sampled_params = [np.array(i.split(' | ')[0].split(",")).astype(float) for i in f.readlines()]
            
        random.shuffle(self.sampled_params)
        self.warmup_samples = random.sample(self.sampled_params, warmup_episodes)
        
        self.llm_brain = LLMBrainReward(
            llm_si_template=llm_si_template,
            llm_model_name=llm_model_name
        )

    def rollout_episode(self, world: BaseWorld, logging_file=None):
        state = world.reset()
        if logging_file:
            logging_file.write(f"state | action | reward\n")

        done = False
        truncated = False
        while not (done or truncated):
            state_in = np.expand_dims(state, axis=0)
            action = self.policy.get_action(state_in.T)
            action = np.reshape(action, (1, self.dim_action))
            
            if world.discretize:
                action_env = np.argmax(action)
                action_env = np.array([action_env])
            else:
                action_env = action
            next_state, reward, done, truncated = world.step(action_env)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")

            state = next_state
            self.total_steps += 1 # Track steps
        
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        return world.get_accu_reward()

    def evaluate_current_policy(self, world, logdir=None):
        results = []
        log_file = None
        if logdir:
            logging_filename = f"{logdir}/training_rollout.txt"
            log_file = open(logging_filename, "w")

        for _ in range(self.num_evaluation_episodes):
            if log_file:
                params = self.policy.get_parameters().reshape(-1)
                params_str = ", ".join([str(x) for x in params])
                log_file.write(f"{params_str}\nparameter ends\n\n")

            result = self.rollout_episode(world, logging_file=log_file)
            
            if log_file:
                log_file.write("\n")
            results.append(result)
            
        print(f"Evaluation over {self.num_evaluation_episodes} episodes: total ep:{self.total_episodes} ")
        
        if log_file:
            log_file.close()
        return np.mean(results)

    def random_warmup(self, world: BaseWorld, logdir, warmup_episodes):
        print(f"--- Starting Warmup for {warmup_episodes} trials ---")
        # Ensure logdir exists (passed from runner)
        
        for i in range(warmup_episodes):
            # self.policy.initialize_policy()
            
            self.policy.update_policy(self.warmup_samples[i])
            params = self.policy.get_parameters().reshape(-1)
            
            logging_filename = f"{logdir}/warmup_rollout_{i}.txt"
            rewards = []
            
            with open(logging_filename, "w") as f:
                for _ in range(self.num_evaluation_episodes):
                    params_str = ", ".join([str(x) for x in params])
                    f.write(f"{params_str}\nparameter ends\n\n")
                    reward = self.rollout_episode(world, logging_file=f)
                    rewards.append(reward)
                    f.write("\n")
            
            avg_reward = np.mean(rewards)
            self.replay_buffer.add(params, avg_reward, None)
            print(f"Warmup {i+1}/{warmup_episodes}: Avg Reward {avg_reward:.2f}")
        print("--- Warmup Complete ---")

    def run_prediction_step(self, world: BaseWorld, iteration, episode_logdir):
        
        def parse_prediction(input_text):
            print(f"LLM Response (Raw): {input_text}")
            try:
                validated_response = PromptOutput.model_validate_json(input_text)
                predicted_reward = validated_response.predicted_reward
                confidence = validated_response.confidence_score
                reasoning = validated_response.reasoning
            except ValidationError as ve:   
                print(f"Validation Error: {ve}")
                # Fallback parsing using regex
                pred_match = re.search(r'"predicted_reward"\s*:\s*([0-9.+-eE]+)', input_text)
                conf_match = re.search(r'"confidence_score"\s*:\s*([0-9.+-eE]+)', input_text)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', input_text, re.DOTALL)

                if pred_match and conf_match and reasoning_match:
                    predicted_reward = float(pred_match.group(1))
                    confidence = float(conf_match.group(1))
                    reasoning = reasoning_match.group(1).strip()
                else:
                    raise ValueError("Failed to parse LLM response.")
                
            
            print(f"Parsed Prediction - Reward: {predicted_reward}, Confidence: {confidence}")
            return predicted_reward, confidence, reasoning
        
        def str_reverse_rl_history(params, n):
            """
            Formats the history of parameters and rewards for the Reverse RL prompt.
            matches format: params[0]: val; ... true_reward: X predicted_reward: Y
            """
            text = ""
            # Iterate through the PredictionBuffer
            # buffer stores tuples: (params, true_reward, pred_reward)
            for params, true_reward, pred_reward, confidence in self.replay_buffer.getTopKItems(params, 20):
                
                # Flatten parameters to ensure easy indexing
                p_flat = np.array(params).reshape(-1)
                
                # Build parameter string: "params[0]: 1.2; params[1]: -0.5; ..."
                line = ""
                for i in range(n):
                    line += f"params[{i}]: {p_flat[i]:.5g}; "
                
                # Append Rewards
                # Handle cases (like warmup) where prediction might be None/0.0 if not logged
                if pred_reward is None and confidence is None:
                    line += f"true_reward: {true_reward:.2f}; predicted_reward(params): N/A; confidence_score: N/A"
                else:
                    line += f"true_reward: {true_reward:.2f}; predicted_reward(params): {pred_reward:.2f}; confidence_score: {confidence:.2f}"
                
                text += line + "\n"
            
            return text

        # 1. Generate Random Parameters
        # self.policy.initialize_policy()
        # target_params = self.policy.get_parameters().reshape(-1)
        target_params = self.sampled_params[iteration % len(self.sampled_params)]

        # 2. Predict (Pass parser to brain)
        # Brain returns 4 values: pred, clean_reasoning, full_transcript, api_time
        pred_reward, confidence, clean_reasoning, full_transcript, api_time = self.llm_brain.llm_predict_reward(
            history_string=str_reverse_rl_history(target_params, self.rank),
            target_params=target_params,
            optimum=self.optimum,
            env_desc_file=self.env_desc_file,
            rank=self.rank,
            parse_prediction_func=parse_prediction,
            step_number=iteration
            
        )
        self.api_call_time += api_time

        # --- LOGGING: Full Transcript (Reasoning file) ---
        with open(f"{episode_logdir}/reward_reasoning.txt", "w") as f:
            f.write(full_transcript)

        # --- LOGGING: Parameters ---
        with open(f"{episode_logdir}/parameters.txt", "w") as f:
            f.write(str(self.policy))

        # 3. Execute Environment
        self.policy.update_policy(target_params)
        true_reward = self.evaluate_current_policy(world, logdir=episode_logdir)

        # 4. Update Buffer
        self.replay_buffer.add(target_params, true_reward, pred_reward, confidence)

        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = true_reward
        _pred_reward = pred_reward
        _confidence = confidence

        # RETURN 8 VALUES as expected by Runner
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward, _pred_reward, _confidence