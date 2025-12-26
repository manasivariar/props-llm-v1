from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.llm_num_optim_reward import LLMNumOptimRewardAgent
from jinja2 import Environment, FileSystemLoader
import os
import traceback
import numpy as np

def run_reverse_rl_loop(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    dim_actions,
    dim_states,
    max_traj_count, 
    template_dir,
    llm_si_template_name,
    llm_model_name,
    num_evaluation_episodes,
    warmup_episodes,
    warmup_dir,
    optimum,
    bias=None,
    env_desc_file=None,
    **kwargs 
):
    assert task == "reverse_rl"

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)

    if "FrozenLake" in gym_env_name or "CliffWalking" in gym_env_name or "Nim" in gym_env_name:
         world = DiscreteStateGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length=kwargs.get('max_traj_length', 1000),
            env_kwargs=kwargs.get('env_kwargs', None)
        )
    else:
        world = ContinualSpaceGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length=kwargs.get('max_traj_length', 1000),
        )

    agent = LLMNumOptimRewardAgent(
        logdir=logdir,
        dim_action=dim_actions,
        dim_state=dim_states,
        max_traj_count=max_traj_count,
        llm_si_template=llm_si_template,
        llm_model_name=llm_model_name,
        warmup_episodes=warmup_episodes,
        num_evaluation_episodes=num_evaluation_episodes,
        bias=bias,
        optimum=optimum,
        env_desc_file=env_desc_file
    )

    print('Reverse RL Init Done')
    
    if not warmup_dir:
        warmup_dir = f"{logdir}/warmup"
        os.makedirs(warmup_dir, exist_ok=True, mode=0o777)
        agent.random_warmup(world, warmup_dir, warmup_episodes)
    else:
        agent.replay_buffer.load(warmup_dir)

    # 4. Logging Setup - CSV Header
    overall_log_file = open(f"{logdir}/overall_log.txt", "w")
    overall_log_file.write("Iteration, CPU Time, API Time, Total Episodes, Total Steps, True Reward, Predicted Reward, Confidence Score\n")
    overall_log_file.flush()
    
    # 5. Main Loop
    for episode in range(num_episodes):
        print(f"--- Iteration: {episode} ---")
        
        current_episode_dir = f"{logdir}/episode_{episode}"
        os.makedirs(current_episode_dir, exist_ok=True, mode=0o777)
        
        success = False
        
        for trial_idx in range(5):
            try:
                cpu_time, api_time, total_episodes, total_steps, true_reward, pred_reward, confidence = agent.run_prediction_step(world, episode, current_episode_dir)
                overall_log_file.write(f"{episode + 1}, {cpu_time:.4f}, {api_time:.4f}, {total_episodes}, {total_steps}, {true_reward:.4f}, {pred_reward:.4f}, {confidence:.2f}\n")
                overall_log_file.flush()
                print(f"{trial_idx + 1}th trial attempt succeeded in training")
                success = True
                break
            except Exception as e:
                print(f"{trial_idx + 1}th trial attempt failed: {e}")
                traceback.print_exc()
                continue
        if not success:
            print(f"Iteration {episode} failed to train after 5 attempts")
            break
    overall_log_file.close()