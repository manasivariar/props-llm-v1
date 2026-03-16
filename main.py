import yaml
import argparse
import os

from runner import llm_num_optim_runner
from runner import llm_num_optim_semantics_runner
from runner.reward_prediction_runner import SymbolicRewardRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--logdir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.logdir is not None:
        config["logdir"] = args.logdir
    elif "logdir" not in config:
        config["logdir"] = f"logs/symbolic_{config.get('gym_env_name', 'default')}"
        os.makedirs(config["logdir"], exist_ok=True)

    task = config.get("task", "")
    
    if task == "symbolic_reward":
        runner = SymbolicRewardRunner(config)
        runner.run()
        
    elif task in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]:
        llm_num_optim_runner.run_training_loop(**config)
    elif task in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]:
        llm_num_optim_semantics_runner.run_training_loop(**config)
    else:
        raise ValueError(f"Task '{task}' not recognized.")

if __name__ == "__main__":
    main()