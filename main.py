import yaml
import argparse
from runner import llm_num_optim_runner
# NEW IMPORT
from runner import reward_prediction_runner 

def main():
    parser = argparse.ArgumentParser(description='LLM Agent Runner')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    task = config.get('task')

    if task == 'cont_space_llm_num_optim':
        runner = llm_num_optim_runner.LLMNumOptimRunner(config)
        runner.run()
    elif task == 'reward_prediction':
        # NEW TASK HANDLER
        runner = reward_prediction_runner.RewardPredictionRunner(config)
        runner.run()
    else:
        print(f"Unknown task: {task}")

if __name__ == "__main__":
    main()