import yaml
import argparse
import os
from runner import reward_prediction_runner 

def main():
    parser = argparse.ArgumentParser(description='LLM Agent Runner')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    task = config.get('task')

    if task == 'reward_prediction':
        # NEW TASK HANDLER
        runner = reward_prediction_runner.StickyRewardPredictionRunner(config)
        runner.run()
    else:
        print(f"Unknown task: {task}")

if __name__ == "__main__":
    main()