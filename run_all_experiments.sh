#!/bin/bash

# Define the list of environments (folder names)
envs=(
    "cliffwalking"
    "hopper"
    "inverted_double_pendulum"
    "invertedpendulum"
    "mountaincardiscrete"
    "mountaincarcontinuous"
    "nav"
    "pong"
    "reacher"
    "swimmer"
    "walker2d"
)

# Loop through each environment
for env in "${envs[@]}"; do
    echo "========================================================"
    echo "STARTING REWARD PREDICTION FOR: $env"
    echo "========================================================"
    
    # Define config path (Assumes standard naming convention)
    CONFIG_PATH="configs/$env/reward_prediction.yaml"
    
    if [ -f "$CONFIG_PATH" ]; then
        # Run the experiment
        python main.py --config "$CONFIG_PATH"
        
        echo "Finished experiment for $env"
    else
        echo "ERROR: Config file not found at $CONFIG_PATH"
        echo "Skipping $env..."
    fi
    
    echo ""
    echo "--------------------------------------------------------"
    echo ""
done

echo "All experiments completed."