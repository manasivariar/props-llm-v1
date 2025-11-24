#!/bin/bash
MANIFEST_FILE="props_manifest"

# Clear the file to start fresh
> $MANIFEST_FILE

echo "--- Generating manifest file ($MANIFEST_FILE) ---"

# --- Define Your Experiments Here ---

# Correlated arrays for experiment configs and log prefixes
declare -a configs=(
    "configs/inverted_double_pendulum/inverteddoublependulum_propsp.yaml"
    "configs/invertedpendulum/invertedpendulum_propsp.yaml"
    "configs/mountaincar/mountaincar_propsp.yaml"
)

declare -a log_prefixes=(
    "idp-logs"
    "ip-logs"
    "mcd-logs"
)

# Define the run modes and trial count
declare -a run_modes=(
    "baseline"
    "summary"
)
NUM_TRIALS=5

# --- Generate Manifest Lines ---
for i in ${!configs[@]}; do
    config_file=${configs[$i]}
    log_prefix=${log_prefixes[$i]}

    for mode in "${run_modes[@]}"; do
        # Set summary_flag to "true" if mode is "summary", "false" otherwise
        summary_flag="false"
        if [ "$mode" == "summary" ]; then
            summary_flag="true"
        fi
        
        # e.g., "idp-logs/baseline" or "idp-logs/summary"
        base_logdir="${log_prefix}/${mode}"
        
        for trial_id in $(seq 1 $NUM_TRIALS); do
            # Format: [Trial ID] [Config Path] [Base Logdir] [Summary Flag]
            echo "$trial_id $config_file $base_logdir $summary_flag" >> $MANIFEST_FILE
        done
    done
done

TOTAL_JOBS=$(wc -l < $MANIFEST_FILE)
echo "--- Done. Created $MANIFEST_FILE with $TOTAL_JOBS total jobs. ---"
echo "Please update your sbatch script's --array directive to --array=1-$TOTAL_JOBS"