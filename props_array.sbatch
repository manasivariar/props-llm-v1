#!/bin/bash
#SBATCH --job-name=props_array
#SBATCH --array=1-30                # Total number of sub-jobs (will be overridden by command line)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # Default cores per sub-job (can be overridden with -c)
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --partition=general
#SBATCH --time=24:00:00             # Wall time for each sub-job
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --output=array_logs/slurm/task_%j_%a.log
#SBATCH --error=array_logs/slurm/task_%j_%a.err

# --- 1. Setup: Read Manifest ---
manifest="${1:?ERROR -- must pass a manifest file as first argument}"
taskid=$SLURM_ARRAY_TASK_ID

if [ ! -f "$manifest" ]; then
    echo "ERROR: Manifest file not found: $manifest"
    exit 1
fi

# Read the task ID line from manifest
# Format: <random_seed> <scaler> <model>
line=$(sed -n "${taskid}p" "$manifest")

if [ -z "$line" ]; then
    echo "ERROR: Task ID $taskid not found in manifest (manifest has $(wc -l < "$manifest") lines)"
    exit 1
fi

# Extract parameters from the line
random_seed=$(echo "$line" | cut -d ' ' -f 1)
config=$(echo "$line" | cut -d ' ' -f 2)
logdir=$(echo "$line" | cut -d ' ' -f 3)
summary=$(echo "$line" | cut -d ' ' -f 4)

# --- Create output directory structure (like Python script does) ---
mkdir -p "${logdir}/slurm"
TRIAL_LOG_DIR="${logdir}/trial_${random_seed}"
mkdir -p "$TRIAL_LOG_DIR"

# --- Load Modules ---
module load mamba/latest
module load ollama/0.11.8

# --- START THE OLLAMA SERVER IN THE BACKGROUND ---
echo "--- Starting Ollama server on $(hostname) ---"
export OLLAMA_HOST=0.0.0.0 
ollama serve &
OLLAMA_PID=$! 
sleep 15
echo "--- Checking Ollama status ---"
ollama list

# --- Activate Mamba/Conda Environment ---
source activate props

echo "========================================================================="
echo "SLURM Job Array Task: $taskid / $SLURM_ARRAY_JOB_ID"
echo "Parameters from manifest:"
echo "  Random Seed: $random_seed"
echo "  Config: $config"
echo "  Log Directory: $logdir"
echo "  Summary Flag: $summary"
echo "========================================================================="


/home/mrajanva/.conda/envs/props/bin/python3 main.py \
    --config "$config" \
    --logdir "$TRIAL_LOG_DIR" \
    --summary "$summary"

# Check exit status
EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "--- ERROR: Task $taskid failed with exit status $EXIT_STATUS ---"
else
    echo "--- Success: Task $taskid completed ---"
fi

# --- CLEAN UP: Stop the Ollama server ---
echo "--- Shutting down Ollama server (PID: $OLLAMA_PID) ---"
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

exit $EXIT_STATUS
