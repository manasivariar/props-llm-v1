#!/bin/bash
# Submit the full job array
# Usage: ./submit_array.sh <manifest_file>

manifest="${1:?ERROR -- must pass a manifest file}"

if [ ! -f "$manifest" ]; then
    echo "ERROR: Manifest file not found: $manifest"
    exit 1
fi

# Count total lines in manifest
total_tasks=$(wc -l < "$manifest")

if [ "$total_tasks" -eq 0 ]; then
    echo "ERROR: Manifest file is empty"
    exit 1
fi

echo "========================================================================="
echo "Submitting full job array"
echo "Manifest: $manifest"
echo "Total tasks: $total_tasks"
echo "========================================================================="
echo ""
echo "Command: sbatch -a 1-${total_tasks} props_array.sbatch $manifest"
echo ""

# Create output directory
mkdir -p array_logs/slurm

# Submit the job array
sbatch -a "1-${total_tasks}" props_array.sbatch "$manifest"

echo ""
echo "========================================================================="
echo "Job array submitted!"
echo "Monitor with: squeue --me"
echo "View details with: scontrol show jobid <job_id>"
echo "========================================================================="
