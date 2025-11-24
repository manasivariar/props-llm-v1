#!/usr/bin/env python3
"""
Submit multiple SLURM jobs from a manifest YAML file.
Each job in the manifest will spawn an independent SLURM array job.
"""

import yaml
import argparse
import subprocess
import os
from pathlib import Path
import sys

def submit_manifest(manifest_file, job_name=None, dry_run=False, verbose=False):
    """
    Parse manifest file and submit SLURM jobs for each config
    
    Args:
        manifest_file: Path to manifest YAML file
        job_name: Optional filter to run only a specific job by name
        dry_run: Preview without submitting
    """
    if not os.path.exists(manifest_file):
        print(f"❌ Manifest file not found: {manifest_file}")
        sys.exit(1)
    
    with open(manifest_file, 'r') as f:
        manifest = yaml.safe_load(f)
    
    jobs = manifest.get('jobs', [])
    
    if not jobs:
        print("❌ No jobs found in manifest file")
        return
    
    # Filter jobs if job_name is specified
    if job_name:
        jobs = [j for j in jobs if j['name'] == job_name]
        if not jobs:
            print(f"❌ Job '{job_name}' not found in manifest")
            return
    
    print(f"📋 Found {len(jobs)} job(s) in manifest\n")
    
    submitted_count = 0
    failed_count = 0
    
    for job in jobs:
        job_name = job['name']
        config = job['config']
        logdir = job['logdir']
        trials = job.get('trials', 5)
        time_limit = job.get('time', '24:00:00')
        cpus = job.get('cpus', 4)
        gpu = job.get('gpu', 'a100:1')
        
        # Validate config exists
        if not os.path.exists(config):
            print(f"❌ Config not found: {config}")
            failed_count += 1
            continue
        
        # Create output directories
        os.makedirs(f"{logdir}/slurm", exist_ok=True, mode=0o755)
        for i in range(1, trials + 1):
            os.makedirs(f"{logdir}/trial_{i}", exist_ok=True, mode=0o755)
        
        print(f"📋 Job: {job_name}")
        print(f"   Config: {config}")
        print(f"   Logdir: {logdir}")
        print(f"   Trials: {trials}")
        print(f"   Time: {time_limit}")
        print(f"   CPUs: {cpus}, GPU: {gpu}")
        
        # Create sbatch script for this job
        sbatch_script = create_sbatch_script(
            job_name=job_name,
            config_file=config,
            base_logdir=logdir,
            trials=trials,
            time_limit=time_limit,
            cpus=cpus,
            gpu=gpu
        )
        
        if dry_run:
            print(f"   [DRY RUN] Would submit with sbatch script")
            if verbose:
                print("\n" + "="*70)
                print(sbatch_script)
                print("="*70 + "\n")
        else:
            # Write sbatch script to temp file and submit
            temp_sbatch = f"/tmp/{job_name}_{os.getpid()}_sbatch.sh"
            with open(temp_sbatch, 'w') as f:
                f.write(sbatch_script)
            
            os.chmod(temp_sbatch, 0o755)
            
            try:
                result = subprocess.run(['sbatch', temp_sbatch], 
                                      capture_output=True, 
                                      text=True,
                                      check=False)
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    print(f"   ✅ Submitted successfully: Job ID {job_id}")
                    submitted_count += 1
                else:
                    print(f"   ❌ Submission failed: {result.stderr}")
                    failed_count += 1
            except Exception as e:
                print(f"   ❌ Error submitting job: {e}")
                failed_count += 1
            finally:
                # Clean up temp file
                if os.path.exists(temp_sbatch):
                    os.remove(temp_sbatch)
        
        print()
    
    # Summary
    if not dry_run:
        print("="*70)
        print(f"Summary: {submitted_count} submitted, {failed_count} failed")
        print("="*70)

def create_sbatch_script(job_name, config_file, base_logdir, trials, time_limit, cpus, gpu):
    """
    Generate SLURM batch script for a specific job
    """
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=1-{trials}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpu}
#SBATCH --constraint=a100_80
#SBATCH --partition=general
#SBATCH --time={time_limit}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --output={base_logdir}/slurm/slurm_output_IP_best_5_recent_65/trial_%j_%a.log
#SBATCH --error={base_logdir}/slurm/slurm_output_IP_best_5_recent_65/trial_%j_%a.err

# --- Job Configuration ---
CONFIG_FILE="{config_file}"
BASE_LOG_DIR="{base_logdir}"

# --- Create a unique directory for this specific trial's output ---
TRIAL_LOG_DIR="${{BASE_LOG_DIR}}/trial_${{SLURM_ARRAY_TASK_ID}}"
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

# --- Run the Python Script ---
echo "--- Starting SLURM Trial ${{SLURM_ARRAY_TASK_ID}} ---"
echo "--- Config: ${{CONFIG_FILE}} ---"
echo "--- Logging to: ${{TRIAL_LOG_DIR}} ---"

/home/mrajanva/.conda/envs/props/bin/python3 main.py --config "$CONFIG_FILE" --logdir "$TRIAL_LOG_DIR"

# Check exit status
EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "--- ERROR: Trial ${{SLURM_ARRAY_TASK_ID}} failed with exit status ${{EXIT_STATUS}} ---"
else
    echo "--- Success: Trial ${{SLURM_ARRAY_TASK_ID}} completed ---"
fi

# --- CLEAN UP: Stop the Ollama server ---
echo "--- Shutting down Ollama server (PID: $OLLAMA_PID) ---"
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

exit $EXIT_STATUS
"""
    return script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit multiple SLURM jobs from a manifest YAML file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview jobs without submitting
  python submit_manifest.py manifests/manifest.yaml --dry-run
  
  # Preview with full script output
  python submit_manifest.py manifests/manifest.yaml --dry-run --verbose
  
  # Submit all jobs
  python submit_manifest.py manifests/manifest.yaml
        """
    )
    parser.add_argument(
        "manifest",
        type=str,
        help="Path to manifest YAML file"
    )
    parser.add_argument(
        "--job",
        type=str,
        default=None,
        help="Optional: run only a specific job by name"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview jobs without submitting to SLURM"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full SBATCH scripts (use with --dry-run)"
    )
    
    args = parser.parse_args()
    submit_manifest(args.manifest, job_name=args.job, dry_run=args.dry_run, verbose=args.verbose)
