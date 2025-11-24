#!/bin/bash
# Benchmark script to test different core counts on the first array task
# Usage: ./benchmark_array.sh <manifest_file>

manifest="${1:?ERROR -- must pass a manifest file}"

if [ ! -f "$manifest" ]; then
    echo "ERROR: Manifest file not found: $manifest"
    exit 1
fi

echo "========================================================================="
echo "Benchmarking job array with manifest: $manifest"
echo "Testing task 1 with different core counts"
echo "========================================================================="

# Create output directory
mkdir -p array_logs/slurm

# Test with different core counts
for cores in 1 4 8; do
    echo ""
    echo "Testing with $cores cores..."
    sbatch -a 1 -c $cores props_array.sbatch "$manifest"
    echo "Submitted benchmark job with $cores cores"
done

echo ""
echo "========================================================================="
echo "Benchmarking submitted!"
echo "Monitor with: squeue --me"
echo "After tests complete, check array_logs/slurm/ for timing information"
echo "========================================================================="
