#!/bin/bash
#SBATCH --job-name=hydrometeor_profile
#SBATCH --output=/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/python_scripts/hydrometeor_profile_%j.out
#SBATCH --error=/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/python_scripts/hydrometeor_profile_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Activate the conda environment that has faxarray
source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
conda activate epygram

# Run the python script
python /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/python_scripts/analyze_hydrometeors.py

echo "Job completed."
