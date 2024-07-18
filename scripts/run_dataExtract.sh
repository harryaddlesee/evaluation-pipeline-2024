#!/bin/bash
#SBATCH --job-name=DATA-EX  
#SBATCH --output=output_%j.txt     
#SBATCH --error=error_%j.txt              
#SBATCH --time=24:00:00             
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                      
#SBATCH --cpus-per-task=2                
#SBATCH --mem=16GB                         
#SBATCH --partition=gpu                   
#SBATCH --gres=gpu:1

# Run your Python script
export BABYLM_ROOT_DIR=/users/ha2098/sharedscratch/venv/projects/evaluation-pipeline-2024
python scripts/dataExtraction.py --output_dir results
