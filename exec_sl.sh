#!/bin/bash
#SBATCH --job-name=@toolbox
#SBATCH --partition=public
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm.%J.out
#SBATCH --mail-user=avinajera2011@gmail.com
#SBATCH --mail-type=ALL
source ./python_Anaconda3/bin/activate
cd ./py_project1/paper_random_forest
python @toolbox.py