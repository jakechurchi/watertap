#!/bin/bash
#SBATCH --job-name=PT_sweep
#SBATCH --account=nawianalysis
#SBATCH --time=02:30:00
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH -L gurobi:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=water_sweep_num_shutdowns.%j.out  # %j will be replaced with the job ID

module load gurobi
module load anaconda3
conda activate watertap-pricetaker
python sweep_water_target_week.py
