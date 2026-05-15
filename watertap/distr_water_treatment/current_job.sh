#!/bin/bash
#SBATCH --job-name=distrib_treat_opt
#SBATCH --account=nawianalysis
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH -L gurobi:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=techs_sweep.%j.out  # %j will be replaced with the job ID


module load gurobi
module load anaconda3
conda activate watertap-pricetaker
python weekly_optimization.py
