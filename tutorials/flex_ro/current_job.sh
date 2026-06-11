#!/bin/bash
#SBATCH --job-name=PT_week
#SBATCH --account=nawianalysis
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --partition=debug
#SBATCH -L gurobi:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=water_targ_sweep_no_flex.%j.out  # %j will be replaced with the job ID

module load gurobi
module load anaconda3
conda activate watertap-pricetaker
python sweep_water_target_week.py
