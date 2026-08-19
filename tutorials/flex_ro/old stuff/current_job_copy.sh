#!/bin/bash
#SBATCH --job-name=PT_copy
#SBATCH --account=nawianalysis
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH -L gurobi@slurmdb:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=PT_one_hour_start_up_delay.%j.out  # %j will be replaced with the job ID

module load gurobi
module load anaconda3
conda activate watertap-pricetaker
python PT_wrd_week_temp_copy.py
