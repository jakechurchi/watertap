#!/bin/bash
#SBATCH --job-name=PT_copy
#SBATCH --account=nawianalysis
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --partition=standard
#SBATCH -L gurobi@slurmdb:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=PT_faster_no_brine_cost.%j.out  # %j will be replaced with the job ID

module load gurobi
module load anaconda3
conda activate watertap-pricetaker
python PT_wrd_week_temp_copy.py
