#!/bin/bash
#SBATCH --job-name=PT_copy
#SBATCH --account=nawianalysis
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --partition=debug
#SBATCH -L gurobi@slurmdb:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=PT_no_startup_brine_cost_RTP.%j.out  # %j will be replaced with the job ID

module load gurobi
module load anaconda3
conda activate watertap-pricetaker
python PT_wrd_week_temp_copy.py
