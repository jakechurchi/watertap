#!/bin/bash
#SBATCH --job-name=PT_week
#SBATCH --account=nawianalysis
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH -L gurobi:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=PT_simple_DR_week.%j.out  # %j will be replaced with the job ID

module load gurobi
module load anaconda3
conda activate watertap-pricetaker
python wrd_pricetaker_short.py
