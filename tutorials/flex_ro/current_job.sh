#!/bin/bash
#SBATCH --job-name=PT_with_DR
#SBATCH --account=nawianalysis
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH -L gurobi:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=PT.%j.out  # %j will be replaced with the job ID


module load gurobi
module load anaconda3
conda activate watertap-pricetaker
python ro_const_RR_wrd.py
