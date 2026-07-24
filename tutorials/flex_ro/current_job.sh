#!/bin/bash
#SBATCH --job-name=PT
#SBATCH --account=nawianalysis
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --partition=standard
#SBATCH -L gurobi@slurmdb:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=PT_DR_with_delayed_shutdowns.%j.out  # %j will be replaced with the job ID

module load gurobi
module load anaconda3
conda activate watertap-pricetaker
python pricetaker_fast_5_28.py
