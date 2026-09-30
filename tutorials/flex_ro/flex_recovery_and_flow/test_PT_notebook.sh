#!/bin/bash
#SBATCH --job-name=PT_tutorial_test
#SBATCH --account=nawianalysis
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --partition=debug
#SBATCH -L gurobi@slurmdb:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=PT_tutorial_test.%j.out  # %j will be replaced with the job ID

module load gurobi
module load anaconda3
conda activate watertap-pricetaker

jupyter nbconvert --to notebook --execute --inplace flex_recovery_and_flow.ipynb
