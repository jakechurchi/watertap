#!/bin/bash
#SBATCH --account=nawianalysis
#SBATCH --time=00:10:00
#SBATCH --job-name=pricetaker_test
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=job_output_filename.%j.out  # %j will be replaced with the job ID

module load gurobi
module load anaconda3
conda activate watertap-pricetaker
python ro_const_RR_wrds.py