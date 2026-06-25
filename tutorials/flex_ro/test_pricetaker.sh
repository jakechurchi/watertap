#!/bin/bash
#SBATCH --job-name=PT_tutorial_test
#SBATCH --account=nawianalysis
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH -L gurobi:1
#SBATCH --mail-user=jake.churchill@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --output=PT_tutorial_test.%j.out  # %j will be replaced with the job ID

module load gurobi
module load anaconda3
conda activate watertap-pricetaker

# Run from repo root so pytest.ini testpaths discovery works
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# Run the tutorial pricetaker test
# Disable cache provider to avoid permission denied errors on shared filesystems
python -m pytest "tutorials/flex_ro/test_pricetaker_WRD.py" --no-cov -p no:cacheprovider