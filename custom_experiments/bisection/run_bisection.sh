#!/bin/bash
#SBATCH --job-name=bisection
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --output=bisection.out

# -----------------------------------------------------------------------------
#                               TO BE CUSTOMIZED
# -----------------------------------------------------------------------------
# Directory containing the source code
L0_SRC_PATH="$HOME/github/constrained_l0"

# Getting helper functions - Do not delete
source $L0_SRC_PATH/bash_scripts/bash_helpers.sh

load_modules_and_env $USER

cd $L0_SRC_PATH

python $L0_SRC_PATH/custom_experiments/bisection/bisection.py