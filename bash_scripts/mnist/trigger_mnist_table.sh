#!/bin/bash

# -----------------------------------------------------------------------------
#                               TO BE CUSTOMIZED
# -----------------------------------------------------------------------------
# Directory containing the source code
L0_SRC_PATH="$HOME/github/constrained_l0"

# Getting helper functions - Do not delete
source $L0_SRC_PATH/bash_scripts/bash_helpers.sh

# Bash script with call to Python script
main_bash_script="${L0_SRC_PATH}/bash_scripts/run_basic_exp.sh"

# SLURM options
slurm_log_dir="$HOME/slurm_logs/const/"
notify_email="" # Leave empty ("") for no email
partition="long"

# Number of GPUs
NUM_GPUS=2

# WandB -- You will likely want to keep these untouched!
use_wandb=True
run_group="neurips_table"
wandb_dir="${SCRATCH}/wandb_logs/"

# Directory for storing training checkpoints (auto-created if it doesn't exist)
checkpoint_dir="${SCRATCH}/l0_checkpoints/mnist"


# -----------------------------------------------------------------------------

# The parameter of this function is the python arguments
submit_sbatch () {
    sbatch --job-name=constl0-mnist-slurm-%j.out \
        --time=1:30:00 \
        --cpus-per-task 6 \
        --mem=32G \
        --gres=gpu:$NUM_GPUS \
        --nodes=1 \
        --ntasks-per-node=$NUM_GPUS \
        --partition=$partition \
        --exclude=rtx[1-7] \
        --output=$slurm_log_dir/mnist-slurm-%j.out \
        --mail-type=ALL --mail-user=$notify_email \
        $main_bash_script $1
}

export L0_SRC_PATH
export NUM_GPUS

# Get basic configuration from YAML file
# MNIST TABLE DEFAULTS === MNIST DEFAULTS (no change!)
yaml_arg="-yaml ${L0_SRC_PATH}/configs/defaults/mnist_defaults.yml"

checkpoint_arg="--checkpoint_dir $checkpoint_dir"

# Set up WandB flags
wandb_arg=$( create_wandb_arg "${use_wandb}" "${run_group}" "${wandb_dir}" )


declare -a seeds=(1 2 3 4 5)

for seed in ${seeds[@]}; do

    seed_arg="--seed $seed"
    
    # Collect common args for core_exp.py script
    python_args="${yaml_arg} ${seed_arg} ${checkpoint_arg} ${wandb_arg}"

    # ---------------------------------------------------------------------
    #                               MLPs       
    # ---------------------------------------------------------------------
    submit_sbatch "${python_args} --model_type MLP -tdst 0.33"
    
    # The line below is meant to be 0.30 and not 0.33
    _reg_list=$( mnist_regularization_list "layer" "0.30" "MLP" )
    submit_sbatch "${python_args} --model_type MLP -tdst ${_reg_list}"

    # ---------------------------------------------------------------------
    #                               LeNets       
    # ---------------------------------------------------------------------
    # This was chosen to find model matching Louizos final model for the
    # setting: [MNIST; LeNet; model-level]
    submit_sbatch "${python_args} --model_type LeNet -tdst 0.1"
    
    submit_sbatch "${python_args} --model_type LeNet -tdst 0.5 0.3 0.7 0.1"

done
