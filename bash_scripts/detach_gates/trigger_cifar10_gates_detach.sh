#!/bin/bash

# This script is used to run CIFAR10 experiments to test the effect of 
# "detaching" the gates when computing the gradient of the weight decay penalty.
# Default values: "configs/defaults/cifar10_gates_detach_defaults.yml"

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

# Number of GPUs.
NUM_GPUS=2

# WandB -- You will likely want to keep these untouched!
use_wandb=True
run_group="gates_detach"
wandb_dir="${SCRATCH}/wandb_logs/"

# Directory for storing training checkpoints (auto-created if it doesn't exist)
checkpoint_dir="${SCRATCH}/l0_checkpoints/cifar10_gates_detach/"

# Config lists
declare -a seeds=(1)
declare -a target_densities=(0.35 0.7)
declare -a reg_types=("layer")
declare -a l2_detach_array=(True False)
declare -a gate_lr_array=(0.1 1.0 6.0)

# -----------------------------------------------------------------------------

# The parameter of this function is the python arguments
submit_sbatch () {
    sbatch --job-name=constl0-cifar10_gates_detach-slurm-%j.out \
        --time=10:00:00 \
        --cpus-per-task 6 \
        --mem=32G \
        --gres=gpu:$NUM_GPUS \
        --nodes=1 \
        --ntasks-per-node=$NUM_GPUS \
        --partition=$partition \
        --exclude=rtx[1-7] \
        --output=$slurm_log_dir/cifar10_gates_detach-slurm-%j.out \
        --mail-type=ALL --mail-user=$notify_email \
        $main_bash_script $1
}

export L0_SRC_PATH
export NUM_GPUS

# Get basic configuration from YAML file
yaml_arg="-yaml ${L0_SRC_PATH}/configs/defaults/cifar10_gates_detach_defaults.yml"

checkpoint_arg="--checkpoint_dir $checkpoint_dir"

# Set up WandB flags
wandb_arg=$( create_wandb_arg "${use_wandb}" "${run_group}" "${wandb_dir}" )


for detach_gates in ${l2_detach_array[@]}; do

    detach_arg=""
    if [ "${detach_gates}" = "True" ]; then
        detach_arg="--l2_detach_gates"
    fi

    for gates_lr in ${gate_lr_array[@]}; do

        gates_lr_arg="--gates_lr $gates_lr"

        for seed in ${seeds[@]}; do

            seed_arg="--seed $seed"

            # Collect common args for core_exp.py script
            python_args="${yaml_arg} ${seed_arg} ${checkpoint_arg} ${wandb_arg} ${gates_lr_arg} ${detach_arg}"

            for reg_type in ${reg_types[@]}; do

                for target_density in ${target_densities[@]}; do

                    # Set up _tdst_list (12 layers)
                    _tdst_list=$( regularization_list "${reg_type}" "${target_density}" 12 )

                    # Trigger the job
                    submit_sbatch "${python_args} -tdst ${_tdst_list}"
                done

            done
        done
    done
done