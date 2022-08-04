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

# WandB -- You will likely want to keep these untouched!
use_wandb=True
run_group="control"
wandb_dir="${SCRATCH}/wandb_logs/"

# Directory for storing training checkpoints (auto-created if it doesn't exist)
checkpoint_dir="${SCRATCH}/l0_checkpoints/cifar"

# Number of GPUs.
NUM_GPUS=2

# We produce controllability plots only for the CIFAR10 dataset, not CIFAR100
declare -a dataset_names=("cifar10") # "cifar100")

# Config lists
declare -a seeds=(1 2 3)
declare -a target_densities=(0.2 0.35 0.5 0.65 0.8)
declare -a lmbdas=(1e-4 1e-3 1e-2 1e-1 1e0)
declare -a reg_types=("layer" "model")

# -----------------------------------------------------------------------------

# The parameter of this function is the python arguments
submit_sbatch () {
    sbatch --job-name=constl0-cifar-slurm-%j.out \
        --time=10:00:00 \
        --cpus-per-task 6 \
        --mem=32G \
        --gres=gpu:$NUM_GPUS \
        --partition=$partition \
        --exclude=rtx[1-7] \
        --output=$slurm_log_dir/cifar-slurm-%j.out \
        --mail-type=ALL --mail-user=$notify_email \
        $main_bash_script $1
}

export L0_SRC_PATH
export NUM_GPUS

# Get basic configuration from YAML file
yaml_arg="-yaml ${L0_SRC_PATH}/configs/defaults/cifar_defaults.yml"

checkpoint_arg="--checkpoint_dir $checkpoint_dir"

# Set up WandB flags
wandb_arg=$( create_wandb_arg "${use_wandb}" "${run_group}" "${wandb_dir}" )


for dataset_name in ${dataset_names[@]}; do

    dataset_arg="--dataset_name $dataset_name"

    for seed in ${seeds[@]}; do

        seed_arg="--seed $seed"

        # Collect common args for core_exp.py script
        python_args="${yaml_arg} ${seed_arg} ${checkpoint_arg} ${wandb_arg} ${dataset_arg}"

        for reg_type in ${reg_types[@]}; do

            for target_density in ${target_densities[@]}; do

                # Set up _tdst_list (12 layers)
                _tdst_list=$( regularization_list "${reg_type}" "${target_density}" 12 )

                # Trigger the job
                submit_sbatch "${python_args} -tdst ${_tdst_list}"
            done

            for lmbda in ${lmbdas[@]}; do

                # Set up _lmbda_list (12 layers)
                _lmbda_list=$( regularization_list "${reg_type}" "${lmbda}" 12 )

                # Trigger the job
                submit_sbatch "${python_args} --lmbdas $_lmbda_list"
            done

        done
    done
done