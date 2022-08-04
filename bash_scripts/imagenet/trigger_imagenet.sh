#!/bin/bash

# -----------------------------------------------------------------------------
#                               TO BE CUSTOMIZED
# -----------------------------------------------------------------------------
# Directory containing the source code
L0_SRC_PATH="$HOME/github/constrained_l0"

# Getting helper functions - Do not delete
source $L0_SRC_PATH/bash_scripts/bash_helpers.sh

# Bash script with call to Python (also copies the dataset to SLURM_TMPDIR) 
main_bash_script="${L0_SRC_PATH}/bash_scripts/imagenet/run_imagenet.sh"

# SLURM options
slurm_log_dir="$HOME/slurm_logs/const/"
notify_email="" # Leave empty ("") for no email
partition="long"

# Number of GPUs. At most 8. Memory is set automatically
NUM_GPUS=4

# WandB -- You will likely want to keep these untouched!
use_wandb=True 
run_group="rebuttal"
wandb_dir="${SCRATCH}/wandb_logs/"

# Directory for storing training checkpoints (auto-created if it doesn't exist)
checkpoint_dir="${SCRATCH}/l0_checkpoints/imagenet"

# Config lists
declare -a seeds=(1)
declare -a target_densities=(0.3 0.5 0.7 0.9)
declare -a lmbdas=()
declare -a reg_types=("model" "layer") 

# -----------------------------------------------------------------------------

# Resources, given number of GPUs requested
if [ "${partition}" = "main" ];
then
  # Set the maximum allowed on main
  mem=48
  cpus=4
  time=30:00:00
else
  mem=$(( $NUM_GPUS * 32 ))
  cpus=8
  time=72:00:00
fi

# The parameter of this function is the python arguments
submit_sbatch () {
  sbatch --job-name=constl0-imagenet-slurm-%j.out \
      --time=$time \
      --cpus-per-task $cpus \
      --mem="$mem"G \
      --gres=gpu:$NUM_GPUS \
      --nodes=1 \
      --ntasks-per-node=$NUM_GPUS \
      --partition=$partition \
      --exclude=rtx[1-7] \
      --output=$slurm_log_dir/imagenet-slurm-%j.out \
      --mail-type=ALL --mail-user=$notify_email \
      $main_bash_script $1
}


export L0_SRC_PATH
export NUM_GPUS

# Get basic configuration from YAML file
yaml_arg="-yaml ${L0_SRC_PATH}/configs/defaults/imagenet_defaults.yml"

checkpoint_arg="--checkpoint_dir $checkpoint_dir"

# Set up WandB flags
wandb_arg=$( create_wandb_arg "${use_wandb}" "${run_group}" "${wandb_dir}" )


for seed in ${seeds[@]}; do

  seed_arg="--seed $seed"
  
  # Collect common args for core_exp.py script
  python_args="${yaml_arg} ${seed_arg} ${checkpoint_arg} ${wandb_arg}"

  for reg_type in ${reg_types[@]}; do
    
    if [ "${reg_type}" = "model" ];
    then
      python_args="${python_args} --dual_lr 3.0e-4"
    else
      python_args="${python_args} --dual_lr 3.0e-5"
    fi

    for target_density in ${target_densities[@]}; do

      # Set up _tdst_list
      _tdst_list=$( regularization_list "${reg_type}" "${target_density}" 48 )

      # Trigger the job
      submit_sbatch "${python_args} -tdst ${_tdst_list}"

    done

    for lmbda in ${lmbdas[@]}; do

      # Set up _lmbda_list
      _lmbda_list=$( regularization_list "${reg_type}" "${lmbda}" 48 )

      # Trigger the job
      submit_sbatch "${python_args} --lmbdas $_lmbda_list"

    done

  done
done