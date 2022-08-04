#!/bin/bash

# This basic script is used to run MNIST, CIFAR-10 and CIFAR-100 experiments

# Did not experience preemption on small datasets
# # -----------------------------------------------------------------------------
# #                                  SET RUN ID 
# # -----------------------------------------------------------------------------
# # Environment variable $SLURM_JOB_ID persists on preemption 
export WANDB_RUN_ID=$SLURM_JOB_ID
# # -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#                        LOAD MODULES AND ENVIRONMENTS
# -----------------------------------------------------------------------------
# L0_SRC_PATH is an environment variable set by trigger script
# Getting helper functions - Do not delete
source $L0_SRC_PATH/bash_scripts/bash_helpers.sh
load_modules_and_env $USER
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#                                  START EXPERIMENT
# -----------------------------------------------------------------------------
# cd back into L0_SRC_PATH to avoid writing random stuff into imagenet/train (prev location)
cd ${L0_SRC_PATH} # this is an environment variable set by trigger_imagenet.sh

# Actually start experiment. NUM_GPUS is an environment variable set in trigger script
torchrun --node_rank 0 --nnodes 1 --nproc_per_node=$NUM_GPUS ./core_exp.py "$@" 

# -----------------------------------------------------------------------------
