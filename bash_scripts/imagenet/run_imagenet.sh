#!/bin/bash

# -----------------------------------------------------------------------------
#                                  SET RUN ID 
# -----------------------------------------------------------------------------
# Environment variable $SLURM_JOB_ID persists on preemption 
export WANDB_RUN_ID=$SLURM_JOB_ID

# -----------------------------------------------------------------------------
#                        LOAD MODULES AND ENVIRONMENTS
# -----------------------------------------------------------------------------
# L0_SRC_PATH is an environment variable set by trigger script
# Getting helper functions - Do not delete
source $L0_SRC_PATH/bash_scripts/bash_helpers.sh
load_modules_and_env $USER

# -----------------------------------------------------------------------------
#                                  COPY IMAGENET
# -----------------------------------------------------------------------------
COPY_START=$(date +%s.%N)

echo "Started copying test data"
mkdir -p $SLURM_TMPDIR/imagenet
cp -r /network/datasets/imagenet.var/imagenet_torchvision/val $SLURM_TMPDIR/imagenet
echo "Finished copying test data"

echo "Started copying train data"
mkdir -p $SLURM_TMPDIR/imagenet/train
cd       $SLURM_TMPDIR/imagenet/train
tar  -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'
echo "Finished unpacking train data"

COPY_END=$(date +%s.%N)
DIFF=$(echo "$COPY_END - $COPY_START" | bc)
echo "Total elapsed copy and extract time:" $DIFF "seconds"
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#                                  START EXPERIMENT
# -----------------------------------------------------------------------------
# cd back into L0_SRC_PATH to avoid writing random stuff into imagenet/train (prev location)
cd ${L0_SRC_PATH} # this is an environment variable set in trigger script

# Actually start experiment. NUM_GPUS is an environment variable set in trigger script
torchrun --node_rank 0 --nnodes 1 --nproc_per_node=$NUM_GPUS ./core_exp.py "$@" 

# -----------------------------------------------------------------------------
