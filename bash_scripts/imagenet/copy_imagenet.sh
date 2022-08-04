#!/bin/bash


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