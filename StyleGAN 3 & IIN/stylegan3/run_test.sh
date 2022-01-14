#!/usr/bin/bash

export WANDB_MODE=disabled 

nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9 

NGPU=2
BATCH=$((4*NGPU))

data_dir="$WORKDIR/data/fashion-synth.zip"
save_dir="$SCRATCH/training-runs-test"

python train.py --outdir=$save_dir --data=$data_dir --cfg=stylegan3-t --gpus=$NGPU --batch=$BATCH --gamma=2 --snap=25 --dlr=0.0015 --glr=0.002