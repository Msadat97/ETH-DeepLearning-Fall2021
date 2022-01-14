#!/usr/bin/bash
#BSUB -n 4
#BSUB -W 24:00
#BSUB -o train_logs.txt
#BSUB -R "rusage[mem=15000, ngpus_excl_p=8]" 
#BSUB -R "select[gpu_model0==GeForceRTX2080Ti]"
#BSUB -G ls_infk


source $HOME/development/bin/activate

export WANDB_NOTES="stylegan3 + bg aug on fashion-synth data"
export WANDB_TAGS="stgan3-fs"

NGPU=8
BATCH=$((4*NGPU))

data_dir="$WORKDIR/data/fashion-synth.zip"
save_dir="$WORKDIR/experiments/training-runs-aliasfree-fs"

python train.py --outdir=$save_dir --data=$data_dir --cfg=stylegan3-t --gpus=$NGPU --batch=$BATCH --gamma=2 --snap=25 --dlr=0.0015 --glr=0.002

# python train.py --outdir=$SCRATCH/training-runs-aliasfree --data=$WORKDIR/data/ffhq-256x256.zip --cfg=stylegan3-t --gpus=$NGPU --batch=$BATCH --gamma=2 --aug=noaug --snap=25 --glr=0.0025 --dlr=0.0025