#!/bin/bash
#SBATCH -J bev_eval
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1 -C gmem32
#SBATCH --output=runs/bev_eval_%j.out

module load anaconda3
module load cuda/11.0

export PATH="/home/de575594/.conda/envs/geolocal/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate geolocal

BEV_DIR="/home/de575594/Deepan/CV/geolocalization/vggt-long/bev-projection/bev_outputs"
AERIAL_DIR="/home/c3-0/datasets/GAMa_dataset/val_gps"

python evaluate.py \
    --bev_dir $BEV_DIR \
    --aerial_dir $AERIAL_DIR \
    --variant 2d \
    --backbone resnet18 \
    --embed_dim 512 \
    --batch_size 64 \
    --output_file results_bev_2d.json

python evaluate.py \
    --bev_dir $BEV_DIR \
    --aerial_dir $AERIAL_DIR \
    --variant 3d \
    --backbone resnet18 \
    --embed_dim 512 \
    --temporal_window 8 \
    --batch_size 32 \
    --output_file results_bev_3d.json