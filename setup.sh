#!/bin/bash
#SBATCH -J sip_rl
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem12
#SBATCH --time=02:00:00
#SBATCH --output=setup_job.out

# Load modules
module load anaconda3
module load cuda/12.0

# Initialize conda properly (IMPORTANT on SLURM)
eval "$(conda shell.bash hook)"

# Activate existing environment
conda activate geolocal

# Install packages (choose ONE of the two sections below)
pip install --no-cache-dir torch torchvision pillow opencv-python pandas numpy

python - << 'EOF'
import torch, cv2, PIL, numpy, pandas
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
