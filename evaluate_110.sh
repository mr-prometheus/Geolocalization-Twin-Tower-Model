#!/bin/bash
#SBATCH -J eval_110deg
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem32
#SBATCH --output=runs/eval_110deg_%j.out

# ==== CONFIGURATION ====
VGGT_OUTPUT_DIR="/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/vggt-output"
AERIAL_DIR="/home/c3-0/datasets/GAMa_dataset/val_gps"
# =======================

module load anaconda3
module load cuda/11.0
export PATH="/home/de575594/.conda/envs/geolocal/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate geolocal

mkdir -p runs

echo "================================================"
echo "Elevated-110deg Zero-Shot Geolocalization"
echo "  Query  : elevated_110deg.png (vggt-output)"
echo "  Gallery: $AERIAL_DIR"
echo "  Start  : $(date)"
echo "================================================"

python evaluate_110.py \
    --vggt_output_dir "$VGGT_OUTPUT_DIR" \
    --aerial_dir      "$AERIAL_DIR" \
    --backbone        resnet18 \
    --embed_dim       512 \
    --input_size      224 \
    --batch_size      64 \
    --num_workers     4 \
    --output_file     results_110deg.json

echo ""
echo "================================================"
echo "Done! End time: $(date)"
echo "================================================"

if [ -f results_110deg.json ]; then
    python -c "
import json
d = json.load(open('results_110deg.json'))
print('RESULTS:')
print(f\"  R@1={d['R@1']:.2f}%  R@5={d['R@5']:.2f}%  R@10={d['R@10']:.2f}%  R@1%={d['R@1%']:.2f}%\")
"
fi
