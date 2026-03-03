#!/bin/bash
#SBATCH -J eval_elevated
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1 -C gmem32
#SBATCH --output=runs/eval_elevated_%j.out

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
echo "Multi-View Elevated Geolocalization"
echo "  Architecture : ResNet18 + cross-view Transformer (CLS attention)"
echo "  Views        : ground_0deg + elevated_45deg + elevated_110deg"
echo "  Loss         : symmetric InfoNCE contrastive"
echo "  VGGT output  : $VGGT_OUTPUT_DIR"
echo "  Aerial dir   : $AERIAL_DIR"
echo "  Start time   : $(date)"
echo "================================================"

# ── Option A: zero-shot eval (pretrained backbone, no training) ──
echo ""
echo ">>> [1/2] Zero-shot evaluation (pretrained backbone)"
python evaluate_elevated.py \
    --vggt_output_dir "$VGGT_OUTPUT_DIR" \
    --aerial_dir      "$AERIAL_DIR" \
    --backbone        resnet18 \
    --embed_dim       512 \
    --num_heads       8 \
    --num_layers      2 \
    --val_split       0.2 \
    --batch_size      64 \
    --num_workers     4 \
    --output_file     results_elevated_zeroshot.json

# ── Option B: contrastive training then eval ─────────────────────
echo ""
echo ">>> [2/2] Contrastive training + evaluation"
python evaluate_elevated.py \
    --vggt_output_dir "$VGGT_OUTPUT_DIR" \
    --aerial_dir      "$AERIAL_DIR" \
    --train \
    --backbone        resnet18 \
    --embed_dim       512 \
    --num_heads       8 \
    --num_layers      2 \
    --epochs          20 \
    --lr              1e-4 \
    --weight_decay    1e-4 \
    --temperature     0.07 \
    --val_split       0.2 \
    --batch_size      64 \
    --num_workers     4 \
    --checkpoint      best_elevated.pth \
    --output_file     results_elevated_trained.json

echo ""
echo "================================================"
echo "Done! End time: $(date)"
echo "================================================"

echo ""
echo "SUMMARY:"
for f in results_elevated_zeroshot.json results_elevated_trained.json; do
    if [ -f "$f" ]; then
        python -c "
import json
d = json.load(open('$f'))
print(f'  $f')
print(f\"    R@1={d['R@1']:.2f}%  R@5={d['R@5']:.2f}%  R@10={d['R@10']:.2f}%  R@1%={d['R@1%']:.2f}%\")
"
    fi
done
