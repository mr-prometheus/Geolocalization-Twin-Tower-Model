#!/bin/bash
#SBATCH -J eval_elevated_split
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1 -C gmem32
#SBATCH --output=runs/eval_elevated_split_%j.out

# ==== CONFIGURATION ====
BASE="/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day"
TRAIN_DIR="$BASE/vggt-output-train-rendered"
TEST_DIR="$BASE/vggt-output-test-rendered"
AERIAL_DIR="/home/c3-0/datasets/GAMa_dataset/val_gps"
# =======================

module load anaconda3
module load cuda/11.0
export PATH="/home/de575594/.conda/envs/geolocal/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate geolocal

mkdir -p runs
RUN_DIR="results/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

echo "================================================"
echo "Multi-View Elevated Geolocalization  (BDD split)"
echo "  Architecture : ResNet18 + cross-view Transformer (CLS attention)"
echo "  Views        : ground_0deg + elevated_45deg + elevated_110deg"
echo "  Loss         : symmetric InfoNCE contrastive"
echo "  Train dir    : $TRAIN_DIR"
echo "  Test  dir    : $TEST_DIR"
echo "  Aerial dir   : $AERIAL_DIR"
echo "  Results dir  : $RUN_DIR"
echo "  Start time   : $(date)"
echo "================================================"

# ── [1/2] Zero-shot eval on test split (no training) ─────────────
echo ""
echo ">>> [1/2] Zero-shot evaluation on test split"
python evaluate_elevated.py \
    --train_dir    "$TRAIN_DIR" \
    --test_dir     "$TEST_DIR" \
    --aerial_dir   "$AERIAL_DIR" \
    --backbone     resnet18 \
    --embed_dim    512 \
    --num_heads    8 \
    --num_layers   2 \
    --batch_size   64 \
    --num_workers  4 \
    --output_file  "$RUN_DIR/results_split_zeroshot.json"

# ── [2/2] Contrastive training on train split, eval on test split ─
echo ""
echo ">>> [2/2] Contrastive training (train split) + evaluation (test split)"
python evaluate_elevated.py \
    --train_dir    "$TRAIN_DIR" \
    --test_dir     "$TEST_DIR" \
    --aerial_dir   "$AERIAL_DIR" \
    --train \
    --backbone     resnet18 \
    --embed_dim    512 \
    --num_heads    8 \
    --num_layers   2 \
    --epochs       20 \
    --lr           1e-4 \
    --weight_decay 1e-4 \
    --temperature  0.07 \
    --batch_size   64 \
    --num_workers  4 \
    --checkpoint   "$RUN_DIR/best_elevated_split.pth" \
    --output_file  "$RUN_DIR/results_split_trained.json"

echo ""
echo "================================================"
echo "Done! End time: $(date)"
echo "================================================"

echo ""
echo "SUMMARY  (run: $RUN_DIR)"
for tag in zeroshot trained; do
    f="$RUN_DIR/results_split_${tag}.json"
    if [ -f "$f" ]; then
        python -c "
import json
d = json.load(open('$f'))
print(f'  results_split_${tag}.json')
print(f\"    R@1={d['R@1']:.2f}%  R@5={d['R@5']:.2f}%  R@10={d['R@10']:.2f}%  R@1%={d['R@1%']:.2f}%\")
"
    fi
done
