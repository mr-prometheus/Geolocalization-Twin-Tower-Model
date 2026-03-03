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
echo "Elevated-View Geolocalization Evaluation"
echo "VGGT output: $VGGT_OUTPUT_DIR"
echo "Aerial dir : $AERIAL_DIR"
echo "Start time : $(date)"
echo "================================================"

# ── 1. Ground-level view (0 deg) ─────────────────────────────────
echo ""
echo ">>> [1/4] Ground view (0 deg)"
python evaluate_elevated.py \
    --vggt_output_dir "$VGGT_OUTPUT_DIR" \
    --aerial_dir "$AERIAL_DIR" \
    --views ground \
    --backbone resnet18 \
    --embed_dim 512 \
    --batch_size 64 \
    --output_file results_elevated_ground.json

# ── 2. Elevated 45-degree view ────────────────────────────────────
echo ""
echo ">>> [2/4] Elevated 45-degree view"
python evaluate_elevated.py \
    --vggt_output_dir "$VGGT_OUTPUT_DIR" \
    --aerial_dir "$AERIAL_DIR" \
    --views elevated_45 \
    --backbone resnet18 \
    --embed_dim 512 \
    --batch_size 64 \
    --output_file results_elevated_45deg.json

# ── 3. Elevated 110-degree (near-top-down) view ───────────────────
echo ""
echo ">>> [3/4] Elevated 110-degree view"
python evaluate_elevated.py \
    --vggt_output_dir "$VGGT_OUTPUT_DIR" \
    --aerial_dir "$AERIAL_DIR" \
    --views elevated_110 \
    --backbone resnet18 \
    --embed_dim 512 \
    --batch_size 64 \
    --output_file results_elevated_110deg.json

# ── 4. All-views fusion (mean-pooled embeddings) ──────────────────
echo ""
echo ">>> [4/4] All-views fusion (ground + 45 + 110)"
python evaluate_elevated.py \
    --vggt_output_dir "$VGGT_OUTPUT_DIR" \
    --aerial_dir "$AERIAL_DIR" \
    --views ground,elevated_45,elevated_110 \
    --backbone resnet18 \
    --embed_dim 512 \
    --batch_size 32 \
    --output_file results_elevated_fusion.json

echo ""
echo "================================================"
echo "All evaluations complete!"
echo "End time: $(date)"
echo "================================================"

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "SUMMARY (R@1 / R@5 / R@10 / R@1%):"
for f in results_elevated_ground.json \
          results_elevated_45deg.json \
          results_elevated_110deg.json \
          results_elevated_fusion.json; do
    if [ -f "$f" ]; then
        r1=$(python  -c "import json; d=json.load(open('$f')); print(f\"{d['R@1']:.2f}\")")
        r5=$(python  -c "import json; d=json.load(open('$f')); print(f\"{d['R@5']:.2f}\")")
        r10=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['R@10']:.2f}\")")
        r1p=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['R@1%']:.2f}\")")
        echo "  $f : $r1 / $r5 / $r10 / $r1p"
    fi
done
