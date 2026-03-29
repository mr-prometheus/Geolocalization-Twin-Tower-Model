# Multi-View Elevated Geolocalization

Contrastive learning pipeline for ground-to-aerial geolocalization using multi-view point-cloud renderings from VGGT.

---

## Overview

Each ground-level clip is represented by **three rendered views** at different elevations. A contrastive model learns to align these multi-view query embeddings with their corresponding aerial gallery images.

```
Query (clip)                      Gallery
┌─────────────────────┐           ┌─────────────┐
│  ground_0deg.png    │           │             │
│  elevated_45deg.png │ ──────►   │ aerial.png  │
│  elevated_110deg.png│           │             │
└─────────────────────┘           └─────────────┘
    MultiViewEncoder               AerialEncoder
         │                               │
    L2-normed                       L2-normed
    512-dim emb                     512-dim emb
```

---

## Architecture

### MultiViewEncoder (Query Side)

Encodes all 3 views simultaneously into a single fused embedding.

| Component | Detail |
|-----------|--------|
| Backbone | ResNet18 or ResNet34, ImageNet pretrained, shared across all 3 views |
| View positional embeddings | Learnable `nn.Embedding(3, feat_dim)` — tells the Transformer which altitude each token came from |
| CLS token | Learnable parameter prepended to the view token sequence |
| Cross-view Transformer | Pre-LayerNorm TransformerEncoder, 2 layers, 8 heads, FFN dim = 4× feat_dim |
| Projection head | Linear → ReLU → Linear (following SimCLR/MoCo convention) |
| Output | L2-normalized 512-dim embedding |

**Forward pass:**
1. All 3 views are stacked as `(B, 3, C, H, W)` and passed through the shared backbone in one batched call → `(B, 3, feat_dim)`
2. View-type positional embeddings are added
3. A CLS token is prepended → sequence of length 4
4. Transformer attends across all 4 tokens
5. CLS output is projected and L2-normalized

### AerialEncoder (Gallery Side)

Standard single-image encoder mirroring the projection head depth of `MultiViewEncoder`.

| Component | Detail |
|-----------|--------|
| Backbone | Same ResNet18/34, ImageNet pretrained |
| Projection head | Linear → ReLU → Linear |
| Output | L2-normalized 512-dim embedding |

---

## Training

### Data

- **Source:** VGGT output directory (`<video_id>/clip_XXXX/`) paired with aerial gallery (`<video_id>/<clip_idx>.png`)
- **Pair requirement:** A clip is included only if all 3 view images **and** a matching aerial image exist
- **Split:** Video IDs are shuffled and split by `--val_split` (default 20%) — the split is at the video level to prevent data leakage

### Loss — Symmetric InfoNCE

For a batch of B pairs:

```
logits = (q_emb @ a_emb.T) / temperature          # (B, B)
loss   = 0.5 * CE(logits, labels) + 0.5 * CE(logits.T, labels)
```

The diagonal is the ground-truth match. Both directions (query→aerial and aerial→query) are optimized simultaneously. Temperature defaults to `0.07`.

### Optimizer

| Setting | Default |
|---------|---------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR schedule | Cosine annealing over all epochs |
| Gradient clipping | max norm = 1.0 |
| Epochs | 20 |
| Batch size | 64 |

### What is learned

- ResNet backbone weights (fine-tuned from ImageNet init)
- Learnable view-type positional embeddings
- CLS token
- Cross-view Transformer (all layers)
- Projection heads in both encoders

### Checkpointing

The best checkpoint (by val R@1) is saved to `best_elevated.pth`. Training can be resumed with `--resume <path>`.

---

## Evaluation

Retrieval metrics computed on the val split after every epoch (or once in zero-shot mode).

| Metric | Description |
|--------|-------------|
| R@1 | Query's true match is the top-1 result |
| R@5 | True match appears in top-5 |
| R@10 | True match appears in top-10 |
| R@1% | True match appears in top 1% of gallery |

Match is determined by `unique_id` equality: `{video_id}_{clip_idx}` (query) vs `{video_id}_{frame_idx}` (gallery).

---

## Usage

### Eval-only (zero-shot, no training)

```bash
python evaluate_elevated.py \
  --vggt_output_dir /path/to/vggt-output \
  --aerial_dir      /path/to/aerial_gallery
```

Uses frozen ImageNet-pretrained weights — no contrastive training is run.

### Train + eval

```bash
python evaluate_elevated.py \
  --vggt_output_dir /path/to/vggt-output \
  --aerial_dir      /path/to/aerial_gallery \
  --train \
  --epochs 20 \
  --batch_size 64 \
  --backbone resnet18
```

### Resume from checkpoint

```bash
python evaluate_elevated.py \
  --vggt_output_dir /path/to/vggt-output \
  --aerial_dir      /path/to/aerial_gallery \
  --train \
  --resume best_elevated.pth
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--vggt_output_dir` | required | Root of VGGT rendered views |
| `--aerial_dir` | required | Root of aerial gallery images |
| `--backbone` | `resnet18` | `resnet18` or `resnet34` |
| `--embed_dim` | `512` | Output embedding dimension |
| `--num_heads` | `8` | Transformer attention heads |
| `--num_layers` | `2` | Transformer encoder layers |
| `--train` | flag | Enable contrastive training |
| `--epochs` | `20` | Training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--temperature` | `0.07` | InfoNCE temperature |
| `--val_split` | `0.2` | Fraction of videos held out for val |
| `--batch_size` | `64` | Batch size |
| `--checkpoint` | `best_elevated.pth` | Where to save best model |
| `--resume` | `None` | Checkpoint path to resume from |
| `--output_file` | `results_elevated.json` | Where to save final metrics |

---

## Data Directory Structure

```
vggt-output/
└── <video_id>/
    └── clip_XXXX/
        ├── ground_0deg.png
        ├── elevated_45deg.png
        └── elevated_110deg.png

aerial_gallery/
└── <video_id>/
    └── <clip_idx>.png   (or .jpg / .jpeg)
```
