"""
Geolocalization via Multi-View Elevated Point-Cloud Rendering
=============================================================
Architecture
  Query  : MultiViewEncoder — ResNet backbone shared across 3 views
            (ground_0deg / elevated_45deg / elevated_110deg)
            → per-view tokens + CLS token → Transformer cross-view attention
            → CLS output → projection head → L2-normed embedding
  Gallery: AerialEncoder   — same ResNet backbone → projection head → L2-normed embedding

Training
  Symmetric InfoNCE contrastive loss on (multi-view clip, aerial image) pairs.
  Video IDs are split into train / val by --val_split fraction.

Evaluation
  R@1, R@5, R@10, R@1% (exact-match cosine retrieval) on the val split.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════
VIEWS = ['ground_0deg.png', 'elevated_45deg.png', 'elevated_110deg.png']
NUM_VIEWS = len(VIEWS)


# ══════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════

class MultiViewEncoder(nn.Module):
    """
    Takes all 3 rendered views simultaneously.
    Each view is encoded by a shared ResNet backbone, then cross-view
    Transformer attention (with a learnable CLS token) fuses them into
    a single embedding.

    Forward input : (B, 3, C, H, W)
    Forward output: (B, embed_dim)  — L2 normalised
    """

    def __init__(self, backbone='resnet18', embed_dim=512,
                 num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()

        if backbone == 'resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = net.fc.in_features          # 512
        elif backbone == 'resnet34':
            net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            feat_dim = net.fc.in_features          # 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        net.fc = nn.Identity()
        self.backbone = net
        self.feat_dim = feat_dim

        # One learnable embedding per view type so the transformer knows
        # which altitude each token came from (ground / 45° / 110°)
        self.view_pos = nn.Embedding(NUM_VIEWS, feat_dim)

        # Global CLS token — its output summarises all three views
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)

        # Cross-view Transformer (Pre-LN for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feat_dim)

        # Two-layer projection head (following SimCLR / MoCo convention)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embed_dim),
        )

    def forward(self, views):
        # views: (B, 3, C, H, W)
        B, V, C, H, W = views.shape

        # Run all views through the shared backbone in one batched call
        feats = self.backbone(views.view(B * V, C, H, W))   # (B*V, feat_dim)
        feats = feats.view(B, V, self.feat_dim)              # (B, 3, feat_dim)

        # Add view-type positional embeddings
        pos_ids = torch.arange(V, device=views.device)
        feats = feats + self.view_pos(pos_ids).unsqueeze(0)  # (B, 3, feat_dim)

        # Prepend CLS token → sequence length becomes 4
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, feat_dim)
        x = torch.cat([cls, feats], dim=1)                   # (B, 4, feat_dim)

        # Global cross-view attention
        x = self.transformer(x)                              # (B, 4, feat_dim)

        # Pool from CLS position, normalise, project
        fused = self.norm(x[:, 0])                           # (B, feat_dim)
        emb = self.projection(fused)                         # (B, embed_dim)
        return F.normalize(emb, p=2, dim=-1)


class AerialEncoder(nn.Module):
    """
    Standard single-image encoder for the aerial gallery.
    Mirrors the projection head depth of MultiViewEncoder.

    Forward input : (B, C, H, W)
    Forward output: (B, embed_dim)  — L2 normalised
    """

    def __init__(self, backbone='resnet18', embed_dim=512):
        super().__init__()

        if backbone == 'resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = net.fc.in_features
        elif backbone == 'resnet34':
            net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            feat_dim = net.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        net.fc = nn.Identity()
        self.backbone = net

        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embed_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        emb = self.projection(features)
        return F.normalize(emb, p=2, dim=-1)


# ══════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════

def info_nce_loss(q_emb, a_emb, temperature=0.07):
    """
    Symmetric InfoNCE (bidirectional cross-entropy).
    q_emb[i] is the positive for a_emb[i] and vice-versa.
    """
    logits = torch.matmul(q_emb, a_emb.T) / temperature   # (B, B)
    labels = torch.arange(len(q_emb), device=logits.device)
    loss_q = F.cross_entropy(logits, labels)
    loss_a = F.cross_entropy(logits.T, labels)
    return (loss_q + loss_a) / 2


# ══════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════

class PairedDataset(Dataset):
    """
    Training dataset.
    For each clip, yields all 3 view images + the matching aerial image.

    Structure:
      <vggt_output>/<video_id>/clip_XXXX/ground_0deg.png
                                         elevated_45deg.png
                                         elevated_110deg.png
      <aerial_root>/<video_id>/<clip_idx>.{png|jpg|jpeg}
    """

    def __init__(self, vggt_output_dir, aerial_root, video_ids=None, transform=None):
        self.transform = transform
        self.samples = []
        self._scan(Path(vggt_output_dir), Path(aerial_root), video_ids)

    def _scan(self, vggt_root, aerial_root, video_ids):
        skipped = 0
        for video_dir in sorted(vggt_root.iterdir()):
            if not video_dir.is_dir():
                continue
            vid = video_dir.name
            if video_ids is not None and vid not in video_ids:
                continue
            aerial_vid_dir = aerial_root / vid
            if not aerial_vid_dir.exists():
                continue

            for clip_dir in sorted(video_dir.iterdir()):
                if not clip_dir.is_dir() or not clip_dir.name.startswith('clip_'):
                    continue
                clip_idx = int(clip_dir.name.split('_')[1])

                view_paths = [clip_dir / v for v in VIEWS]
                if not all(p.exists() for p in view_paths):
                    skipped += 1
                    continue

                aerial_img = next(
                    (aerial_vid_dir / f"{clip_idx}{ext}"
                     for ext in ('.png', '.jpg', '.jpeg')
                     if (aerial_vid_dir / f"{clip_idx}{ext}").exists()),
                    None
                )
                if aerial_img is None:
                    skipped += 1
                    continue

                self.samples.append({
                    'video_id':    vid,
                    'clip_idx':    clip_idx,
                    'view_paths':  [str(p) for p in view_paths],
                    'aerial_path': str(aerial_img),
                    'unique_id':   f"{vid}_{clip_idx}",
                })

        if skipped:
            print(f"  [PairedDataset] Skipped {skipped} clips (missing views or aerial match)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        views = []
        for vp in s['view_paths']:
            img = Image.open(vp).convert('RGB')
            if self.transform:
                img = self.transform(img)
            views.append(img)

        aerial = Image.open(s['aerial_path']).convert('RGB')
        if self.transform:
            aerial = self.transform(aerial)

        return {
            'views':     torch.stack(views, dim=0),   # (3, C, H, W)
            'aerial':    aerial,                       # (C, H, W)
            'unique_id': s['unique_id'],
            'video_id':  s['video_id'],
            'clip_idx':  s['clip_idx'],
        }


class QueryDataset(Dataset):
    """
    Evaluation query dataset — all 3 views per clip, no aerial needed.
    """

    def __init__(self, vggt_output_dir, video_ids=None, transform=None):
        self.transform = transform
        self.samples = []
        self._scan(Path(vggt_output_dir), video_ids)

    def _scan(self, vggt_root, video_ids):
        for video_dir in sorted(vggt_root.iterdir()):
            if not video_dir.is_dir():
                continue
            vid = video_dir.name
            if video_ids is not None and vid not in video_ids:
                continue
            for clip_dir in sorted(video_dir.iterdir()):
                if not clip_dir.is_dir() or not clip_dir.name.startswith('clip_'):
                    continue
                clip_idx = int(clip_dir.name.split('_')[1])
                view_paths = [clip_dir / v for v in VIEWS]
                if not all(p.exists() for p in view_paths):
                    continue
                self.samples.append({
                    'video_id':   vid,
                    'clip_idx':   clip_idx,
                    'view_paths': [str(p) for p in view_paths],
                    'unique_id':  f"{vid}_{clip_idx}",
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        views = []
        for vp in s['view_paths']:
            img = Image.open(vp).convert('RGB')
            if self.transform:
                img = self.transform(img)
            views.append(img)
        return {
            'views':     torch.stack(views, dim=0),   # (3, C, H, W)
            'video_id':  s['video_id'],
            'clip_idx':  s['clip_idx'],
            'unique_id': s['unique_id'],
        }


class AerialGalleryDataset(Dataset):
    """
    Evaluation gallery — aerial images, optionally filtered to valid pairs.
    """

    def __init__(self, aerial_root, valid_pairs=None, transform=None):
        self.transform = transform
        self.samples = []
        self._scan(Path(aerial_root), valid_pairs)

    def _scan(self, aerial_root, valid_pairs):
        for video_dir in sorted(aerial_root.iterdir()):
            if not video_dir.is_dir():
                continue
            vid = video_dir.name
            for img_file in sorted(video_dir.iterdir()):
                if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                frame_idx = int(img_file.stem)
                if valid_pairs is not None and (vid, frame_idx) not in valid_pairs:
                    continue
                self.samples.append({
                    'video_id':   vid,
                    'frame_idx':  frame_idx,
                    'image_path': str(img_file),
                    'unique_id':  f"{vid}_{frame_idx}",
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image':     image,
            'video_id':  s['video_id'],
            'frame_idx': s['frame_idx'],
            'unique_id': s['unique_id'],
        }


# ══════════════════════════════════════════════════════════════════
# Embedding extraction (no-grad, for evaluation)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_query_embeddings(model, loader, device):
    model.eval()
    embeddings, unique_ids = [], []
    for batch in tqdm(loader, desc='  [eval] query embeddings'):
        emb = model(batch['views'].to(device))
        embeddings.append(emb.cpu())
        unique_ids.extend(batch['unique_id'])
    return torch.cat(embeddings, dim=0), unique_ids


@torch.no_grad()
def extract_aerial_embeddings(model, loader, device):
    model.eval()
    embeddings, unique_ids = [], []
    for batch in tqdm(loader, desc='  [eval] gallery embeddings'):
        emb = model(batch['image'].to(device))
        embeddings.append(emb.cpu())
        unique_ids.extend(batch['unique_id'])
    return torch.cat(embeddings, dim=0), unique_ids


# ══════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════

def compute_recall_metrics(query_emb, gallery_emb, query_ids, gallery_ids):
    sim = torch.matmul(query_emb, gallery_emb.T)
    sorted_idx = torch.argsort(sim, dim=1, descending=True)
    N_q, N_g = len(query_ids), len(gallery_ids)

    results = {}
    for k in [1, 5, 10]:
        correct = sum(
            query_ids[i] in [gallery_ids[j] for j in sorted_idx[i, :k].tolist()]
            for i in range(N_q)
        )
        results[f'R@{k}'] = correct / N_q * 100

    k1p = max(1, int(N_g * 0.01))
    correct = sum(
        query_ids[i] in [gallery_ids[j] for j in sorted_idx[i, :k1p].tolist()]
        for i in range(N_q)
    )
    results['R@1%'] = correct / N_q * 100
    return results


def print_results(results, label=''):
    tag = f'  [{label}]' if label else ''
    print(f"\n{'='*60}")
    print(f"Multi-View Geolocalization Results{tag}")
    print(f"  Views: ground_0deg + elevated_45deg + elevated_110deg")
    print(f"{'='*60}")
    print(f"  Queries : {results.get('num_queries', '?')}")
    print(f"  Gallery : {results.get('num_gallery', '?')}")
    print(f"  R@1     : {results['R@1']:.2f}%")
    print(f"  R@5     : {results['R@5']:.2f}%")
    print(f"  R@10    : {results['R@10']:.2f}%")
    print(f"  R@1%    : {results['R@1%']:.2f}%")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Views   : ground_0deg + elevated_45deg + elevated_110deg (fixed)")
    print(f"Mode    : {'train + eval' if args.train else 'eval only (pretrained backbone)'}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Video-ID split (or explicit train/test dirs) ──────────────
    if args.train_dir and args.test_dir:
        # Use pre-split directories
        train_root = Path(args.train_dir)
        test_root  = Path(args.test_dir)
        train_ids  = set(d.name for d in train_root.iterdir() if d.is_dir())
        val_ids    = set(d.name for d in test_root.iterdir()  if d.is_dir())
        print(f"\nUsing explicit split:")
        print(f"  Train dir : {train_root}  ({len(train_ids)} videos)")
        print(f"  Test  dir : {test_root}   ({len(val_ids)} videos)")
        query_root = str(test_root)
    else:
        if not args.vggt_output_dir:
            raise ValueError("Provide --vggt_output_dir or both --train_dir and --test_dir")
        # Fall back to random split of vggt_output_dir
        all_video_ids = sorted(
            d.name for d in Path(args.vggt_output_dir).iterdir() if d.is_dir()
        )
        random.shuffle(all_video_ids)
        n_val     = max(1, int(len(all_video_ids) * args.val_split))
        val_ids   = set(all_video_ids[:n_val])
        train_ids = set(all_video_ids[n_val:])
        print(f"\nVideos  : {len(all_video_ids)} total  |  train {len(train_ids)}  |  val {len(val_ids)}")
        query_root = args.vggt_output_dir

    # ── Evaluation datasets (always built) ────────────────────────
    val_query   = QueryDataset(query_root, val_ids, transform)
    val_pairs   = {(s['video_id'], s['clip_idx']) for s in val_query.samples}
    val_gallery = AerialGalleryDataset(args.aerial_dir, valid_pairs=val_pairs, transform=transform)

    print(f"Val queries : {len(val_query)}  |  Val gallery: {len(val_gallery)}")

    val_q_loader = DataLoader(val_query,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_a_loader = DataLoader(val_gallery, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Models ───────────────────────────────────────────────────
    query_encoder  = MultiViewEncoder(
        args.backbone, args.embed_dim, args.num_heads, args.num_layers
    ).to(device)
    aerial_encoder = AerialEncoder(args.backbone, args.embed_dim).to(device)

    # ── Load checkpoint if resuming ───────────────────────────────
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        query_encoder.load_state_dict(ckpt['query_encoder'])
        aerial_encoder.load_state_dict(ckpt['aerial_encoder'])
        print(f"Resumed from {args.resume}")

    # ── Training ─────────────────────────────────────────────────
    if args.train:
        train_aerial = args.train_aerial_dir if args.train_aerial_dir else args.aerial_dir
        if not Path(train_aerial).exists():
            raise RuntimeError(f"Train aerial dir not found: {train_aerial}\n"
                               "Pass --train_aerial_dir pointing to e.g. train_gps/")
        train_dataset = PairedDataset(
            args.train_dir if args.train_dir else args.vggt_output_dir,
            train_aerial, train_ids, transform
        )
        if len(train_dataset) == 0:
            raise RuntimeError("No training pairs found. Check paths and val_split.")
        print(f"Train pairs : {len(train_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )

        optimizer = torch.optim.AdamW(
            list(query_encoder.parameters()) + list(aerial_encoder.parameters()),
            lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        best_r1      = 0.0
        best_results = {}

        for epoch in range(1, args.epochs + 1):
            query_encoder.train()
            aerial_encoder.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
                views  = batch['views'].to(device)    # (B, 3, C, H, W)
                aerial = batch['aerial'].to(device)   # (B, C, H, W)

                q_emb = query_encoder(views)
                a_emb = aerial_encoder(aerial)
                loss  = info_nce_loss(q_emb, a_emb, args.temperature)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(query_encoder.parameters()) + list(aerial_encoder.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch:>3d}  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

            # ── Val after each epoch ──────────────────────────────
            q_emb, q_ids = extract_query_embeddings(query_encoder,  val_q_loader, device)
            a_emb, a_ids = extract_aerial_embeddings(aerial_encoder, val_a_loader, device)

            results = compute_recall_metrics(q_emb, a_emb, q_ids, a_ids)
            results.update({
                'num_queries': len(q_ids),
                'num_gallery': len(a_ids),
                'epoch': epoch,
                'loss':  avg_loss,
            })

            print(f"  R@1={results['R@1']:.2f}%  R@5={results['R@5']:.2f}%  "
                  f"R@10={results['R@10']:.2f}%  R@1%={results['R@1%']:.2f}%")

            if results['R@1'] > best_r1:
                best_r1      = results['R@1']
                best_results = results.copy()
                torch.save({
                    'query_encoder':  query_encoder.state_dict(),
                    'aerial_encoder': aerial_encoder.state_dict(),
                    'args':           vars(args),
                    'results':        results,
                }, args.checkpoint)
                print(f"  * New best R@1={best_r1:.2f}%  ->  saved to {args.checkpoint}")

        print_results(best_results, label='best val')

    else:
        # ── Eval-only (zero-shot with pretrained backbone) ────────
        q_emb, q_ids = extract_query_embeddings(query_encoder,  val_q_loader, device)
        a_emb, a_ids = extract_aerial_embeddings(aerial_encoder, val_a_loader, device)

        results = compute_recall_metrics(q_emb, a_emb, q_ids, a_ids)
        results.update({'num_queries': len(q_ids), 'num_gallery': len(a_ids)})
        print_results(results, label='zero-shot')

        best_results = results

    # ── Save results ──────────────────────────────────────────────
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(best_results, f, indent=2)
        print(f"Results saved to {args.output_file}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Multi-view elevated geolocalization — attention + contrastive'
    )
    # Paths
    p.add_argument('--vggt_output_dir', default=None,
                   help='Root vggt-output dir (used for random val_split mode): '
                        '<video_id>/clip_XXXX/{ground,elevated_45,elevated_110}.png')
    p.add_argument('--train_dir', default=None,
                   help='Pre-split train dir (overrides vggt_output_dir + val_split)')
    p.add_argument('--test_dir', default=None,
                   help='Pre-split test dir (overrides vggt_output_dir + val_split)')
    p.add_argument('--aerial_dir', required=True,
                   help='Aerial gallery dir: <video_id>/<clip_idx>.{png|jpg}')
    p.add_argument('--train_aerial_dir', default=None,
                   help='Aerial dir for training pairs (defaults to --aerial_dir if not set)')

    # Model
    p.add_argument('--backbone',   default='resnet18', choices=['resnet18', 'resnet34'])
    p.add_argument('--embed_dim',  type=int, default=512)
    p.add_argument('--num_heads',  type=int, default=8,
                   help='Attention heads in cross-view Transformer')
    p.add_argument('--num_layers', type=int, default=2,
                   help='Transformer encoder layers')

    # Training
    p.add_argument('--train',       action='store_true',
                   help='Run contrastive training before evaluation')
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.07,
                   help='InfoNCE temperature')
    p.add_argument('--val_split',   type=float, default=0.2,
                   help='Fraction of video IDs held out for validation')

    # I/O
    p.add_argument('--input_size',  type=int, default=224)
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--checkpoint',  default='best_elevated.pth')
    p.add_argument('--resume',      default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--output_file', default='results_elevated.json')

    main(p.parse_args())
