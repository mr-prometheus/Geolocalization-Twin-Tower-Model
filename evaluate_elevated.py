import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json


# ─────────────────────────────────────────────
# Encoders  (identical to evaluate.py)
# ─────────────────────────────────────────────
class ImageEncoder(nn.Module):
    """Single-image encoder used for both elevated-view queries and aerial gallery."""
    def __init__(self, backbone='resnet18', embed_dim=512):
        super().__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        return nn.functional.normalize(embeddings, p=2, dim=-1)


# ─────────────────────────────────────────────
# Dataset: elevated / ground views from vggt-output
# ─────────────────────────────────────────────
VIEW_FILENAMES = {
    'ground':       'ground_0deg.png',
    'elevated_45':  'elevated_45deg.png',
    'elevated_110': 'elevated_110deg.png',
}


class ElevatedViewDataset(Dataset):
    """
    Scans the vggt-output directory tree:
        <vggt_output>/<video_id>/clip_XXXX/{ground_0deg,elevated_45deg,elevated_110deg}.png

    For each clip, loads all requested views as a tensor of shape (num_views, C, H, W).
    Unique-id format: "{video_id}_{clip_idx}" — matches AerialGalleryDataset exactly.
    """

    def __init__(self, vggt_output_dir, views, transform=None):
        self.root = Path(vggt_output_dir)
        self.views = views          # list of view keys, e.g. ['elevated_110']
        self.transform = transform
        self.samples = []
        self._scan()

    def _scan(self):
        missing_view_count = 0
        for video_dir in sorted(self.root.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            for clip_dir in sorted(video_dir.iterdir()):
                if not clip_dir.is_dir() or not clip_dir.name.startswith('clip_'):
                    continue
                clip_idx = int(clip_dir.name.split('_')[1])
                view_paths = {}
                all_exist = True
                for v in self.views:
                    p = clip_dir / VIEW_FILENAMES[v]
                    if not p.exists():
                        all_exist = False
                        missing_view_count += 1
                        break
                    view_paths[v] = str(p)
                if all_exist:
                    self.samples.append({
                        'video_id':  video_id,
                        'clip_idx':  clip_idx,
                        'view_paths': view_paths,
                        'unique_id': f"{video_id}_{clip_idx}",
                    })

        if missing_view_count:
            print(f"  [warn] Skipped {missing_view_count} clip(s) with missing view images.")

        self.video_clips = defaultdict(list)
        for i, s in enumerate(self.samples):
            self.video_clips[s['video_id']].append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = []
        for v in self.views:
            img = Image.open(sample['view_paths'][v]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        # shape: (num_views, C, H, W)
        return {
            'images':    torch.stack(images, dim=0),
            'video_id':  sample['video_id'],
            'clip_idx':  sample['clip_idx'],
            'unique_id': sample['unique_id'],
        }


# ─────────────────────────────────────────────
# Dataset: aerial gallery  (same as evaluate.py)
# ─────────────────────────────────────────────
class AerialGalleryDataset(Dataset):
    def __init__(self, aerial_root, valid_pairs=None, transform=None):
        self.aerial_root = Path(aerial_root)
        self.transform = transform
        self.valid_pairs = valid_pairs
        self.samples = []
        self._scan_directory()

    def _scan_directory(self):
        for video_dir in sorted(self.aerial_root.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            image_files = sorted(
                [f for f in video_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            )
            for img_file in image_files:
                frame_idx = int(img_file.stem)
                if self.valid_pairs is not None:
                    if (video_id, frame_idx) not in self.valid_pairs:
                        continue
                self.samples.append({
                    'video_id':   video_id,
                    'frame_idx':  frame_idx,
                    'image_path': str(img_file),
                    'unique_id':  f"{video_id}_{frame_idx}",
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image':     image,
            'video_id':  sample['video_id'],
            'frame_idx': sample['frame_idx'],
            'unique_id': sample['unique_id'],
        }


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
def get_transforms(input_size=224):
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────
@torch.no_grad()
def extract_query_embeddings(model, dataloader, device):
    """
    For each clip, encode all views independently and mean-pool the embeddings.
    Result is L2-normalised.
    """
    model.eval()
    embeddings, video_ids, clip_indices, unique_ids = [], [], [], []

    for batch in tqdm(dataloader, desc='Extracting query (elevated) embeddings'):
        imgs = batch['images'].to(device)          # (B, V, C, H, W)
        B, V, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * V, C, H, W)      # (B*V, C, H, W)
        emb_flat = model(imgs_flat)                 # (B*V, D)  — already L2-normed
        emb = emb_flat.view(B, V, -1).mean(dim=1)  # (B, D)    — mean of per-view embeddings
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        embeddings.append(emb.cpu())

        for i in range(B):
            video_ids.append(batch['video_id'][i])
            cidx = batch['clip_idx'][i]
            cidx = cidx.item() if torch.is_tensor(cidx) else cidx
            clip_indices.append(cidx)
            unique_ids.append(batch['unique_id'][i])

    return torch.cat(embeddings, dim=0), video_ids, clip_indices, unique_ids


@torch.no_grad()
def extract_aerial_embeddings(model, dataloader, device):
    model.eval()
    embeddings, video_ids, frame_indices, unique_ids = [], [], [], []

    for batch in tqdm(dataloader, desc='Extracting gallery (aerial) embeddings'):
        x = batch['image'].to(device)
        emb = model(x)
        embeddings.append(emb.cpu())

        for i in range(len(batch['video_id'])):
            video_ids.append(batch['video_id'][i])
            fidx = batch['frame_idx'][i]
            fidx = fidx.item() if torch.is_tensor(fidx) else fidx
            frame_indices.append(fidx)
            unique_ids.append(batch['unique_id'][i])

    return torch.cat(embeddings, dim=0), video_ids, frame_indices, unique_ids


# ─────────────────────────────────────────────
# Metrics  (identical to evaluate.py)
# ─────────────────────────────────────────────
def compute_recall_metrics(query_emb, gallery_emb, query_ids, gallery_ids):
    similarity = torch.matmul(query_emb, gallery_emb.T)
    num_queries = query_emb.size(0)
    num_gallery = gallery_emb.size(0)

    sorted_indices = torch.argsort(similarity, dim=1, descending=True)

    results = {}
    k_values = [1, 5, 10]

    for k in k_values:
        correct = 0
        for i in range(num_queries):
            gt_id = query_ids[i]
            top_k_ids = [gallery_ids[j] for j in sorted_indices[i, :k].tolist()]
            if gt_id in top_k_ids:
                correct += 1
        results[f'R@{k}'] = correct / num_queries * 100

    k_1_percent = max(1, int(num_gallery * 0.01))
    correct = 0
    for i in range(num_queries):
        gt_id = query_ids[i]
        top_k_ids = [gallery_ids[j] for j in sorted_indices[i, :k_1_percent].tolist()]
        if gt_id in top_k_ids:
            correct += 1
    results['R@1%'] = correct / num_queries * 100

    return results


# ─────────────────────────────────────────────
# Dataset info summary
# ─────────────────────────────────────────────
def print_dataset_info(query_dataset, aerial_dataset):
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")

    q_videos = defaultdict(list)
    for s in query_dataset.samples:
        q_videos[s['video_id']].append(s['clip_idx'])

    a_videos = defaultdict(list)
    for s in aerial_dataset.samples:
        a_videos[s['video_id']].append(s['frame_idx'])

    print(f"\n[Query Dataset  — views: {query_dataset.views}]")
    print(f"  Total clips : {len(query_dataset)}")
    print(f"  Total videos: {len(q_videos)}")
    for vid in sorted(q_videos.keys()):
        clips = sorted(q_videos[vid])
        print(f"    - {vid}: {len(clips)} clips (idx {min(clips)}-{max(clips)})")

    print(f"\n[Aerial Gallery Dataset]")
    print(f"  Total images: {len(aerial_dataset)}")
    print(f"  Total videos: {len(a_videos)}")
    for vid in sorted(a_videos.keys()):
        frames = sorted(a_videos[vid])
        print(f"    - {vid}: {len(frames)} images (idx {min(frames)}-{max(frames)})")

    common = set(q_videos) & set(a_videos)
    q_only = set(q_videos) - set(a_videos)
    a_only = set(a_videos) - set(q_videos)

    print(f"\n[Alignment]")
    print(f"  Common videos : {len(common)}")
    if q_only:
        print(f"  Query-only    : {sorted(q_only)}")
    if a_only:
        print(f"  Aerial-only   : {sorted(a_only)}")

    matched = sum(
        1 for s in query_dataset.samples
        if any(a['video_id'] == s['video_id'] and a['frame_idx'] == s['clip_idx']
               for a in aerial_dataset.samples)
    )
    print(f"\n[Pair Matching]")
    print(f"  Matched query-aerial pairs: {matched}/{len(query_dataset)}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Views  : {args.views}")

    views = [v.strip() for v in args.views.split(',')]
    for v in views:
        if v not in VIEW_FILENAMES:
            raise ValueError(f"Unknown view '{v}'. Choose from: {list(VIEW_FILENAMES.keys())}")

    transform = get_transforms(args.input_size)

    # ── Query dataset ──────────────────────────────────────────────
    query_dataset = ElevatedViewDataset(args.vggt_output_dir, views, transform)
    if len(query_dataset) == 0:
        raise RuntimeError(
            f"No valid clips found under '{args.vggt_output_dir}'. "
            "Check that the path is correct and that view images exist."
        )

    # ── Aerial gallery (restrict to clips present in query set) ────
    valid_pairs = {(s['video_id'], s['clip_idx']) for s in query_dataset.samples}
    aerial_dataset = AerialGalleryDataset(args.aerial_dir, valid_pairs=valid_pairs, transform=transform)

    print_dataset_info(query_dataset, aerial_dataset)

    query_loader = DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    aerial_loader = DataLoader(
        aerial_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Models ────────────────────────────────────────────────────
    query_encoder  = ImageEncoder(args.backbone, args.embed_dim).to(device)
    aerial_encoder = ImageEncoder(args.backbone, args.embed_dim).to(device)

    # ── Embeddings ────────────────────────────────────────────────
    query_emb,  q_vids, q_clips, q_ids = extract_query_embeddings(query_encoder,  query_loader,  device)
    aerial_emb, a_vids, a_frames, a_ids = extract_aerial_embeddings(aerial_encoder, aerial_loader, device)

    print(f"\nQuery  embeddings : {query_emb.shape}")
    print(f"Gallery embeddings: {aerial_emb.shape}")

    # ── Metrics ───────────────────────────────────────────────────
    results = compute_recall_metrics(query_emb, aerial_emb, q_ids, a_ids)
    results['num_queries']  = len(q_ids)
    results['num_gallery']  = len(a_ids)
    results['views']        = views

    view_label = '+'.join(views)
    print(f"\n{'='*60}")
    print(f"Elevated-View Evaluation Results  [{view_label}]")
    print(f"{'='*60}")
    print(f"\nRecall  (N={results['num_queries']} queries, Gallery={results['num_gallery']}):")
    print(f"  R@1  : {results['R@1']:.2f}%")
    print(f"  R@5  : {results['R@5']:.2f}%")
    print(f"  R@10 : {results['R@10']:.2f}%")
    print(f"  R@1% : {results['R@1%']:.2f}%")
    print(f"{'='*60}\n")

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate geolocalization using elevated views from vggt-output')
    parser.add_argument('--vggt_output_dir', type=str, required=True,
                        help='Root vggt-output directory (contains <video_id>/clip_XXXX/ sub-dirs)')
    parser.add_argument('--aerial_dir', type=str, required=True,
                        help='Aerial gallery directory (same format as evaluate.py)')
    parser.add_argument('--views', type=str, default='elevated_110',
                        help='Comma-separated list of views to use: ground, elevated_45, elevated_110 '
                             '(default: elevated_110). Multiple views are mean-pooled.')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    main(args)
