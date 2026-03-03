import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import r3d_18, R3D_18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class BEV2DEncoder(nn.Module):
    def __init__(self, backbone='resnet18', embed_dim=512):
        super().__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        return nn.functional.normalize(embeddings, p=2, dim=-1)


class BEV3DEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        return nn.functional.normalize(embeddings, p=2, dim=-1)


class AerialEncoder(nn.Module):
    def __init__(self, backbone='resnet18', embed_dim=512):
        super().__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        return nn.functional.normalize(embeddings, p=2, dim=-1)


class BEVClipDataset(Dataset):
    def __init__(self, bev_root, transform=None, variant='2d', temporal_window=8):
        self.bev_root = Path(bev_root)
        self.transform = transform
        self.variant = variant
        self.temporal_window = temporal_window
        self.samples = []
        self._scan_directory()

    def _scan_directory(self):
        for video_dir in sorted(self.bev_root.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            clip_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('clip_')])

            for clip_dir in clip_dirs:
                clip_idx = int(clip_dir.name.split('_')[1])
                original_dir = clip_dir / 'original'

                if original_dir.exists():
                    bev_image = original_dir / 'point_cloud_bev.png'
                    if bev_image.exists():
                        self.samples.append({
                            'video_id': video_id,
                            'clip_idx': clip_idx,
                            'bev_path': str(bev_image),
                            'clip_dir': str(clip_dir)
                        })

        self.video_clips = defaultdict(list)
        for i, s in enumerate(self.samples):
            self.video_clips[s['video_id']].append(i)

    def __len__(self):
        return len(self.samples)

    def _get_temporal_sequence(self, idx):
        sample = self.samples[idx]
        video_id = sample['video_id']
        clip_idx = sample['clip_idx']

        video_indices = sorted(self.video_clips[video_id], key=lambda i: self.samples[i]['clip_idx'])
        center_pos = next((i for i, vi in enumerate(video_indices) if self.samples[vi]['clip_idx'] == clip_idx), 0)

        half_window = self.temporal_window // 2
        start_pos = max(0, center_pos - half_window)
        end_pos = min(len(video_indices), start_pos + self.temporal_window)
        start_pos = max(0, end_pos - self.temporal_window)

        sequence_indices = video_indices[start_pos:end_pos]

        while len(sequence_indices) < self.temporal_window:
            sequence_indices.append(sequence_indices[-1])

        return sequence_indices[:self.temporal_window]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.variant == '2d':
            image = Image.open(sample['bev_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {
                'image': image,
                'video_id': sample['video_id'],
                'clip_idx': sample['clip_idx']
            }
        else:
            sequence_indices = self._get_temporal_sequence(idx)
            frames = []
            for si in sequence_indices:
                img = Image.open(self.samples[si]['bev_path']).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            video_tensor = torch.stack(frames, dim=1)
            return {
                'video': video_tensor,
                'video_id': sample['video_id'],
                'clip_idx': sample['clip_idx']
            }


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
            image_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

            for img_file in image_files:
                frame_idx = int(img_file.stem)

                if self.valid_pairs is not None:
                    if (video_id, frame_idx) not in self.valid_pairs:
                        continue

                self.samples.append({
                    'video_id': video_id,
                    'frame_idx': frame_idx,
                    'image_path': str(img_file),
                    'unique_id': f"{video_id}_{frame_idx}"
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'video_id': sample['video_id'],
            'frame_idx': sample['frame_idx'],
            'unique_id': sample['unique_id']
        }


def get_transforms(input_size=224):
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


@torch.no_grad()
def extract_bev_embeddings(model, dataloader, device, variant='2d'):
    model.eval()
    embeddings, video_ids, clip_indices, unique_ids = [], [], [], []

    for batch in tqdm(dataloader, desc='Extracting BEV embeddings'):
        if variant == '2d':
            x = batch['image'].to(device)
        else:
            x = batch['video'].to(device)
        emb = model(x)
        embeddings.append(emb.cpu())

        for i in range(len(batch['video_id'])):
            vid = batch['video_id'][i]
            cidx = batch['clip_idx'][i].item() if torch.is_tensor(batch['clip_idx'][i]) else batch['clip_idx'][i]
            video_ids.append(vid)
            clip_indices.append(cidx)
            unique_ids.append(f"{vid}_{cidx}")

    return torch.cat(embeddings, dim=0), video_ids, clip_indices, unique_ids


@torch.no_grad()
def extract_aerial_embeddings(model, dataloader, device):
    model.eval()
    embeddings, video_ids, frame_indices, unique_ids = [], [], [], []

    for batch in tqdm(dataloader, desc='Extracting aerial embeddings'):
        x = batch['image'].to(device)
        emb = model(x)
        embeddings.append(emb.cpu())

        for i in range(len(batch['video_id'])):
            video_ids.append(batch['video_id'][i])
            fidx = batch['frame_idx'][i].item() if torch.is_tensor(batch['frame_idx'][i]) else batch['frame_idx'][i]
            frame_indices.append(fidx)
            unique_ids.append(batch['unique_id'][i])

    return torch.cat(embeddings, dim=0), video_ids, frame_indices, unique_ids


def compute_recall_metrics(query_emb, gallery_emb, query_ids, gallery_ids):
    similarity = torch.matmul(query_emb, gallery_emb.T)
    num_queries = query_emb.size(0)
    num_gallery = gallery_emb.size(0)

    sorted_indices = torch.argsort(similarity, dim=1, descending=True)

    gallery_id_to_idx = {gid: i for i, gid in enumerate(gallery_ids)}

    results = {}
    k_values = [1, 5, 10]

    for k in k_values:
        correct = 0
        for i in range(num_queries):
            gt_id = query_ids[i]
            if gt_id not in gallery_id_to_idx:
                continue
            top_k_indices = sorted_indices[i, :k].tolist()
            top_k_ids = [gallery_ids[j] for j in top_k_indices]
            if gt_id in top_k_ids:
                correct += 1
        results[f'R@{k}'] = correct / num_queries * 100

    k_1_percent = max(1, int(num_gallery * 0.01))
    correct = 0
    for i in range(num_queries):
        gt_id = query_ids[i]
        if gt_id not in gallery_id_to_idx:
            continue
        top_k_indices = sorted_indices[i, :k_1_percent].tolist()
        top_k_ids = [gallery_ids[j] for j in top_k_indices]
        if gt_id in top_k_ids:
            correct += 1
    results['R@1%'] = correct / num_queries * 100

    return results


def print_results(results, variant):
    print(f"\n{'='*60}")
    print(f"BEV-{variant.upper()} Evaluation Results (Exact-Match Retrieval)")
    print(f"{'='*60}")
    print(f"\nClip-Level Recall (N={results.get('num_queries', 'N/A')} queries, Gallery={results.get('num_gallery', 'N/A')}):")
    print(f"  R@1:  {results['R@1']:.2f}%")
    print(f"  R@5:  {results['R@5']:.2f}%")
    print(f"  R@10: {results['R@10']:.2f}%")
    print(f"  R@1%: {results['R@1%']:.2f}%")
    print(f"{'='*60}\n")


def print_dataset_info(bev_dataset, aerial_dataset, valid_pairs):
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    
    bev_videos = defaultdict(list)
    for s in bev_dataset.samples:
        bev_videos[s['video_id']].append(s['clip_idx'])
    
    aerial_videos = defaultdict(list)
    for s in aerial_dataset.samples:
        aerial_videos[s['video_id']].append(s['frame_idx'])
    
    print(f"\n[BEV Dataset]")
    print(f"  Total clips: {len(bev_dataset)}")
    print(f"  Total videos: {len(bev_videos)}")
    print(f"  Video IDs:")
    for vid in sorted(bev_videos.keys()):
        clips = sorted(bev_videos[vid])
        print(f"    - {vid}: {len(clips)} clips (indices {min(clips)}-{max(clips)})")
    
    print(f"\n[Aerial Gallery Dataset]")
    print(f"  Total images: {len(aerial_dataset)}")
    print(f"  Total videos: {len(aerial_videos)}")
    print(f"  Video IDs:")
    for vid in sorted(aerial_videos.keys()):
        frames = sorted(aerial_videos[vid])
        print(f"    - {vid}: {len(frames)} images (indices {min(frames)}-{max(frames)})")
    
    common_videos = set(bev_videos.keys()) & set(aerial_videos.keys())
    bev_only = set(bev_videos.keys()) - set(aerial_videos.keys())
    aerial_only = set(aerial_videos.keys()) - set(bev_videos.keys())
    
    print(f"\n[Alignment Check]")
    print(f"  Common videos: {len(common_videos)}")
    if common_videos:
        print(f"    {sorted(common_videos)}")
    if bev_only:
        print(f"  BEV only (no aerial match): {sorted(bev_only)}")
    if aerial_only:
        print(f"  Aerial only (no BEV match): {sorted(aerial_only)}")
    
    print(f"\n[Pair Matching]")
    matched_pairs = 0
    unmatched_bev = []
    for s in bev_dataset.samples:
        pair = (s['video_id'], s['clip_idx'])
        aerial_match = any(a['video_id'] == s['video_id'] and a['frame_idx'] == s['clip_idx'] for a in aerial_dataset.samples)
        if aerial_match:
            matched_pairs += 1
        else:
            unmatched_bev.append(pair)
    
    print(f"  Matched BEV-Aerial pairs: {matched_pairs}/{len(bev_dataset.samples)}")
    if unmatched_bev and len(unmatched_bev) <= 10:
        print(f"  Unmatched BEV clips: {unmatched_bev}")
    elif unmatched_bev:
        print(f"  Unmatched BEV clips: {len(unmatched_bev)} (showing first 10: {unmatched_bev[:10]})")
    
    print(f"{'='*60}\n")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Variant: BEV-{args.variant.upper()}")

    transform = get_transforms(args.input_size)

    bev_dataset = BEVClipDataset(args.bev_dir, transform, args.variant, args.temporal_window)

    valid_pairs = set()
    for sample in bev_dataset.samples:
        valid_pairs.add((sample['video_id'], sample['clip_idx']))

    aerial_dataset = AerialGalleryDataset(args.aerial_dir, valid_pairs=valid_pairs, transform=transform)
    
    print_dataset_info(bev_dataset, aerial_dataset, valid_pairs)

    bev_loader = DataLoader(bev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    aerial_loader = DataLoader(aerial_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.variant == '2d':
        bev_encoder = BEV2DEncoder(args.backbone, args.embed_dim).to(device)
    else:
        bev_encoder = BEV3DEncoder(args.embed_dim).to(device)
    aerial_encoder = AerialEncoder(args.backbone, args.embed_dim).to(device)

    bev_emb, bev_vids, bev_clips, bev_unique_ids = extract_bev_embeddings(bev_encoder, bev_loader, device, args.variant)
    aerial_emb, aerial_vids, aerial_frames, aerial_unique_ids = extract_aerial_embeddings(aerial_encoder, aerial_loader, device)

    print(f"\nQuery embeddings: {bev_emb.shape}")
    print(f"Gallery embeddings: {aerial_emb.shape}")

    results = compute_recall_metrics(bev_emb, aerial_emb, bev_unique_ids, aerial_unique_ids)
    results['num_queries'] = len(bev_unique_ids)
    results['num_gallery'] = len(aerial_unique_ids)

    print_results(results, args.variant)

    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bev_dir', type=str, required=True)
    parser.add_argument('--aerial_dir', type=str, required=True)
    parser.add_argument('--variant', type=str, default='2d', choices=['2d', '3d'])
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--temporal_window', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    main(args)