"""
Zero-shot geolocalization using elevated_110deg.png from VGGT output.
Single-view ResNet18 + linear projection → L2-normed embedding,
matched against the aerial gallery via cosine similarity.

Structure expected:
  <vggt_output>/<video_id>/clip_XXXX/elevated_110deg.png
  <aerial_dir>/<video_id>/<clip_idx>.{png|jpg|jpeg}
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════

class SingleViewEncoder(nn.Module):
    def __init__(self, backbone='resnet18', embed_dim=512):
        super().__init__()
        if backbone == 'resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == 'resnet34':
            net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        in_features = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        emb = self.projection(features)
        return nn.functional.normalize(emb, p=2, dim=-1)


# ══════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════

class Elevated110Dataset(Dataset):
    """Reads elevated_110deg.png from vggt-output clips."""

    def __init__(self, vggt_output_dir, transform=None):
        self.transform = transform
        self.samples = []
        self._scan(Path(vggt_output_dir))

    def _scan(self, root):
        for video_dir in sorted(root.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            clip_dirs = sorted(
                [d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('clip_')]
            )
            for clip_dir in clip_dirs:
                img_path = clip_dir / 'elevated_110deg.png'
                if img_path.exists():
                    clip_idx = int(clip_dir.name.split('_')[1])
                    self.samples.append({
                        'video_id': video_id,
                        'clip_idx': clip_idx,
                        'image_path': str(img_path),
                        'unique_id': f"{video_id}_{clip_idx}",
                    })
        print(f"Found {len(self.samples)} elevated_110deg clips.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'video_id': s['video_id'],
                'clip_idx': s['clip_idx'], 'unique_id': s['unique_id']}


class AerialGalleryDataset(Dataset):
    def __init__(self, aerial_root, valid_pairs=None, transform=None):
        self.transform = transform
        self.samples = []
        self._scan(Path(aerial_root), valid_pairs)

    def _scan(self, root, valid_pairs):
        for video_dir in sorted(root.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            for img_file in sorted(video_dir.iterdir()):
                if img_file.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                    continue
                frame_idx = int(img_file.stem)
                if valid_pairs is not None and (video_id, frame_idx) not in valid_pairs:
                    continue
                self.samples.append({
                    'video_id': video_id,
                    'frame_idx': frame_idx,
                    'image_path': str(img_file),
                    'unique_id': f"{video_id}_{frame_idx}",
                })
        print(f"Found {len(self.samples)} aerial gallery images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'video_id': s['video_id'],
                'frame_idx': s['frame_idx'], 'unique_id': s['unique_id']}


# ══════════════════════════════════════════════════════════════════
# Embedding extraction
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, dataloader, id_key, device):
    model.eval()
    embeddings, unique_ids = [], []
    for batch in tqdm(dataloader, desc='Extracting embeddings'):
        emb = model(batch['image'].to(device))
        embeddings.append(emb.cpu())
        unique_ids.extend(batch['unique_id'])
    return torch.cat(embeddings, dim=0), unique_ids


# ══════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════

def compute_recall(query_emb, gallery_emb, query_ids, gallery_ids):
    sim = torch.matmul(query_emb, gallery_emb.T)
    sorted_idx = torch.argsort(sim, dim=1, descending=True)
    N = query_emb.size(0)
    G = gallery_emb.size(0)

    results = {}
    for k in [1, 5, 10]:
        correct = sum(
            query_ids[i] in [gallery_ids[j] for j in sorted_idx[i, :k].tolist()]
            for i in range(N)
            if query_ids[i] in gallery_ids
        )
        results[f'R@{k}'] = correct / N * 100

    k1pct = max(1, int(G * 0.01))
    correct = sum(
        query_ids[i] in [gallery_ids[j] for j in sorted_idx[i, :k1pct].tolist()]
        for i in range(N)
        if query_ids[i] in gallery_ids
    )
    results['R@1%'] = correct / N * 100
    results['num_queries'] = N
    results['num_gallery'] = G
    return results


def print_results(results):
    print(f"\n{'='*60}")
    print("Elevated-110deg Zero-Shot Evaluation Results")
    print(f"{'='*60}")
    print(f"  Queries : {results['num_queries']}")
    print(f"  Gallery : {results['num_gallery']}")
    print(f"  R@1  : {results['R@1']:.2f}%")
    print(f"  R@5  : {results['R@5']:.2f}%")
    print(f"  R@10 : {results['R@10']:.2f}%")
    print(f"  R@1% : {results['R@1%']:.2f}%")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    query_ds = Elevated110Dataset(args.vggt_output_dir, transform)
    valid_pairs = {(s['video_id'], s['clip_idx']) for s in query_ds.samples}
    gallery_ds = AerialGalleryDataset(args.aerial_dir, valid_pairs=valid_pairs, transform=transform)

    query_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    query_enc = SingleViewEncoder(args.backbone, args.embed_dim).to(device)
    gallery_enc = SingleViewEncoder(args.backbone, args.embed_dim).to(device)

    query_emb, query_ids = extract_embeddings(query_enc, query_loader, 'unique_id', device)
    gallery_emb, gallery_ids = extract_embeddings(gallery_enc, gallery_loader, 'unique_id', device)

    print(f"Query embeddings  : {query_emb.shape}")
    print(f"Gallery embeddings: {gallery_emb.shape}")

    results = compute_recall(query_emb, gallery_emb, query_ids, gallery_ids)
    print_results(results)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--vggt_output_dir', required=True,
                   help='Root of vggt-output (contains <video_id>/clip_XXXX/)')
    p.add_argument('--aerial_dir', required=True,
                   help='Root of aerial gallery (contains <video_id>/<frame_idx>.png)')
    p.add_argument('--backbone',    default='resnet18', choices=['resnet18', 'resnet34'])
    p.add_argument('--embed_dim',   type=int, default=512)
    p.add_argument('--input_size',  type=int, default=224)
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--output_file', default='results_110deg.json')
    main(p.parse_args())
