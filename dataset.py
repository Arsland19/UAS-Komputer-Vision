"""
Dataset loaders for:
  1. Kvasir-SEG  (images/ + masks/)
  2. Archive dataset  (PNG/Original/ + PNG/Ground Truth/)

Expected folder layout on your machine (D:\S2\CV\Final\fixcode):
  fixcode/
    kvasir-seg/
      Kvasir-SEG/
        images/   ← .jpg polyp images
        masks/    ← .jpg binary masks
    archive/
      PNG/
        Original/       ← .png images
        Ground Truth/   ← .png masks
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────
#  Shared augmentation (image + mask sync)
# ─────────────────────────────────────────
class JointTransform:
    def __init__(self, size=352, augment=True):
        self.size    = size
        self.augment = augment

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Resize
        image = TF.resize(image, (self.size, self.size), interpolation=Image.BILINEAR)
        mask  = TF.resize(mask,  (self.size, self.size), interpolation=Image.NEAREST)

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # Random rotation ±30°
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

            # Color jitter on image only
            jitter = T.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)
            image = jitter(image)

        # To tensor
        img_t  = TF.to_tensor(image)                     # [3,H,W] float32 in [0,1]
        img_t  = TF.normalize(img_t,
                               mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

        mask_t = TF.to_tensor(mask)                       # [1,H,W] float32
        mask_t = (mask_t > 0.5).float()                   # binarise

        return img_t, mask_t


# ─────────────────────────────────────────
#  1. Kvasir-SEG Dataset
# ─────────────────────────────────────────
class KvasirSEGDataset(Dataset):
    """
    root: path to Kvasir-SEG/ folder
          (should contain images/ and masks/ sub-folders)
    """

    def __init__(self, root: str, split='train', val_ratio=0.1,
                 size=352, augment=True, seed=42):
        super().__init__()
        root = Path(root)
        img_dir  = root / 'images'
        mask_dir = root / 'masks'

        all_imgs = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        random.seed(seed)
        random.shuffle(all_imgs)

        n_val = max(1, int(len(all_imgs) * val_ratio))
        if split == 'val':
            self.img_paths  = all_imgs[:n_val]
        else:
            self.img_paths  = all_imgs[n_val:]

        self.mask_dir  = mask_dir
        self.transform = JointTransform(size=size, augment=(augment and split == 'train'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        # Kvasir masks use same stem, may be .jpg or .png
        stem = img_path.stem
        for ext in ('.jpg', '.png'):
            mask_path = self.mask_dir / (stem + ext)
            if mask_path.exists():
                break

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        img_t, mask_t = self.transform(image, mask)
        return img_t, mask_t, str(img_path)


# ─────────────────────────────────────────
#  2. Archive (PNG) Dataset
# ─────────────────────────────────────────
class ArchivePNGDataset(Dataset):
    """
    root: path to archive/PNG/ folder
          (should contain Original/ and Ground Truth/ sub-folders)
    """

    def __init__(self, root: str, split='train', val_ratio=0.1,
                 size=352, augment=True, seed=42):
        super().__init__()
        root = Path(root)
        img_dir  = root / 'Original'
        mask_dir = root / 'Ground Truth'

        all_imgs = sorted(img_dir.glob('*.png'))
        random.seed(seed)
        random.shuffle(all_imgs)

        n_val = max(1, int(len(all_imgs) * val_ratio))
        if split == 'val':
            self.img_paths  = all_imgs[:n_val]
        else:
            self.img_paths  = all_imgs[n_val:]

        self.mask_dir  = mask_dir
        self.transform = JointTransform(size=size, augment=(augment and split == 'train'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = self.mask_dir / img_path.name

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        img_t, mask_t = self.transform(image, mask)
        return img_t, mask_t, str(img_path)


# ─────────────────────────────────────────
#  3. Factory – build dataloaders
# ─────────────────────────────────────────
def build_dataloaders(
    kvasir_root: str,
    archive_root: str,
    batch_size:   int  = 8,
    num_workers:  int  = 4,
    img_size:     int  = 352,
    val_ratio:    float = 0.1,
):
    """
    kvasir_root : path ending in ...Kvasir-SEG/
    archive_root: path ending in .../archive/PNG/
    Returns train_loader, val_loader
    """
    kvasir_root  = Path(kvasir_root)
    archive_root = Path(archive_root)

    datasets_train, datasets_val = [], []

    if kvasir_root.exists():
        datasets_train.append(KvasirSEGDataset(kvasir_root, 'train', val_ratio, img_size))
        datasets_val.append(  KvasirSEGDataset(kvasir_root, 'val',   val_ratio, img_size, augment=False))
        print(f"[Kvasir-SEG] train={len(datasets_train[-1])}, val={len(datasets_val[-1])}")
    else:
        print(f"[WARN] Kvasir-SEG not found at {kvasir_root}")

    if archive_root.exists():
        datasets_train.append(ArchivePNGDataset(archive_root, 'train', val_ratio, img_size))
        datasets_val.append(  ArchivePNGDataset(archive_root, 'val',   val_ratio, img_size, augment=False))
        print(f"[Archive PNG] train={len(datasets_train[-1])}, val={len(datasets_val[-1])}")
    else:
        print(f"[WARN] Archive PNG not found at {archive_root}")

    assert len(datasets_train) > 0, "No dataset found! Check your paths."

    train_set = ConcatDataset(datasets_train) if len(datasets_train) > 1 else datasets_train[0]
    val_set   = ConcatDataset(datasets_val)   if len(datasets_val)   > 1 else datasets_val[0]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    # Quick smoke test — update paths to your machine
    BASE = r'D:\S2\CV\Final\fixcode'
    tl, vl = build_dataloaders(
        kvasir_root  = BASE + r'\kvasir-seg\Kvasir-SEG',
        archive_root = BASE + r'\archive\PNG',
        batch_size=4, num_workers=0,
    )
    imgs, masks, paths = next(iter(tl))
    print(f"Train batch — imgs: {imgs.shape}, masks: {masks.shape}")
