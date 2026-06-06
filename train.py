"""
Train Enhanced PraNet on Kvasir-SEG + Archive dataset.

Usage (from project root):
    python train.py

Edit the CONFIG section below to match your machine paths.
"""

import os
import time
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

from model   import EnhancedPraNet
from loss    import EnhancedPraNetLoss
from dataset import build_dataloaders
from metrics import MetricTracker


# ═══════════════════════════════════════════
#  ► CONFIG — edit these paths
# ═══════════════════════════════════════════
BASE_DIR    = r'D:\S2\CV\Final\fixcode'

KVASIR_ROOT  = BASE_DIR + r'\kvasir-seg\Kvasir-SEG'
ARCHIVE_ROOT = BASE_DIR + r'\archive\PNG'
SAVE_DIR     = BASE_DIR + r'\checkpoints'

IMG_SIZE    = 352
BATCH_SIZE  = 8
NUM_WORKERS = 4         # set 0 on Windows if multiprocessing issues
EPOCHS      = 50
LR          = 1e-4
WEIGHT_DECAY= 1e-4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ═══════════════════════════════════════════


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    tracker = MetricTracker()

    for step, (imgs, masks, _) in enumerate(loader):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with autocast('cuda', enabled=(device == 'cuda')):
            outputs = model(imgs)
            loss, breakdown = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        # use final prediction (pred2) for metrics
        tracker.update(loss.item(), outputs[3], masks)

        if (step + 1) % 20 == 0:
            s = tracker.summary()
            print(f"  [Epoch {epoch} | step {step+1}/{len(loader)}] "
                  f"loss={s['loss']:.4f}  dice={s['dice']:.4f}  iou={s['iou']:.4f}")

    return tracker.summary()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    tracker = MetricTracker()

    for imgs, masks, _ in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        with autocast('cuda', enabled=(device == 'cuda')):
            outputs = model(imgs)
            loss, _ = criterion(outputs, masks)
        tracker.update(loss.item(), outputs[3], masks)

    return tracker.summary()


def main():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Kvasir : {KVASIR_ROOT}")
    print(f"Archive: {ARCHIVE_ROOT}")

    train_loader, val_loader = build_dataloaders(
        kvasir_root  = KVASIR_ROOT,
        archive_root = ARCHIVE_ROOT,
        batch_size   = BATCH_SIZE,
        num_workers  = NUM_WORKERS,
        img_size     = IMG_SIZE,
    )

    model     = EnhancedPraNet(pretrained=True).to(DEVICE)
    criterion = EnhancedPraNetLoss(edge_weight=0.5, lambda_edge=0.4)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    best_dice  = 0.0
    log_path   = Path(SAVE_DIR) / 'training_log.csv'

    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,train_dice,train_iou,val_loss,val_dice,val_iou\n')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, criterion,
                                   optimizer, scaler, DEVICE, epoch)
        val_m   = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"\n[Epoch {epoch}/{EPOCHS}]  {elapsed:.0f}s  "
              f"Train — loss={train_m['loss']:.4f} dice={train_m['dice']:.4f} iou={train_m['iou']:.4f}  |  "
              f"Val — loss={val_m['loss']:.4f} dice={val_m['dice']:.4f} iou={val_m['iou']:.4f}\n")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_m['loss']:.6f},{train_m['dice']:.6f},{train_m['iou']:.6f},"
                    f"{val_m['loss']:.6f},{val_m['dice']:.6f},{val_m['iou']:.6f}\n")

        # Save best
        if val_m['dice'] > best_dice:
            best_dice = val_m['dice']
            ckpt_path = Path(SAVE_DIR) / 'best_model.pth'
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'best_dice': best_dice}, ckpt_path)
            print(f"  ✓ Saved best model → {ckpt_path}  (dice={best_dice:.4f})")

        # Save latest
        torch.save(model.state_dict(), Path(SAVE_DIR) / 'latest_model.pth')

    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
