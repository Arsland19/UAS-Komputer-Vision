# Enhanced PraNet — Polyp Segmentation

**Architecture:** Parallel Reverse Attention Network (PraNet)
**Modifications:**
- CBAM (Convolutional Block Attention Module) on each encoder level
- FFT Frequency Enhancement on the deepest encoder feature
- Edge-aware Boundary Loss

---

## Folder structure expected on your machine

```
D:\S2\CV\Final\fixcode\
├── kvasir-seg\
│   └── Kvasir-SEG\
│       ├── images\   ← .jpg polyp images
│       └── masks\    ← .jpg binary masks
├── archive\
│   └── PNG\
│       ├── Original\       ← .png images
│       └── Ground Truth\   ← .png masks
└── checkpoints\            ← created automatically
```

> **Tip:** Extract `kvasir-seg.zip` → you get `Kvasir-SEG/images` and `Kvasir-SEG/masks`.
> Extract `archive.zip` → you get `PNG/Original` and `PNG/Ground Truth`.

---

## Project files

| File | Description |
|------|-------------|
| `model.py`   | Full architecture: PraNet + CBAM + FFT |
| `loss.py`    | BCE-IoU loss + Edge-aware Boundary Loss |
| `dataset.py` | Kvasir-SEG & Archive dataloaders |
| `metrics.py` | Dice, IoU trackers |
| `train.py`   | Training loop with AMP, cosine LR |
| `infer.py`   | Inference + overlay visualisation |

---

## Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows

# 2. Install PyTorch with CUDA (adjust cuda version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install other dependencies
pip install -r requirements.txt
```

---

## Training

Open `train.py` and edit the **CONFIG** block at the top:

```python
BASE_DIR    = r'D:\S2\CV\Final\fixcode'
KVASIR_ROOT  = BASE_DIR + r'\kvasir-seg\Kvasir-SEG'
ARCHIVE_ROOT = BASE_DIR + r'\archive\PNG'
SAVE_DIR     = BASE_DIR + r'\checkpoints'

BATCH_SIZE  = 8      # reduce to 4 if GPU OOM
EPOCHS      = 50
LR          = 1e-4
```

Then run:

```bash
python train.py
```

Training logs are saved to `checkpoints/training_log.csv`.
Best model (by val Dice) → `checkpoints/best_model.pth`.

---

## Inference

```bash
python infer.py \
  --checkpoint D:\S2\CV\Final\fixcode\checkpoints\best_model.pth \
  --img_dir    D:\S2\CV\Final\fixcode\kvasir-seg\Kvasir-SEG\images \
  --out_dir    D:\S2\CV\Final\fixcode\results
```

Each image produces:
- `<name>_mask.png`    — binary segmentation mask
- `<name>_overlay.png` — original image with green mask overlay

---

## Architecture overview

```
Input (3, 352, 352)
     │
  ResNet50 encoder
     ├── layer1 → CBAM → f1 (256)
     ├── layer2 → CBAM → f2 (512)  → RFB → 32ch
     ├── layer3 → CBAM → f3 (1024) → RFB → 32ch
     └── layer4 → CBAM → FFT → f4 (2048) → RFB → 32ch
                                    │
                             Global map head
                                    │
                        ┌───────────┘
                   RA decoder (3 stages)
                        │
              pred2 (finest) + edge head
                        │
                   Upsample to 352×352
```

**Loss:**
```
L_total = 1.0 × L_global
        + 0.8 × L_lat4
        + 0.6 × L_lat3
        + 0.5 × L_lat2
        + 0.4 × L_edge-boundary
```

Each `L_i` = BCE-IoU loss.
`L_edge-boundary` = pixel-wise BCE up-weighted at GT boundary regions (Laplacian edges).
