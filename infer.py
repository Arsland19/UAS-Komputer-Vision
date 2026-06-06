"""
Run inference with trained Enhanced PraNet model.

Usage:
    python infer.py --checkpoint checkpoints/best_model.pth \
                    --img_dir  path/to/test/images \
                    --out_dir  results/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import EnhancedPraNet


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_model(checkpoint: str) -> EnhancedPraNet:
    model = EnhancedPraNet(pretrained=False).to(DEVICE)
    ckpt  = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state', ckpt)          # support both save formats
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model: EnhancedPraNet, img_path: str, threshold: float = 0.5):
    """Returns (original_image_np, predicted_mask_np, overlay_np)"""
    orig = cv2.imread(img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    oh, ow = orig.shape[:2]

    pil  = Image.fromarray(orig)
    inp  = IMG_TRANSFORM(pil).unsqueeze(0).to(DEVICE)

    outputs = model(inp)
    pred    = outputs[3]                    # final finest prediction
    pred    = torch.sigmoid(pred)
    pred    = F.interpolate(pred, size=(oh, ow), mode='bilinear', align_corners=False)
    pred_np = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255

    # Colour overlay
    overlay  = orig.copy()
    mask_col = np.zeros_like(orig)
    mask_col[:, :, 1] = pred_np             # green channel
    overlay  = cv2.addWeighted(overlay, 0.7, mask_col, 0.3, 0)

    return orig, pred_np, overlay


def run(checkpoint: str, img_dir: str, out_dir: str, threshold: float = 0.5):
    model   = load_model(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
    print(f"Found {len(img_paths)} images in {img_dir}")

    for p in img_paths:
        orig, mask, overlay = predict(model, str(p), threshold)
        stem = p.stem

        cv2.imwrite(str(out_dir / f'{stem}_mask.png'),
                    mask)
        cv2.imwrite(str(out_dir / f'{stem}_overlay.png'),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"  Saved {stem}_mask.png  +  {stem}_overlay.png")

    print(f"\nDone. Results in {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img_dir',    required=True)
    parser.add_argument('--out_dir',    default='results')
    parser.add_argument('--threshold',  type=float, default=0.5)
    args = parser.parse_args()
    run(args.checkpoint, args.img_dir, args.out_dir, args.threshold)
