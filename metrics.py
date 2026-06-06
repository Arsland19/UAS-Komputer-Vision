"""
Evaluation metrics for binary segmentation.
"""

import torch
import numpy as np


def dice_score(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred   = (torch.sigmoid(pred) > threshold).float()
    inter  = (pred * target).sum(dim=(1, 2, 3))
    denom  = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice   = ((2.0 * inter + smooth) / (denom + smooth)).mean()
    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred  = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou   = ((inter + smooth) / (union + smooth)).mean()
    return iou.item()


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._dice = []
        self._iou  = []
        self._loss = []

    def update(self, loss_val: float, pred: torch.Tensor, target: torch.Tensor):
        self._loss.append(loss_val)
        self._dice.append(dice_score(pred.detach().cpu(), target.detach().cpu()))
        self._iou.append( iou_score( pred.detach().cpu(), target.detach().cpu()))

    def summary(self):
        return {
            'loss': np.mean(self._loss),
            'dice': np.mean(self._dice),
            'iou':  np.mean(self._iou),
        }
