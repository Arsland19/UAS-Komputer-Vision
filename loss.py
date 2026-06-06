"""
Loss Functions for Enhanced PraNet
Includes:
  1. BCE + IoU loss (standard PraNet loss)
  2. Edge-aware Boundary Loss using Laplacian/Sobel edge detection
  3. Combined multi-scale loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
def _sigmoid(x):
    return torch.sigmoid(x)


# ─────────────────────────────────────────
#  1. BCE + IoU Loss  (PraNet standard)
# ─────────────────────────────────────────
class BCEIoULoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_val = self.bce(pred, target)

        pred_s = torch.sigmoid(pred)
        inter  = (pred_s * target).sum(dim=(2, 3))
        union  = pred_s.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + 1e-6
        iou    = (inter / union).mean()

        return bce_val + (1.0 - iou)


# ─────────────────────────────────────────
#  2. Edge-aware Boundary Loss
# ─────────────────────────────────────────
class EdgeBoundaryLoss(nn.Module):
    """
    Extracts edges from the GT mask using a Laplacian kernel,
    then penalises the prediction specifically at boundary regions.
    """

    def __init__(self, edge_weight: float = 0.5):
        super().__init__()
        self.edge_weight = edge_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # Laplacian kernel for edge extraction
        lap = torch.tensor([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', lap.view(1, 1, 3, 3))

    def _extract_edges(self, mask: torch.Tensor) -> torch.Tensor:
        """mask: (B,1,H,W) float in [0,1]"""
        edges = F.conv2d(mask.float(), self.laplacian.to(mask.device).float(), padding=1)
        edges = edges.abs()
        edges = (edges > 0.1).float()
        return edges

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_map = self.bce(pred, target)           # (B,1,H,W)

        # Edge mask from GT — computed in float32, cast back to match
        edges = self._extract_edges(target).to(pred.dtype)

        # Up-weight loss at boundaries
        weight_map = 1.0 + self.edge_weight * edges
        weighted   = (bce_map * weight_map).mean()

        return weighted


# ─────────────────────────────────────────
#  3. Combined Multi-scale Loss
# ─────────────────────────────────────────
class EnhancedPraNetLoss(nn.Module):
    """
    Multi-output loss for EnhancedPraNet.
    Weights: global > lateral4 > lateral3 > lateral2 > edge
    """

    def __init__(self,
                 edge_weight: float = 0.5,
                 lambda_edge: float = 0.4):
        super().__init__()
        self.bce_iou   = BCEIoULoss()
        self.edge_loss = EdgeBoundaryLoss(edge_weight=edge_weight)
        self.lambda_edge = lambda_edge

    def forward(self, outputs, target):
        """
        outputs: tuple (global_up, pred4, pred3, pred2, edge_up)
        target:  (B,1,H,W) binary mask
        """
        global_up, pred4, pred3, pred2, edge_up = outputs

        # Main segmentation losses (descending weight)
        l_global = self.bce_iou(global_up, target)
        l4       = self.bce_iou(pred4, target)
        l3       = self.bce_iou(pred3, target)
        l2       = self.bce_iou(pred2, target)

        # Edge-aware loss on final prediction
        l_edge   = self.edge_loss(pred2, target)

        total = (1.0 * l_global +
                 0.8 * l4 +
                 0.6 * l3 +
                 0.5 * l2 +
                 self.lambda_edge * l_edge)

        return total, {
            'global': l_global.item(),
            'lat4':   l4.item(),
            'lat3':   l3.item(),
            'lat2':   l2.item(),
            'edge':   l_edge.item(),
        }


if __name__ == '__main__':
    criterion = EnhancedPraNetLoss()
    B = 2
    preds  = [torch.randn(B, 1, 352, 352) for _ in range(5)]
    target = (torch.rand(B, 1, 352, 352) > 0.5).float()
    loss, breakdown = criterion(tuple(preds), target)
    print(f"Total loss: {loss.item():.4f}")
    print(breakdown)
