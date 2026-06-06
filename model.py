"""
Enhanced PraNet (Parallel Reverse Attention Network)
Modifications:
  1. CBAM (Convolutional Block Attention Module)
  2. FFT-based frequency enhancement
  3. Edge-aware Boundary Loss (in loss.py)

Reference: PraNet: Parallel Reverse Attention Network for Polyp Segmentation
           Fan et al., MICCAI 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ─────────────────────────────────────────
#  1. CBAM
# ─────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ─────────────────────────────────────────
#  2. FFT Frequency Enhancement Module
# ─────────────────────────────────────────
class FFTEnhancement(nn.Module):
    """
    Applies FFT to the feature map, enhances high-frequency components
    (edges/boundaries), then returns to spatial domain.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Learnable frequency weight (channel-wise)
        self.freq_weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.norm = nn.BatchNorm2d(in_channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, H, W)
        orig_dtype = x.dtype
        x = x.float()                                      # FFT requires float32

        # FFT along spatial dims
        x_fft = torch.fft.rfft2(x, norm='ortho')          # complex (B,C,H,W//2+1)

        # Amplitude & phase
        amp   = x_fft.abs()
        phase = x_fft.angle()

        # Enhance amplitude (learn which frequencies matter)
        amp_w = amp * self.freq_weight.float().abs()

        # Reconstruct complex tensor
        x_fft_w = torch.polar(amp_w, phase)

        # Inverse FFT back to spatial domain
        x_back = torch.fft.irfft2(x_fft_w, s=(x.shape[2], x.shape[3]), norm='ortho')

        out = self.act(self.norm(x + x_back))              # residual fusion
        return out.to(orig_dtype)                          # restore original dtype


# ─────────────────────────────────────────
#  3. Basic convolution blocks
# ─────────────────────────────────────────
class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RFB(nn.Module):
    """Receptive Field Block — lightweight multi-scale context."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 4
        self.b0 = ConvBnRelu(in_ch, mid, 1, p=0)
        self.b1 = nn.Sequential(ConvBnRelu(in_ch, mid, 1, p=0),
                                 ConvBnRelu(mid, mid, 3))
        self.b2 = nn.Sequential(ConvBnRelu(in_ch, mid, 1, p=0),
                                 ConvBnRelu(mid, mid, 5, p=2))
        self.b3 = nn.Sequential(ConvBnRelu(in_ch, mid, 1, p=0),
                                 ConvBnRelu(mid, mid, 3),
                                 ConvBnRelu(mid, mid, 3, dilation=3, p=3))
        self.fuse = ConvBnRelu(mid * 4, out_ch, 1, p=0)

    def forward(self, x):
        return self.fuse(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], 1))


# ─────────────────────────────────────────
#  4. Reverse Attention Module
# ─────────────────────────────────────────
class ReverseAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, coarse_pred, target_size):
        # Upsample coarse prediction to match x
        coarse = F.interpolate(coarse_pred, size=x.shape[2:], mode='bilinear', align_corners=False)
        # Reverse attention: focus on NOT-yet-predicted regions
        attention = 1.0 - torch.sigmoid(coarse)
        x = x * attention
        x = self.conv(x)
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)


# ─────────────────────────────────────────
#  5. Enhanced PraNet
# ─────────────────────────────────────────
class EnhancedPraNet(nn.Module):
    """
    PraNet backbone (Res2Net-like via ResNet50) with:
      - CBAM on each encoder level
      - FFT Enhancement on the deepest feature
      - Three RA (Reverse Attention) decoder branches
      - Global map head
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        # ── Encoder (ResNet50) ──────────────────────
        backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # C=256
        self.layer2 = backbone.layer2   # C=512
        self.layer3 = backbone.layer3   # C=1024
        self.layer4 = backbone.layer4   # C=2048

        # ── CBAM on each level ──────────────────────
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        # ── FFT Enhancement on deepest feature ──────
        self.fft4 = FFTEnhancement(2048)

        # ── Feature aggregation (RFB) ───────────────
        self.rfb2 = RFB(512,  32)
        self.rfb3 = RFB(1024, 32)
        self.rfb4 = RFB(2048, 32)

        # ── Global map ──────────────────────────────
        self.global_head = nn.Conv2d(32, num_classes, 1)

        # ── Reverse Attention decoder ────────────────
        self.ra4 = ReverseAttention(32, 32)
        self.ra3 = ReverseAttention(32, 32)
        self.ra2 = ReverseAttention(32, 32)

        # ── Final prediction heads ───────────────────
        self.head4 = nn.Conv2d(32, num_classes, 1)
        self.head3 = nn.Conv2d(32, num_classes, 1)
        self.head2 = nn.Conv2d(32, num_classes, 1)

        # ── Edge head (auxiliary boundary output) ────
        self.edge_conv = nn.Sequential(
            ConvBnRelu(32, 32),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # ── Encoder ─────────────────────────────────
        x0 = self.layer0(x)           # /4
        x1 = self.cbam1(self.layer1(x0))  # 256,  /4
        x2 = self.cbam2(self.layer2(x1))  # 512,  /8
        x3 = self.cbam3(self.layer3(x2))  # 1024, /16
        x4 = self.cbam4(self.layer4(x3))  # 2048, /32

        # FFT on deepest feature
        x4 = self.fft4(x4)

        # ── Aggregate ───────────────────────────────
        f2 = self.rfb2(x2)            # 32, /8
        f3 = self.rfb3(x3)            # 32, /16
        f4 = self.rfb4(x4)            # 32, /32

        # ── Global map ──────────────────────────────
        global_map = self.global_head(f4)                    # /32
        global_up  = F.interpolate(global_map, size=(H, W),
                                    mode='bilinear', align_corners=False)

        # ── RA decoder ──────────────────────────────
        ra4_feat = self.ra4(f4, global_map, f3.shape[2:])    # /16
        pred4    = self.head4(ra4_feat)

        ra3_feat = self.ra3(f3 + ra4_feat, pred4, f2.shape[2:])  # /8
        pred3    = self.head3(ra3_feat)

        ra2_feat = self.ra2(f2 + ra3_feat, pred3, (H // 4, W // 4))  # /4
        pred2    = self.head2(ra2_feat)

        # ── Edge output (auxiliary) ──────────────────
        edge_map = self.edge_conv(ra2_feat)

        # ── Upsample all to original size ────────────
        pred4_up = F.interpolate(pred4, (H, W), mode='bilinear', align_corners=False)
        pred3_up = F.interpolate(pred3, (H, W), mode='bilinear', align_corners=False)
        pred2_up = F.interpolate(pred2, (H, W), mode='bilinear', align_corners=False)
        edge_up  = F.interpolate(edge_map, (H, W), mode='bilinear', align_corners=False)

        # Return: (global, lateral4, lateral3, lateral2, edge)
        return global_up, pred4_up, pred3_up, pred2_up, edge_up


if __name__ == '__main__':
    model = EnhancedPraNet(pretrained=False)
    dummy = torch.randn(2, 3, 352, 352)
    outs  = model(dummy)
    for i, o in enumerate(outs):
        print(f"out[{i}]: {o.shape}")
