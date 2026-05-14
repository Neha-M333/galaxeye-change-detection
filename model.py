
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ──────────────────────────────────────────────
# Attention modules
# ──────────────────────────────────────────────
class SCSE(nn.Module):
    """Concurrent Spatial & Channel Squeeze-Excitation."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid()
        )
        self.sse = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        cse = self.cse(x).view(x.size(0), x.size(1), 1, 1)
        sse = self.sse(x)
        return x * cse + x * sse

class CBAM(nn.Module):
    """Convolutional Block Attention Module — channel then spatial."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.ca_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_max = nn.AdaptiveMaxPool2d(1)
        self.ca_fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels)
        )
        self.sa_conv = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(
            self.ca_fc(self.ca_avg(x)) + self.ca_fc(self.ca_max(x))
        ).view(x.size(0), x.size(1), 1, 1)
        x = x * ca
        sa = torch.sigmoid(self.sa_conv(
            torch.cat([x.mean(dim=1, keepdim=True),
                       x.max(dim=1, keepdim=True).values], dim=1)
        ))
        return x * sa

# ──────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_cbam=False):
        super().__init__()
        total_in = in_ch + skip_ch
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,    out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.attn = CBAM(out_ch) if use_cbam else SCSE(out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.attn(self.conv(x))

# ──────────────────────────────────────────────
# Cross-modal attention fusion
# ──────────────────────────────────────────────
class CrossModalFusion(nn.Module):
    """
    Cross-attention fusion: EO features attend to SAR and vice versa,
    then both are gated by their absolute difference.
    More expressive than simple abs-diff + concat + 1x1.
    """
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 4, 4)
        self.eo_gate  = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),     nn.Sigmoid()
        )
        self.sar_gate = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),     nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )

    def forward(self, eo, sar):
        # Align spatial dims if EO and SAR encoders differ slightly
        if eo.shape[-2:] != sar.shape[-2:]:
            sar = F.interpolate(sar, size=eo.shape[-2:],
                                mode="bilinear", align_corners=False)
        diff    = torch.abs(eo - sar)
        eo_att  = self.eo_gate(torch.cat([eo, sar], dim=1))
        sar_att = self.sar_gate(torch.cat([sar, eo], dim=1))
        return self.proj(torch.cat([diff, eo * eo_att, sar * sar_att], dim=1))

# ──────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────
class SiameseChangeDetector(nn.Module):
    """
    Dual EfficientNet-B4 encoder + CrossModalFusion at each scale +
    CBAM-augmented decoder + optional deep supervision.
    """
    def __init__(self, pretrained=True, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # EO encoder — 3 channels
        self.eo_encoder = timm.create_model(
            "efficientnet_b4", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3, 4)
        )

        # SAR encoder — 1 channel, weight-averaged from pretrained
        self.sar_encoder = timm.create_model(
            "efficientnet_b4", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3, 4)
        )
        old_conv = self.sar_encoder.conv_stem
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                old_conv.weight.mean(dim=1, keepdim=True)
            )
        self.sar_encoder.conv_stem = new_conv

        # ── CORRECTED: EfficientNet-B4 actual features_only channel dims ──
        enc_ch = [24, 32, 56, 160, 448]   # was [24, 32, 56, 160, 272] — 272 is wrong

        # Cross-modal fusion at each encoder scale
        self.fuse = nn.ModuleList([
            CrossModalFusion(c) for c in enc_ch
        ])

        # Decoder — CBAM on deeper blocks for richer spatial attention
        self.dec4 = DecoderBlock(enc_ch[4], enc_ch[3], 256, use_cbam=True)
        self.dec3 = DecoderBlock(256,        enc_ch[2], 128, use_cbam=True)
        self.dec2 = DecoderBlock(128,        enc_ch[1], 64,  use_cbam=False)
        self.dec1 = DecoderBlock(64,         enc_ch[0], 32,  use_cbam=False)
        self.dec0 = DecoderBlock(32,         0,         16,  use_cbam=False)

        # Main segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )

        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_head3 = nn.Conv2d(128, 1, 1)   # /8 scale
            self.aux_head4 = nn.Conv2d(256, 1, 1)   # /16 scale

    def forward(self, eo, sar):
        eo_feats  = self.eo_encoder(eo)
        sar_feats = self.sar_encoder(sar)

        # Cross-modal fusion at each scale
        fused = [self.fuse[i](eo_feats[i], sar_feats[i])
                 for i in range(5)]

        # Decode
        d4 = self.dec4(fused[4], fused[3])
        d3 = self.dec3(d4,       fused[2])
        d2 = self.dec2(d3,       fused[1])
        d1 = self.dec1(d2,       fused[0])
        d0 = self.dec0(d1)

        out = F.interpolate(self.head(d0), size=eo.shape[-2:],
                            mode="bilinear", align_corners=False)

        if self.deep_supervision and self.training:
            aux3 = F.interpolate(self.aux_head3(d3), size=eo.shape[-2:],
                                 mode="bilinear", align_corners=False)
            aux4 = F.interpolate(self.aux_head4(d4), size=eo.shape[-2:],
                                 mode="bilinear", align_corners=False)
            return out, aux3, aux4
        return out

# ──────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma=2.5, label_smoothing=0.05):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) +                       0.5 * self.label_smoothing
        bce   = F.binary_cross_entropy_with_logits(logits, targets,
                                                    reduction="none")
        probs = torch.sigmoid(logits)
        pt    = targets * probs + (1 - targets) * (1 - probs)
        alpha = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha * (1 - pt) ** self.gamma * bce).mean()

class DiceLoss(nn.Module):
    def forward(self, logits, targets, smooth=1.0):
        probs = torch.sigmoid(logits).view(-1)
        tgts  = targets.view(-1)
        inter = (probs * tgts).sum()
        return 1 - (2 * inter + smooth) / (probs.sum() + tgts.sum() + smooth)

class TverskyLoss(nn.Module):
    """
    alpha=0.3, beta=0.7 → penalises FN more than FP.
    Biases model toward high recall on change pixels.
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        tgts  = targets.view(-1)
        tp = (probs * tgts).sum()
        fp = (probs * (1 - tgts)).sum()
        fn = ((1 - probs) * tgts).sum()
        tversky = (tp + self.smooth) /                   (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

class CombinedLoss(nn.Module):
    def __init__(self, focal_w=0.4, dice_w=0.4, tversky_w=0.2,
                 alpha=0.80, gamma=2.5, tv_alpha=0.3, tv_beta=0.7,
                 label_smoothing=0.05, ds_weight=0.3):
        super().__init__()
        self.focal_w    = focal_w
        self.dice_w     = dice_w
        self.tversky_w  = tversky_w
        self.ds_weight  = ds_weight
        self.focal      = FocalLoss(alpha, gamma, label_smoothing)
        self.dice       = DiceLoss()
        self.tversky    = TverskyLoss(tv_alpha, tv_beta)

    def _base_loss(self, logits, targets):
        return (self.focal_w   * self.focal(logits, targets) +
                self.dice_w    * self.dice(logits,  targets) +
                self.tversky_w * self.tversky(logits, targets))

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main, aux3, aux4 = outputs
            main_loss = self._base_loss(main, targets)
            aux_loss  = (self._base_loss(aux3, targets) +
                         self._base_loss(aux4, targets)) / 2
            return main_loss + self.ds_weight * aux_loss
        return self._base_loss(outputs, targets)

def get_model(config):
    return SiameseChangeDetector(
        pretrained=config["model"]["pretrained"],
        deep_supervision=config["model"].get("deep_supervision", True)
    )

def get_loss(config):
    return CombinedLoss(
        focal_w=config["loss"]["focal_weight"],
        dice_w=config["loss"]["dice_weight"],
        tversky_w=config["loss"].get("tversky_weight", 0.2),
        alpha=config["loss"]["focal_alpha"],
        gamma=config["loss"]["focal_gamma"],
        tv_alpha=config["loss"].get("tversky_alpha", 0.3),
        tv_beta=config["loss"].get("tversky_beta", 0.7),
        label_smoothing=config["loss"].get("label_smoothing", 0.05),
        ds_weight=config["model"].get("ds_weight", 0.3)
    )
