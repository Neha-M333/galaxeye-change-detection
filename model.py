# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EOEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            'efficientnet_b4', pretrained=True, features_only=True)

    def forward(self, x):
        return self.encoder(x)


class SAREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            'efficientnet_b4', pretrained=True, features_only=True)
        old_conv = self.encoder.conv_stem
        new_conv = nn.Conv2d(1, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=False)
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                old_conv.weight.mean(dim=1, keepdim=True))
        self.encoder.conv_stem = new_conv

    def forward(self, x):
        return self.encoder(x)


class CrossModalFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate_eo = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.gate_sar = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, eo_feat, sar_feat):
        combined  = torch.cat([eo_feat, sar_feat], dim=1)
        gate_eo   = self.gate_eo(combined)
        gate_sar  = self.gate_sar(combined)
        gated_eo  = gate_eo  * eo_feat
        gated_sar = gate_sar * sar_feat
        diff      = torch.abs(eo_feat - sar_feat)
        fused     = torch.cat([diff, gated_eo, gated_sar], dim=1)
        return self.project(fused)


class MultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion1 = CrossModalFusion(24)
        self.fusion2 = CrossModalFusion(32)
        self.fusion3 = CrossModalFusion(56)
        self.fusion4 = CrossModalFusion(160)
        self.fusion5 = CrossModalFusion(448)

    def forward(self, eo_feats, sar_feats):
        return [
            self.fusion1(eo_feats[0], sar_feats[0]),
            self.fusion2(eo_feats[1], sar_feats[1]),
            self.fusion3(eo_feats[2], sar_feats[2]),
            self.fusion4(eo_feats[3], sar_feats[3]),
            self.fusion5(eo_feats[4], sar_feats[4]),
        ]


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg     = self.channel_mlp(self.channel_avg(x))
        mx      = self.channel_mlp(self.channel_max(x))
        ch_gate = torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        x = x * ch_gate
        sp_avg  = x.mean(dim=1, keepdim=True)
        sp_max  = x.max(dim=1,  keepdim=True).values
        sp_gate = torch.sigmoid(self.spatial_conv(
            torch.cat([sp_avg, sp_max], dim=1)))
        return x * sp_gate


class SCSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_se = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.channel_se(x) + x * self.spatial_se(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 attention="none"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if attention == "cbam":
            self.attn = CBAM(out_channels)
        elif attention == "scse":
            self.attn = SCSE(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:],
                          mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.attn(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block5 = DecoderBlock(448, 160, 256, attention="cbam")
        self.block4 = DecoderBlock(256, 56,  128, attention="cbam")
        self.block3 = DecoderBlock(128, 32,  64,  attention="scse")
        self.block2 = DecoderBlock(64,  24,  32,  attention="scse")
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.head        = nn.Conv2d(16, 1, 1)
        self.aux_head_8  = nn.Conv2d(128, 1, 1)
        self.aux_head_16 = nn.Conv2d(256, 1, 1)

    def forward(self, fused_feats, training=False):
        f1, f2, f3, f4, f5 = fused_feats
        x     = self.block5(f5, f4)
        aux16 = self.aux_head_16(x)
        x     = self.block4(x, f3)
        aux8  = self.aux_head_8(x)
        x     = self.block3(x, f2)
        x     = self.block2(x, f1)
        x     = F.interpolate(x, scale_factor=2,
                              mode="bilinear", align_corners=False)
        x     = self.block1(x)
        out   = self.head(x)
        if training:
            return out, aux8, aux16
        return out


class ChangeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eo_encoder  = EOEncoder()
        self.sar_encoder = SAREncoder()
        self.fusion      = MultiScaleFusion()
        self.decoder     = Decoder()

    def forward(self, eo, sar, training=False):
        eo_feats  = self.eo_encoder(eo)
        sar_feats = self.sar_encoder(sar)
        fused     = self.fusion(eo_feats, sar_feats)
        return self.decoder(fused, training=training)
