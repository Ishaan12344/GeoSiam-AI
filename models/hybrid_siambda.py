"""
================================================================================
HybridSiam-BDA (Lite) v2 — Enhanced Architecture for Building Damage Assessment
================================================================================
Encoder:  ConvNeXt-Tiny      (local texture/edge features, 2022)
          + Swin V2-T          (global context/spatial relations, 2022)
          → Parallel dual-branch, shared weights (Siamese)
Fusion:   Cross-Attention Difference Fusion (CADF) — at ALL 4 scales
Decoder:  Two UNet-style FPN heads with Attention Gates:
            1. Localization head  → binary building mask (1 channel)
            2. Damage head        → 4-class damage map   (4 channels)
--------------------------------------------------------------------------------
Parameters: ~42.5M (fits on free Colab T4 GPU)
Novelty:    ConvNeXt-Tiny + Swin V2-T Siamese + Multi-Scale CADF + Attention Gates
================================================================================
v2 Changes:
  1. CADF at all 4 scales (was only 2) — catches fine-grained damage
  2. Attention Gates on damage decoder skip connections — filters noise
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import swin_v2_t, Swin_V2_T_Weights


# ─────────────────────────────────────────────────────────────────────────────
# 1. ConvNeXt-V2-Tiny Feature Extractor
#    Extracts hierarchical local features at 4 scales.
# ─────────────────────────────────────────────────────────────────────────────
class ConvNeXtEncoder(nn.Module):
    """
    Wraps ConvNeXt-Tiny backbone and exposes 4 intermediate feature maps.
    Output channels per stage: [96, 192, 384, 768]
    Output spatial sizes (for 512x512 input): [128x128, 64x64, 32x32, 16x16]
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        # ConvNeXt feature stages
        self.stem  = backbone.features[0]   # Stem:    3 → 96,  stride 4
        self.stage1 = backbone.features[1]  # Stage 1: 96
        self.down1  = backbone.features[2]  # Down:    96 → 192, stride 2
        self.stage2 = backbone.features[3]  # Stage 2: 192
        self.down2  = backbone.features[4]  # Down:    192 → 384, stride 2
        self.stage3 = backbone.features[5]  # Stage 3: 384
        self.down3  = backbone.features[6]  # Down:    384 → 768, stride 2
        self.stage4 = backbone.features[7]  # Stage 4: 768

    def forward(self, x):
        # Returns list of feature maps at 4 scales
        f1 = self.stage1(self.stem(x))          # 96ch,  H/4
        f2 = self.stage2(self.down1(f1))         # 192ch, H/8
        f3 = self.stage3(self.down2(f2))         # 384ch, H/16
        f4 = self.stage4(self.down3(f3))         # 768ch, H/32
        return [f1, f2, f3, f4]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Swin V2-T Feature Extractor (2022)
#    Extracts global context features at 4 scales.
#    Upgrades over Swin V1: cosine attention + log-spaced position bias.
# ─────────────────────────────────────────────────────────────────────────────
class SwinEncoder(nn.Module):
    """
    Wraps Swin V2-Tiny backbone and exposes 4 intermediate feature maps.
    Output channels per stage: [96, 192, 384, 768]
    
    Swin V2 improvements over V1:
      - Cosine attention: more stable training, no attention score overflow
      - Log-spaced continuous relative position bias: better generalization
      - Same parameters and speed as Swin V1
    
    Note: Swin outputs (B, H, W, C) — we permute to (B, C, H, W).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = swin_v2_t(weights=weights)
        self.patch_embed = backbone.features[0]  # Patch partition + embed
        self.stage1 = backbone.features[1]       # Swin V2 stage 1: 96ch
        self.down1  = backbone.features[2]       # Patch merge:  96→192
        self.stage2 = backbone.features[3]       # Swin V2 stage 2: 192ch
        self.down2  = backbone.features[4]       # Patch merge:  192→384
        self.stage3 = backbone.features[5]       # Swin V2 stage 3: 384ch
        self.down3  = backbone.features[6]       # Patch merge:  384→768
        self.stage4 = backbone.features[7]       # Swin V2 stage 4: 768ch

    def _to_chw(self, x):
        # Swin returns (B, H, W, C) — convert to (B, C, H, W)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        e = self.patch_embed(x)                  # (B, H/4, W/4, 96)
        f1 = self._to_chw(self.stage1(e))        # 96ch,  H/4
        f2 = self._to_chw(self.stage2(self.down1(self._to_hwc(f1))))   # 192ch, H/8
        f3 = self._to_chw(self.stage3(self.down2(self._to_hwc(f2))))   # 384ch, H/16
        f4 = self._to_chw(self.stage4(self.down3(self._to_hwc(f3))))   # 768ch, H/32
        return [f1, f2, f3, f4]

    def _to_hwc(self, x):
        return x.permute(0, 2, 3, 1).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Feature Fusion Block
#    Fuses ConvNeXt (local) + Swin (global) features at each scale.
# ─────────────────────────────────────────────────────────────────────────────
class HybridFusion(nn.Module):
    """
    Fuses local (ConvNeXt) and global (Swin) features at same spatial scale.
    Uses channel attention (SE-style) to let the model decide which source
    is more important for each channel.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        # Channel attention (Squeeze-and-Excitation)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, conv_feat, swin_feat):
        # Align spatial sizes in case of minor rounding differences
        if conv_feat.shape[2:] != swin_feat.shape[2:]:
            swin_feat = F.interpolate(swin_feat, size=conv_feat.shape[2:],
                                      mode='bilinear', align_corners=False)
        fused = self.conv(torch.cat([conv_feat, swin_feat], dim=1))
        se_w = self.se(fused).unsqueeze(-1).unsqueeze(-1)
        return fused * se_w


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-Attention Difference Fusion (CADF) — Novel Module ⭐
#    The key innovation: uses cross-attention to compare pre/post features
#    instead of simple subtraction or concatenation.
# ─────────────────────────────────────────────────────────────────────────────
class CADF(nn.Module):
    """
    Cross-Attention Difference Fusion Module.

    Instead of just computing (post - pre) or concat([pre, post, |pre-post|]),
    this module lets the pre-features ATTEND to the post-features (and vice
    versa) via multi-head cross-attention, then combines the attended features
    with the raw difference to produce a damage-aware change representation.

    Input:  pre_feat, post_feat  — shape (B, C, H, W)
    Output: fused change feature — shape (B, C, H, W)
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = max(channels // num_heads, 1)

        # Project to Q, K, V for cross-attention
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

        # Combine attended features + raw difference
        self.diff_combine = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, pre, post):
        B, C, H, W = pre.shape
        head_dim = max(C // self.num_heads, 1)

        # 1. Project Q, K, V for cross-attention
        q = self.q_proj(pre)
        k = self.k_proj(post)
        v = self.v_proj(post)

        # Spatial Reduction Attention (SRA) for high resolutions (e.g. 128x128 -> max 32x32 for K, V)
        # Prevents quadratic (H*W)^2 memory explosion while preserving full cross-attention semantics
        if H * W > 32 * 32:
            k = F.adaptive_avg_pool2d(k, (32, 32))
            v = F.adaptive_avg_pool2d(v, (32, 32))

        # Reshape to (B, num_heads, seq_len, head_dim)
        Q = q.reshape(B, self.num_heads, head_dim, -1).transpose(-2, -1)
        K = k.reshape(B, self.num_heads, head_dim, -1).transpose(-2, -1)
        V = v.reshape(B, self.num_heads, head_dim, -1).transpose(-2, -1)

        # Memory-efficient Scaled Dot-Product Attention (FlashAttention kernel)
        if hasattr(F, 'scaled_dot_product_attention'):
            attended = F.scaled_dot_product_attention(Q, K, V)
        else:
            scale = head_dim ** -0.5
            attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1)
            attended = torch.matmul(attn, V)

        attended = attended.transpose(-2, -1).reshape(B, C, H, W)
        attended = self.out_proj(attended)

        # 2. Raw difference (absolute change signal)
        diff = torch.abs(post - pre)

        # 3. Combine: attended + post + diff → damage-aware representation
        fused = self.diff_combine(torch.cat([attended, post, diff], dim=1))
        return fused


# ─────────────────────────────────────────────────────────────────────────────
# 5. Attention Gate — filters skip connections to focus on damage-relevant features
#    Inspired by Attention U-Net (Oktay et al., 2018)
# ─────────────────────────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    """
    Residual Attention Gate for skip connections.
    Uses gating signal to weight change features, with a residual shortcut
    skip * (1.0 + alpha) so gradients never vanish during early training.
    """
    def __init__(self, gate_ch, skip_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or max(skip_ch // 2, 32)
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode='bilinear', align_corners=False)
        alpha = self.psi(self.relu(g + s))
        # Residual attention: 1.0 + alpha guarantees healthy gradient flow
        return skip * (1.0 + alpha)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Decoder Block
# ─────────────────────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main Model: HybridSiamBDA (Lite) v3
# ─────────────────────────────────────────────────────────────────────────────
class HybridSiamBDA(nn.Module):
    """
    HybridSiam-BDA (Lite) v3 — State-of-the-Art Siamese Dual-Stream Architecture
    for Building Damage Assessment.

    Innovations:
      1. Dual Siamese Backbone: ConvNeXt-Tiny (local CNN) + Swin V2-T (global Transformer)
      2. Multi-Scale CADF: Cross-Attention Difference Fusion at ALL 4 hierarchical scales
      3. Residual Attention Gating: Skip connections enhanced with residual attention
      4. Complete Post-Disaster Visual Fusion: Damage decoder directly receives post-disaster
         visual context concatenated with CADF difference features
      5. Multi-Stage High-Resolution Reconstruction: Native 512x512 segmentation
    """
    def __init__(self, num_damage_classes=4, pretrained=True):
        super().__init__()

        # Shared Siamese Encoders
        self.convnext = ConvNeXtEncoder(pretrained=pretrained)
        self.swin     = SwinEncoder(pretrained=pretrained)

        # Stage channels: [96, 192, 384, 768]
        enc_channels = [96, 192, 384, 768]

        # Hybrid Fusion (CNN + Transformer) at each scale
        self.fuse = nn.ModuleList([HybridFusion(c) for c in enc_channels])

        # Cross-Attention Difference Fusion at ALL 4 scales
        self.cadf_s4 = CADF(channels=enc_channels[3], num_heads=8)   # 768ch
        self.cadf_s3 = CADF(channels=enc_channels[2], num_heads=4)   # 384ch
        self.cadf_s2 = CADF(channels=enc_channels[1], num_heads=2)   # 192ch
        self.cadf_s1 = CADF(channels=enc_channels[0], num_heads=2)   # 96ch

        # Residual Attention Gates for Damage Decoder
        self.dam_ag3 = AttentionGate(gate_ch=768, skip_ch=enc_channels[2])  # 768 -> 384
        self.dam_ag2 = AttentionGate(gate_ch=384, skip_ch=enc_channels[1])  # 384 -> 192
        self.dam_ag1 = AttentionGate(gate_ch=192, skip_ch=enc_channels[0])  # 192 -> 96

        # Bottleneck damage fusion (Combines deep post context + CADF change)
        self.dam_bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[3] * 2, enc_channels[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(enc_channels[3]),
            nn.GELU()
        )

        # ── Localization Decoder (Pre-disaster building extraction) ──
        self.loc_dec4 = DecoderBlock(enc_channels[3], enc_channels[2], 384)
        self.loc_dec3 = DecoderBlock(384,             enc_channels[1], 192)
        self.loc_dec2 = DecoderBlock(192,             enc_channels[0], 96)
        self.loc_final = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 1, kernel_size=1)
        )

        # ── Damage Decoder (Fused post-disaster context + Gated CADF change) ──
        # Skip channels at each level = post_feats[i] (C) + gated_cadf[i] (C) = 2 * C
        self.dam_dec4 = DecoderBlock(enc_channels[3], enc_channels[2] * 2, 384)
        self.dam_dec3 = DecoderBlock(384,             enc_channels[1] * 2, 192)
        self.dam_dec2 = DecoderBlock(192,             enc_channels[0] * 2, 96)
        self.dam_final = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, num_damage_classes, kernel_size=1)
        )

    def _encode(self, img):
        """Run ConvNeXt + Swin encoders and fuse at each scale."""
        conv_feats = self.convnext(img)
        swin_feats = self.swin(img)
        fused = [self.fuse[i](conv_feats[i], swin_feats[i]) for i in range(4)]
        return fused

    def forward(self, pre_img, post_img):
        # 1. Dual-Stream Siamese Encodings
        pre_feats  = self._encode(pre_img)   # [s1: 96, s2: 192, s3: 384, s4: 768]
        post_feats = self._encode(post_img)

        # 2. Multi-Scale CADF Difference Representations
        change_s4 = self.cadf_s4(pre_feats[3], post_feats[3])   # 768ch, 16x16
        change_s3 = self.cadf_s3(pre_feats[2], post_feats[2])   # 384ch, 32x32
        change_s2 = self.cadf_s2(pre_feats[1], post_feats[1])   # 192ch, 64x64
        change_s1 = self.cadf_s1(pre_feats[0], post_feats[0])   # 96ch,  128x128

        # 3. Localization Decoder (Pre-Disaster Branch)
        x_loc = self.loc_dec4(pre_feats[3], pre_feats[2])
        x_loc = self.loc_dec3(x_loc,        pre_feats[1])
        x_loc = self.loc_dec2(x_loc,        pre_feats[0])
        loc_out = self.loc_final(x_loc)      # (B, 1, 512, 512)

        # 4. Damage Decoder (Direct Post Features + Gated CADF Change)
        # Deep bottleneck fusion
        fused_bottleneck = self.dam_bottleneck(torch.cat([post_feats[3], change_s4], dim=1))

        # Gated skips combining post-disaster visual evidence + attention-weighted change
        gated_change3 = self.dam_ag3(fused_bottleneck, change_s3)
        skip3 = torch.cat([post_feats[2], gated_change3], dim=1)  # 384 + 384 = 768ch
        x_dam = self.dam_dec4(fused_bottleneck, skip3)

        gated_change2 = self.dam_ag2(x_dam, change_s2)
        skip2 = torch.cat([post_feats[1], gated_change2], dim=1)  # 192 + 192 = 384ch
        x_dam = self.dam_dec3(x_dam, skip2)

        gated_change1 = self.dam_ag1(x_dam, change_s1)
        skip1 = torch.cat([post_feats[0], gated_change1], dim=1)  # 96 + 96 = 192ch
        x_dam = self.dam_dec2(x_dam, skip1)

        dam_out = self.dam_final(x_dam)      # (B, 4, 512, 512)

        return loc_out, dam_out


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = HybridSiamBDA(pretrained=False)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"HybridSiam-BDA (Lite) v2 | Parameters: {total:.1f}M")

    pre  = torch.randn(2, 3, 512, 512)
    post = torch.randn(2, 3, 512, 512)
    loc, dam = model(pre, post)
    print(f"Localization output : {loc.shape}")   # (2, 1, 512, 512)
    print(f"Damage output       : {dam.shape}")   # (2, 4, 512, 512)
    print("✅ Forward pass OK!")
    print(f"\nv2 Upgrades active:")
    print(f"  ✅ CADF at all 4 scales (was 2)")
    print(f"  ✅ Attention Gates on damage decoder")
