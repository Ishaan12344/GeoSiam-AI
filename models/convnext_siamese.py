import torch
import torch.nn as nn
import torchvision.models as models

class DecoderBlock(nn.Module):
    """
    Standard two-layer convolutional decoder block with BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvNeXtSiameseBDA(nn.Module):
    """
    ConvNeXt-Tiny based Siamese Dual-Stream U-Net for Building Damage Assessment.
    Leverages 7x7 depthwise convolutions and modernized ConvNet design.
    Dual-head architecture:
      - Localization head: segment building footprints from pre-disaster imagery.
      - Damage head: classify damage severity from bi-temporal difference fused features.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(ConvNeXtSiameseBDA, self).__init__()

        # ConvNeXt-Tiny Backbone
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_tiny(weights=weights)

        # Stage channels: [96, 192, 384, 768]
        # Scales for 512x512: [128x128, 64x64, 32x32, 16x16]
        self.stem = convnext.features[0]    # 3 -> 96, stride 4
        self.stage1 = convnext.features[1]  # 96
        self.down1 = convnext.features[2]   # 96 -> 192, stride 2
        self.stage2 = convnext.features[3]  # 192
        self.down2 = convnext.features[4]   # 192 -> 384, stride 2
        self.stage3 = convnext.features[5]  # 384
        self.down3 = convnext.features[6]   # 384 -> 768, stride 2
        self.stage4 = convnext.features[7]  # 768

        # -------------------------------------------------------------
        # Localization Decoder (Pre-Disaster Branch)
        # -------------------------------------------------------------
        self.up4_loc = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.dec4_loc = DecoderBlock(384 + 384, 384)

        self.up3_loc = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.dec3_loc = DecoderBlock(192 + 192, 192)

        self.up2_loc = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec2_loc = DecoderBlock(96 + 96, 96)

        # Upsample by 4x back to input resolution (512x512)
        self.up1_loc = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.final_loc = nn.Conv2d(24, 1, kernel_size=1)

        # -------------------------------------------------------------
        # Damage Classification Decoder (Bi-Temporal Fused Branch)
        # -------------------------------------------------------------
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(768 * 3, 768, kernel_size=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

        self.up4_dam = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        # Skip: up(dam) + pre + post + |pre-post|
        self.dec4_dam = DecoderBlock(384 + 384 * 3, 384)

        self.up3_dam = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.dec3_dam = DecoderBlock(192 + 192 * 3, 192)

        self.up2_dam = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec2_dam = DecoderBlock(96 + 96 * 3, 96)

        self.up1_dam = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.final_dam = nn.Conv2d(24, num_classes, kernel_size=1)

    def extract_features(self, x):
        f1 = self.stage1(self.stem(x))          # 96ch,  128x128
        f2 = self.stage2(self.down1(f1))         # 192ch, 64x64
        f3 = self.stage3(self.down2(f2))         # 384ch, 32x32
        f4 = self.stage4(self.down3(f3))         # 768ch, 16x16
        return [f1, f2, f3, f4]

    def forward(self, pre_img, post_img):
        # 1. Shared ConvNeXt Feature Extraction
        pre_feats = self.extract_features(pre_img)
        post_feats = self.extract_features(post_img)

        # 2. Localization Decoder (Pre-Disaster Branch)
        x_loc = self.up4_loc(pre_feats[3])
        x_loc = torch.cat([x_loc, pre_feats[2]], dim=1)
        x_loc = self.dec4_loc(x_loc)

        x_loc = self.up3_loc(x_loc)
        x_loc = torch.cat([x_loc, pre_feats[1]], dim=1)
        x_loc = self.dec3_loc(x_loc)

        x_loc = self.up2_loc(x_loc)
        x_loc = torch.cat([x_loc, pre_feats[0]], dim=1)
        x_loc = self.dec2_loc(x_loc)

        x_loc = self.up1_loc(x_loc)
        loc_output = self.final_loc(x_loc)

        # 3. Bi-Temporal Feature Fusion for Damage Decoder
        diff_3 = torch.abs(pre_feats[3] - post_feats[3])
        fused_bottleneck = torch.cat([pre_feats[3], post_feats[3], diff_3], dim=1)
        fused_bottleneck = self.bottleneck_fusion(fused_bottleneck)

        x_dam = self.up4_dam(fused_bottleneck)
        diff_2 = torch.abs(pre_feats[2] - post_feats[2])
        x_dam = torch.cat([x_dam, pre_feats[2], post_feats[2], diff_2], dim=1)
        x_dam = self.dec4_dam(x_dam)

        x_dam = self.up3_dam(x_dam)
        diff_1 = torch.abs(pre_feats[1] - post_feats[1])
        x_dam = torch.cat([x_dam, pre_feats[1], post_feats[1], diff_1], dim=1)
        x_dam = self.dec3_dam(x_dam)

        x_dam = self.up2_dam(x_dam)
        diff_0 = torch.abs(pre_feats[0] - post_feats[0])
        x_dam = torch.cat([x_dam, pre_feats[0], post_feats[0], diff_0], dim=1)
        x_dam = self.dec2_dam(x_dam)

        x_dam = self.up1_dam(x_dam)
        dam_output = self.final_dam(x_dam)

        return loc_output, dam_output
