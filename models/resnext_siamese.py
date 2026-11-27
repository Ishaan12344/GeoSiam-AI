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


class ResNeXtSiameseBDA(nn.Module):
    """
    ResNeXt-50 (32x4d) based Siamese Dual-Stream U-Net for Building Damage Assessment.
    Uses grouped convolutions (cardinality=32) for enhanced feature representation.
    Dual-head architecture:
      - Localization head: segment building footprints from pre-disaster imagery.
      - Damage head: classify damage severity from bi-temporal difference fused features.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNeXtSiameseBDA, self).__init__()

        # ResNeXt-50 32x4d Backbone
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        resnext = models.resnext50_32x4d(weights=weights)

        # Shared Encoder Feature Stages
        # Input: 512x512
        self.conv1 = resnext.conv1       # 64ch,  stride 2 -> 256x256
        self.bn1 = resnext.bn1
        self.relu = resnext.relu
        self.maxpool = resnext.maxpool   # 64ch,  stride 2 -> 128x128

        self.layer1 = resnext.layer1     # 256ch,  128x128
        self.layer2 = resnext.layer2     # 512ch,  64x64
        self.layer3 = resnext.layer3     # 1024ch, 32x32
        self.layer4 = resnext.layer4     # 2048ch, 16x16

        # Decoder Channels
        filters = [64, 256, 512, 1024, 2048]

        # -------------------------------------------------------------
        # Localization Decoder (Pre-Disaster Branch)
        # -------------------------------------------------------------
        self.up4_loc = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.dec4_loc = DecoderBlock(filters[3] + filters[3], 512)

        self.up3_loc = nn.ConvTranspose2d(512, filters[2], kernel_size=2, stride=2)
        self.dec3_loc = DecoderBlock(filters[2] + filters[2], 256)

        self.up2_loc = nn.ConvTranspose2d(256, filters[1], kernel_size=2, stride=2)
        self.dec2_loc = DecoderBlock(filters[1] + filters[1], 128)

        self.up1_loc = nn.ConvTranspose2d(128, filters[0], kernel_size=2, stride=2)
        self.dec1_loc = DecoderBlock(filters[0] + filters[0], 64)

        self.up0_loc = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_loc = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # -------------------------------------------------------------
        # Damage Classification Decoder (Bi-Temporal Fused Branch)
        # -------------------------------------------------------------
        # Bottleneck difference fusion: pre, post, |pre - post|
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(filters[4] * 3, filters[4], kernel_size=1, bias=False),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True)
        )

        self.up4_dam = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        # Skip: up(dam) + pre + post + |pre-post|
        self.dec4_dam = DecoderBlock(filters[3] + filters[3] * 3, 512)

        self.up3_dam = nn.ConvTranspose2d(512, filters[2], kernel_size=2, stride=2)
        self.dec3_dam = DecoderBlock(filters[2] + filters[2] * 3, 256)

        self.up2_dam = nn.ConvTranspose2d(256, filters[1], kernel_size=2, stride=2)
        self.dec2_dam = DecoderBlock(filters[1] + filters[1] * 3, 128)

        self.up1_dam = nn.ConvTranspose2d(128, filters[0], kernel_size=2, stride=2)
        self.dec1_dam = DecoderBlock(filters[0] + filters[0] * 3, 64)

        self.up0_dam = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_dam = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def extract_features(self, x):
        x0 = self.relu(self.bn1(self.conv1(x))) # 64ch,  256x256
        x_pool = self.maxpool(x0)               # 64ch,  128x128
        x1 = self.layer1(x_pool)                # 256ch, 128x128
        x2 = self.layer2(x1)                    # 512ch, 64x64
        x3 = self.layer3(x2)                    # 1024ch, 32x32
        x4 = self.layer4(x3)                    # 2048ch, 16x16
        return [x0, x1, x2, x3, x4]

    def forward(self, pre_img, post_img):
        # 1. Shared ResNeXt Feature Extraction
        pre_feats = self.extract_features(pre_img)
        post_feats = self.extract_features(post_img)

        # 2. Localization Decoder (Pre-Disaster Branch)
        x_loc = self.up4_loc(pre_feats[4])
        x_loc = torch.cat([x_loc, pre_feats[3]], dim=1)
        x_loc = self.dec4_loc(x_loc)

        x_loc = self.up3_loc(x_loc)
        x_loc = torch.cat([x_loc, pre_feats[2]], dim=1)
        x_loc = self.dec3_loc(x_loc)

        x_loc = self.up2_loc(x_loc)
        x_loc = torch.cat([x_loc, pre_feats[1]], dim=1)
        x_loc = self.dec2_loc(x_loc)

        x_loc = self.up1_loc(x_loc)
        x_loc = torch.cat([x_loc, pre_feats[0]], dim=1)
        x_loc = self.dec1_loc(x_loc)

        x_loc = self.up0_loc(x_loc)
        loc_output = self.final_loc(x_loc)

        # 3. Bi-Temporal Feature Fusion for Damage Decoder
        diff_4 = torch.abs(pre_feats[4] - post_feats[4])
        fused_bottleneck = torch.cat([pre_feats[4], post_feats[4], diff_4], dim=1)
        fused_bottleneck = self.bottleneck_fusion(fused_bottleneck)

        x_dam = self.up4_dam(fused_bottleneck)
        diff_3 = torch.abs(pre_feats[3] - post_feats[3])
        x_dam = torch.cat([x_dam, pre_feats[3], post_feats[3], diff_3], dim=1)
        x_dam = self.dec4_dam(x_dam)

        x_dam = self.up3_dam(x_dam)
        diff_2 = torch.abs(pre_feats[2] - post_feats[2])
        x_dam = torch.cat([x_dam, pre_feats[2], post_feats[2], diff_2], dim=1)
        x_dam = self.dec3_dam(x_dam)

        x_dam = self.up2_dam(x_dam)
        diff_1 = torch.abs(pre_feats[1] - post_feats[1])
        x_dam = torch.cat([x_dam, pre_feats[1], post_feats[1], diff_1], dim=1)
        x_dam = self.dec2_dam(x_dam)

        x_dam = self.up1_dam(x_dam)
        diff_0 = torch.abs(pre_feats[0] - post_feats[0])
        x_dam = torch.cat([x_dam, pre_feats[0], post_feats[0], diff_0], dim=1)
        x_dam = self.dec1_dam(x_dam)

        x_dam = self.up0_dam(x_dam)
        dam_output = self.final_dam(x_dam)

        return loc_output, dam_output
