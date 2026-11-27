import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance in building damage classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            valid = (targets >= 0) & (targets < len(self.alpha))
            alpha_t = torch.ones_like(ce_loss)
            alpha_t[valid] = self.alpha[targets[valid]]
            focal_loss = alpha_t * focal_loss

        valid_mask = (targets != self.ignore_index)
        if self.reduction == 'mean':
            if valid_mask.sum() > 0:
                return focal_loss[valid_mask].mean()
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss[valid_mask].sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for binary building localization mask segmentation.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class JointBDALoss(nn.Module):
    """
    Multi-task Joint Loss combining Building Localization (Dice + BCE)
    and Damage Categorization (Focal Loss).
    L_total = L_loc + lambda_damage * L_damage
    """
    def __init__(self, class_weights=None, lambda_damage=1.5):
        super(JointBDALoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        self.lambda_damage = lambda_damage

    def forward(self, loc_logits, loc_targets, dam_logits, dam_targets):
        # Localization Loss
        l_dice = self.dice_loss(loc_logits, loc_targets)
        l_bce = self.bce_loss(loc_logits, loc_targets)
        l_loc = l_dice + l_bce

        # Damage Classification Loss
        l_dam = self.focal_loss(dam_logits, dam_targets)

        # Total Weighted Multi-Task Loss
        total_loss = l_loc + (self.lambda_damage * l_dam)
        return total_loss, l_loc, l_dam
