import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def compute_localization_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    Computes IoU, Precision, Recall, and F1-Score for Building Localization.
    """
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    
    iou = (intersection + 1e-6) / (union + 1e-6)

    tp = intersection
    fp = pred_bin.sum() - tp
    fn = gt_bin.sum() - tp

    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    return {
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_loc": float(f1)
    }

def compute_damage_metrics(pred_labels, gt_labels, gt_loc_mask=None, num_classes=4):
    """
    Computes Class-wise F1-scores, Weighted Damage F1, and Confusion Matrix.
    When gt_loc_mask is provided, evaluates damage classification ONLY inside building footprints.
    """
    pred_flat = pred_labels.flatten()
    gt_flat = gt_labels.flatten()

    # Filter to valid classes and building footprints
    if gt_loc_mask is not None:
        loc_flat = (gt_loc_mask > 0).flatten()
        mask = loc_flat & (gt_flat >= 0) & (gt_flat < num_classes)
    else:
        mask = (gt_flat >= 0) & (gt_flat < num_classes)

    pred_valid = pred_flat[mask]
    gt_valid = gt_flat[mask]

    if len(gt_valid) == 0:
        pred_valid = pred_flat
        gt_valid = gt_flat

    precision, recall, f1_scores, support = precision_recall_fscore_support(
        gt_valid, pred_valid, labels=list(range(num_classes)), zero_division=0
    )

    # Official xView2 Metric: Harmonic Mean of the 4 Damage Class F1 Scores
    # H = 4 / sum(1 / (F1_c + 1e-6))
    inv_sum = sum(1.0 / (f1 + 1e-6) for f1 in f1_scores)
    harmonic_f1 = float(num_classes / inv_sum) if inv_sum > 0 else 0.0

    weighted_f1 = float(np.average(f1_scores, weights=support)) if support.sum() > 0 else 0.0
    cm = confusion_matrix(gt_valid, pred_valid, labels=list(range(num_classes)))

    metrics = {
        "Undamaged": float(f1_scores[0]),
        "Minor_Dmg": float(f1_scores[1]),
        "Major_Dmg": float(f1_scores[2]),
        "Destroyed": float(f1_scores[3]),
        "Harmonic_F1_damage": harmonic_f1,
        "Weighted_F1_damage": weighted_f1,
        "Confusion_Matrix": cm.tolist()
    }
    return metrics

def compute_overall_score(f1_loc, f1_damage, alpha=0.3, beta=0.7):
    """
    Official xBD / xView2 evaluation metric:
    Score = 0.3 * Localization_F1 + 0.7 * Damage_F1 (Harmonic Mean)
    """
    return float(alpha * f1_loc + beta * f1_damage)
