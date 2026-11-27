"""
Building Damage Assessment (BDA) - Checkpoint Inference & Evaluation Script
Allows inspecting model outputs (visual overlay + performance metrics) from saved .pth checkpoints.
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DEVICE, CHECKPOINT_DIR, VISUALS_DIR, OUTPUT_DIR,
    ACTIVE_DATA_DIR, DAMAGE_CLASSES
)
from dataset import EBDDataset
from models.convnext_siamese import ConvNeXtSiameseBDA
from models.hybrid_siambda import HybridSiamBDA
from models.resnext_siamese import ResNeXtSiameseBDA
from metrics import (
    compute_localization_metrics,
    compute_damage_metrics,
    compute_overall_score
)
from visualize import plot_bda_results


def find_checkpoint(model_name=None, custom_path=None):
    """Finds checkpoint file automatically from common locations."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    search_dirs = [
        CHECKPOINT_DIR,
        os.path.dirname(os.path.abspath(__file__)),
        OUTPUT_DIR
    ]

    names_to_try = []
    if model_name:
        names_to_try.extend([
            f"{model_name}_best.pth",
            f"{model_name}_best.zip",
            f"{model_name}.pth"
        ])
    else:
        # Default order of preference
        names_to_try.extend([
            "ConvNeXt_Siamese_best.pth",
            "ConvNeXt_Siamese_best.zip",
            "HybridSiam_BDA_Lite_best.pth",
            "HybridSiam_BDA_Lite_best.zip",
            "ResNeXt50_Siamese_best.pth",
            "ResNeXt50_Siamese_best.zip"
        ])

    for d in search_dirs:
        for name in names_to_try:
            full_path = os.path.join(d, name)
            if os.path.exists(full_path):
                return full_path

    return None


def get_model_instance(model_name):
    """Instantiates model architecture without downloading pretrained weights."""
    name_lower = model_name.lower()
    if "convnext" in name_lower:
        return ConvNeXtSiameseBDA(pretrained=False)
    elif "hybrid" in name_lower:
        return HybridSiamBDA(pretrained=False)
    elif "resnext" in name_lower:
        return ResNeXtSiameseBDA(pretrained=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from ConvNeXt_Siamese, HybridSiam_BDA_Lite, ResNeXt50_Siamese")


def infer_and_evaluate(
    checkpoint_path,
    model_name="ConvNeXt_Siamese",
    num_samples=10,
    save_vis=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "=" * 70)
    print(f"🔍 EVALUATION & OUTPUT INSPECTION")
    print(f"=" * 70)
    print(f"• Hardware Device : {device}")
    print(f"• Checkpoint Path : {checkpoint_path}")
    print(f"• Model           : {model_name}")

    # 1. Instantiate & load weights
    model = get_model_instance(model_name).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print("✅ Model weights loaded successfully into memory!\n")

    # 2. Load evaluation dataset
    print(f"Loading validation dataset from: {ACTIVE_DATA_DIR} ...")
    val_dataset = EBDDataset(data_dir=ACTIVE_DATA_DIR, split="val", num_samples=num_samples)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    total_iou = 0.0
    total_f1_loc = 0.0
    dam_f1_accum = np.zeros(4)
    first_sample_saved = False

    os.makedirs(VISUALS_DIR, exist_ok=True)

    with torch.no_grad():
        for idx, (pre_img, post_img, loc_gt, dam_gt) in enumerate(val_loader):
            pre_dev = pre_img.to(device)
            post_dev = post_img.to(device)

            # Model Forward Pass
            loc_logits, dam_logits = model(pre_dev, post_dev)

            # Predictions
            loc_prob = torch.sigmoid(loc_logits).squeeze().cpu().numpy()
            pred_loc = (loc_prob > 0.5).astype(np.float32)
            pred_dam = torch.argmax(dam_logits.squeeze(0), dim=0).cpu().numpy()

            gt_loc_np = loc_gt.squeeze().numpy()
            gt_dam_np = dam_gt.squeeze().numpy()

            # Calculate batch metrics
            loc_m = compute_localization_metrics(pred_loc, gt_loc_np)
            dam_m = compute_damage_metrics(pred_dam, gt_dam_np, gt_loc_mask=gt_loc_np)

            total_iou += loc_m["IoU"]
            total_f1_loc += loc_m["F1_loc"]
            dam_f1_accum += np.array([
                dam_m["Undamaged"],
                dam_m["Minor_Dmg"],
                dam_m["Major_Dmg"],
                dam_m["Destroyed"]
            ])

            # Save qualitative visual overlay for first 2 samples
            if save_vis and idx < 2:
                vis_path = os.path.join(VISUALS_DIR, f"{model_name}_prediction_sample_{idx+1}.png")
                plot_bda_results(
                    pre_img.squeeze(0).permute(1, 2, 0).numpy(),
                    post_img.squeeze(0).permute(1, 2, 0).numpy(),
                    gt_loc_np,
                    gt_dam_np,
                    pred_loc,
                    pred_dam,
                    save_path=vis_path
                )
                print(f"🖼️ [Sample {idx+1}] Saved 6-panel visualization overlay -> {vis_path}")
                first_sample_saved = True

    n = max(1, len(val_loader))
    avg_iou = total_iou / n
    avg_f1_loc = total_f1_loc / n
    avg_dam_f1 = dam_f1_accum / n

    inv_sum = sum(1.0 / (f + 1e-6) for f in avg_dam_f1)
    harmonic_f1_dam = float(4.0 / inv_sum) if inv_sum > 0 else 0.0
    overall_score = compute_overall_score(avg_f1_loc, harmonic_f1_dam)

    print("\n" + "=" * 70)
    print("📊 QUANTITATIVE OUTPUT & ACCURACY REPORT")
    print("=" * 70)
    print(f"Localization IoU           : {avg_iou:.4f}")
    print(f"Localization F1-Score      : {avg_f1_loc:.4f}")
    print("-" * 70)
    for c_idx, name in DAMAGE_CLASSES.items():
        print(f"Damage F1 [{name:<12}] : {avg_dam_f1[c_idx]:.4f}")
    print("-" * 70)
    print(f"Harmonic Damage F1         : {harmonic_f1_dam:.4f}")
    print(f"⭐ Overall EBD Score       : {overall_score:.4f}")
    print("=" * 70)
    print(f"✅ Check complete! Visualizations available in: {VISUALS_DIR}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDA Checkpoint Inference & Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint file")
    parser.add_argument("--model", type=str, default="ConvNeXt_Siamese", help="Model name (ConvNeXt_Siamese, HybridSiam_BDA_Lite, ResNeXt50_Siamese)")
    parser.add_argument("--samples", type=int, default=10, help="Number of validation samples to evaluate on")
    args = parser.parse_args()

    ckpt = find_checkpoint(model_name=args.model, custom_path=args.checkpoint)
    if not ckpt:
        print(f"❌ Error: Could not find checkpoint for {args.model}.")
        print("Please place your checkpoint in 'outputs/checkpoints/' or pass --checkpoint <path>")
    else:
        infer_and_evaluate(ckpt, model_name=args.model, num_samples=args.samples)
