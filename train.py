import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import json
import gc
import contextlib
import torch
from torch.utils.data import DataLoader
import numpy as np

# -----------------------------------------------------------------------
# Universal Mixed-Precision helpers — compatible with PyTorch 1.6 to 2.x+
# -----------------------------------------------------------------------
try:
    from torch.cuda.amp import GradScaler as _GradScaler
    _GRADSCALER_AVAIL = True
except ImportError:
    _GRADSCALER_AVAIL = False

def _make_scaler():
    if torch.cuda.is_available():
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            return torch.amp.GradScaler('cuda')
        elif _GRADSCALER_AVAIL:
            return _GradScaler()
    class _NoOpScaler:
        def scale(self, loss): return loss
        def step(self, opt): opt.step()
        def update(self): pass
    return _NoOpScaler()

def _autocast_ctx():
    """Return an autocast context compatible with all PyTorch versions."""
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda')
    return torch.cuda.amp.autocast(enabled=True)

# -----------------------------------------------------------------------

from config import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, CLASS_WEIGHTS,
    CHECKPOINT_DIR, OUTPUT_DIR, DATA_DIR, ACTIVE_DATA_DIR
)
from dataset import EBDDataset
from models.resnext_siamese import ResNeXtSiameseBDA
from models.convnext_siamese import ConvNeXtSiameseBDA
from models.hybrid_siambda import HybridSiamBDA
from losses import JointBDALoss
from metrics import compute_localization_metrics, compute_damage_metrics, compute_overall_score


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def train_model(model_name, model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE,
                patience=5):
    """
    Train a single BDA model with proper regularization and early stopping.

    Args:
        patience: Number of epochs without improvement before stopping early.
                  Set to 0 or None to disable early stopping.
    """
    print(f"\n=======================================================")
    print(f"🚀 Starting Training for Model Backbone: {model_name}")
    print(f"=======================================================")

    clear_gpu_memory()
    model = model.to(device)
    criterion = JointBDALoss(class_weights=CLASS_WEIGHTS, lambda_damage=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = _make_scaler()

    best_overall_score = 0.0
    best_loc_metrics = {"F1_loc": 0.0, "IoU": 0.0}
    best_dam_metrics = {"Undamaged": 0.0, "Minor_Dmg": 0.0, "Major_Dmg": 0.0, "Destroyed": 0.0}
    history = {"train_loss": [], "val_iou": [], "val_f1_loc": [], "val_f1_damage": [], "val_score": []}

    # Early stopping state
    epochs_without_improvement = 0
    use_early_stopping = patience and patience > 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (pre_img, post_img, loc_gt, dam_gt) in enumerate(train_loader, 1):
            pre_img, post_img = pre_img.to(device), post_img.to(device)
            loc_gt, dam_gt = loc_gt.to(device), dam_gt.to(device)

            optimizer.zero_grad()

            with _autocast_ctx():
                loc_pred, dam_pred = model(pre_img, post_img)
                loss, _, _ = criterion(loc_pred, loc_gt, dam_pred, dam_gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                elapsed = time.time() - start_time
                print(f"[{model_name}] Epoch [{epoch:02d}/{epochs:02d}] | Batch [{batch_idx:04d}/{len(train_loader):04d}] | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

        scheduler.step()
        epoch_loss = running_loss / max(1, len(train_loader))

        # Memory-Efficient Streaming Validation Step (Zero RAM accumulation)
        model.eval()
        running_tp_loc, running_fp_loc, running_fn_loc = 0, 0, 0
        running_cm = np.zeros((4, 4), dtype=np.int64)

        with torch.no_grad():
            for val_idx, (pre_img, post_img, loc_gt, dam_gt) in enumerate(val_loader, 1):
                pre_img, post_img = pre_img.to(device), post_img.to(device)
                with _autocast_ctx():
                    loc_pred, dam_pred = model(pre_img, post_img)

                # 1. Streaming Localization Metrics
                pred_loc_bin = (torch.sigmoid(loc_pred).squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
                gt_loc_bin = (loc_gt.squeeze(1) > 0).numpy().astype(np.uint8)

                tp_l = np.logical_and(pred_loc_bin, gt_loc_bin).sum()
                running_tp_loc += tp_l
                running_fp_loc += pred_loc_bin.sum() - tp_l
                running_fn_loc += gt_loc_bin.sum() - tp_l

                # 2. Streaming Damage Confusion Matrix (inside building footprints)
                pred_dam_cls = torch.argmax(dam_pred, dim=1).cpu().numpy()
                gt_dam_cls = dam_gt.numpy()

                valid_mask = (gt_loc_bin > 0) & (gt_dam_cls >= 0) & (gt_dam_cls < 4)
                if valid_mask.any():
                    pv = pred_dam_cls[valid_mask].astype(np.int64)
                    gv = gt_dam_cls[valid_mask].astype(np.int64)
                    running_cm += np.bincount(4 * gv + pv, minlength=16).reshape(4, 4)

                # Cap validation to 400 batches during training epochs for high speed
                if val_idx >= 400:
                    break

        # Calculate final localization F1 and IoU
        loc_union = running_tp_loc + running_fp_loc + running_fn_loc
        val_iou = float(running_tp_loc / (loc_union + 1e-6))
        val_f1_loc = float(2 * running_tp_loc / (2 * running_tp_loc + running_fp_loc + running_fn_loc + 1e-6))

        # Calculate class-wise damage F1 and harmonic mean
        tp_d = np.diag(running_cm).astype(np.float64)
        fp_d = (running_cm.sum(axis=0) - tp_d).astype(np.float64)
        fn_d = (running_cm.sum(axis=1) - tp_d).astype(np.float64)
        f1_damage = (2.0 * tp_d) / (2.0 * tp_d + fp_d + fn_d + 1e-6)

        # Also compute precision and recall per damage class
        prec_damage = tp_d / (tp_d + fp_d + 1e-6)
        rec_damage = tp_d / (tp_d + fn_d + 1e-6)

        inv_sum = sum(1.0 / (f1 + 1e-6) for f1 in f1_damage)
        harmonic_f1_damage = float(4.0 / inv_sum) if inv_sum > 0 else 0.0
        overall_score = compute_overall_score(val_f1_loc, harmonic_f1_damage)

        history["train_loss"].append(epoch_loss)
        history["val_iou"].append(val_iou)
        history["val_f1_loc"].append(val_f1_loc)
        history["val_f1_damage"].append(harmonic_f1_damage)
        history["val_score"].append(overall_score)

        elapsed = time.time() - start_time
        print(f"\n✨ Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Loss: {epoch_loss:.4f} | Overall Score: {overall_score:.4f} | Loc_F1: {val_f1_loc:.4f} | Undamaged: {f1_damage[0]:.4f} | Minor: {f1_damage[1]:.4f} | Major: {f1_damage[2]:.4f} | Destroyed: {f1_damage[3]:.4f}")
        print(f"   Precision → Undamaged: {prec_damage[0]:.4f} | Minor: {prec_damage[1]:.4f} | Major: {prec_damage[2]:.4f} | Destroyed: {prec_damage[3]:.4f}")
        print(f"   Recall    → Undamaged: {rec_damage[0]:.4f} | Minor: {rec_damage[1]:.4f} | Major: {rec_damage[2]:.4f} | Destroyed: {rec_damage[3]:.4f}\n")

        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_loc_metrics = {"F1_loc": val_f1_loc, "IoU": val_iou}
            best_dam_metrics = {
                "Undamaged": float(f1_damage[0]),
                "Minor_Dmg": float(f1_damage[1]),
                "Major_Dmg": float(f1_damage[2]),
                "Destroyed": float(f1_damage[3])
            }
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth"))
            epochs_without_improvement = 0
            print(f"💾 Saved new best checkpoint! (Score: {overall_score:.4f})")
        else:
            epochs_without_improvement += 1
            if use_early_stopping:
                print(f"⏳ No improvement for {epochs_without_improvement}/{patience} epochs.")
                if epochs_without_improvement >= patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch}! Best score: {best_overall_score:.4f}")
                    break

    # Save final performance history
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    del optimizer, criterion, scheduler, scaler
    clear_gpu_memory()

    return history, best_overall_score, best_loc_metrics, best_dam_metrics


def run_comparative_experiment(data_dir=ACTIVE_DATA_DIR, epochs=30, num_train=None, num_val=None):
    """
    Run the 3-way comparative benchmark.
    
    Args:
        num_train: Number of training samples. None = use all available real data.
        num_val:   Number of validation samples. None = use all available real data.
    """
    clear_gpu_memory()
    print(f"\n{'='*70}")
    print(f"📋 EXPERIMENT CONFIGURATION")
    print(f"{'='*70}")
    print(f"Data Directory  : {data_dir}")
    print(f"Epochs          : {epochs}")
    print(f"Batch Size      : {BATCH_SIZE}")
    print(f"Learning Rate   : {LEARNING_RATE}")
    print(f"Early Stopping  : patience=5")
    print(f"Augmentation    : ON (train) / OFF (val)")
    print(f"{'='*70}\n")

    print(f"Creating Datasets & DataLoaders from: {data_dir} ...")
    train_dataset = EBDDataset(
        data_dir=data_dir, split="train",
        num_samples=num_train,
        use_augmentation=True,
        split_by_event=True
    )
    val_dataset = EBDDataset(
        data_dir=data_dir, split="val",
        num_samples=num_val,
        use_augmentation=False,
        split_by_event=True
    )

    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    print(f"\n📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # 1. Train ResNeXt-50 Backbone Siamese Model
    resnext_model = ResNeXtSiameseBDA(pretrained=True)
    resnext_hist, resnext_score, resnext_loc, resnext_dam = train_model(
        "ResNeXt50_Siamese", resnext_model, train_loader, val_loader, epochs=epochs, patience=5
    )

    del resnext_model
    clear_gpu_memory()

    # 2. Train ConvNeXt-Tiny Backbone Siamese Model
    convnext_model = ConvNeXtSiameseBDA(pretrained=True)
    convnext_hist, convnext_score, convnext_loc, convnext_dam = train_model(
        "ConvNeXt_Siamese", convnext_model, train_loader, val_loader, epochs=epochs, patience=5
    )

    del convnext_model
    clear_gpu_memory()

    # 3. Train HybridSiam-BDA (Lite) Novel Siamese Model
    hybrid_model = HybridSiamBDA(pretrained=True)
    hybrid_hist, hybrid_score, hybrid_loc, hybrid_dam = train_model(
        "HybridSiam_BDA_Lite", hybrid_model, train_loader, val_loader, epochs=epochs, patience=5
    )

    del hybrid_model
    clear_gpu_memory()

    # Summary Report matching official EBD / xView2 Leaderboard Criteria
    print("\n=======================================================================================================================")
    print("🏆 FINAL EXTENDED BUILDING DAMAGE (EBD) COMPARATIVE BENCHMARK LEADERBOARD REPORT")
    print("=======================================================================================================================")
    print(f"Architecture              | F1 score (overall) | Localization F1 | Undamaged | Minor Dmg | Major Dmg | Destroyed")
    print(f"-----------------------------------------------------------------------------------------------------------------------")
    print(f"ResNeXt-50 Siamese UNet   | {resnext_score:.5f}            | {resnext_loc['F1_loc']:.5f}         | {resnext_dam['Undamaged']:.5f}   | {resnext_dam['Minor_Dmg']:.5f}   | {resnext_dam['Major_Dmg']:.5f}   | {resnext_dam['Destroyed']:.5f}")
    print(f"ConvNeXt-Tiny Siamese UNet| {convnext_score:.5f}            | {convnext_loc['F1_loc']:.5f}         | {convnext_dam['Undamaged']:.5f}   | {convnext_dam['Minor_Dmg']:.5f}   | {convnext_dam['Major_Dmg']:.5f}   | {convnext_dam['Destroyed']:.5f}")
    print(f"HybridSiam-BDA (Lite)     | {hybrid_score:.5f}            | {hybrid_loc['F1_loc']:.5f}         | {hybrid_dam['Undamaged']:.5f}   | {hybrid_dam['Minor_Dmg']:.5f}   | {hybrid_dam['Major_Dmg']:.5f}   | {hybrid_dam['Destroyed']:.5f}")
    print("=======================================================================================================================\n")

    return {
        "ResNeXt50": {"Score": resnext_score, "F1_loc": resnext_loc["F1_loc"], "Undamaged": resnext_dam["Undamaged"], "Minor_Dmg": resnext_dam["Minor_Dmg"], "Major_Dmg": resnext_dam["Major_Dmg"], "Destroyed": resnext_dam["Destroyed"], "IoU": resnext_loc["IoU"]},
        "ConvNeXt": {"Score": convnext_score, "F1_loc": convnext_loc["F1_loc"], "Undamaged": convnext_dam["Undamaged"], "Minor_Dmg": convnext_dam["Minor_Dmg"], "Major_Dmg": convnext_dam["Major_Dmg"], "Destroyed": convnext_dam["Destroyed"], "IoU": convnext_loc["IoU"]},
        "HybridSiam": {"Score": hybrid_score, "F1_loc": hybrid_loc["F1_loc"], "Undamaged": hybrid_dam["Undamaged"], "Minor_Dmg": hybrid_dam["Minor_Dmg"], "Major_Dmg": hybrid_dam["Major_Dmg"], "Destroyed": hybrid_dam["Destroyed"], "IoU": hybrid_loc["IoU"]}
    }


def train_convnext_only(data_dir=ACTIVE_DATA_DIR, epochs=20, patience=5, num_train=None, num_val=None):
    """
    Dedicated training for ConvNeXt-Tiny Siamese Building Damage Assessment model.
    Runs faster, focused only on ConvNeXt-Tiny backbone with early stopping & augmentations.
    """
    clear_gpu_memory()
    print(f"\n{'='*70}")
    print(f"🚀 CONVNEXT-TINY SIAMESE BDA TRAINING ENGINE")
    print(f"{'='*70}")
    print(f"Data Directory  : {data_dir}")
    print(f"Epochs          : {epochs}")
    print(f"Batch Size      : {BATCH_SIZE}")
    print(f"Learning Rate   : {LEARNING_RATE}")
    print(f"Early Stopping  : patience={patience}")
    print(f"Augmentation    : ON (train) / OFF (val)")
    print(f"{'='*70}\n")

    print(f"Creating Datasets & DataLoaders from: {data_dir} ...")
    train_dataset = EBDDataset(
        data_dir=data_dir, split="train",
        num_samples=num_train,
        use_augmentation=True,
        split_by_event=True
    )
    val_dataset = EBDDataset(
        data_dir=data_dir, split="val",
        num_samples=num_val,
        use_augmentation=False,
        split_by_event=True
    )

    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(f"\n📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    convnext_model = ConvNeXtSiameseBDA(pretrained=True)
    convnext_hist, convnext_score, convnext_loc, convnext_dam = train_model(
        "ConvNeXt_Siamese", convnext_model, train_loader, val_loader, epochs=epochs, patience=patience
    )

    del convnext_model
    clear_gpu_memory()

    print("\n" + "=" * 70)
    print("🏆 FINAL CONVNEXT-TINY SIAMESE BDA EVALUATION REPORT")
    print("=" * 70)
    print(f"Overall EBD Score       : {convnext_score:.5f}")
    print(f"Localization F1-Score   : {convnext_loc['F1_loc']:.5f} (IoU: {convnext_loc['IoU']:.5f})")
    print(f"Undamaged Building F1   : {convnext_dam['Undamaged']:.5f}")
    print(f"Minor Damage F1         : {convnext_dam['Minor_Dmg']:.5f}")
    print(f"Major Damage F1         : {convnext_dam['Major_Dmg']:.5f}")
    print(f"Destroyed Building F1   : {convnext_dam['Destroyed']:.5f}")
    print("=" * 70 + "\n")

    return {
        "ConvNeXt": {
            "Score": convnext_score,
            "F1_loc": convnext_loc["F1_loc"],
            "IoU": convnext_loc["IoU"],
            **convnext_dam
        }
    }


if __name__ == "__main__":
    train_convnext_only()
