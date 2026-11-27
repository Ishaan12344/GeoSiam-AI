import os
import io
import time
import base64
import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import CHECKPOINT_DIR, IMAGE_SIZE, DAMAGE_CLASSES, DAMAGE_COLORS
from models.convnext_siamese import ConvNeXtSiameseBDA

app = FastAPI(
    title="GeoSiam AI — Building Damage Assessment",
    description="Pre- and Post-Disaster Building Damage Assessment using ConvNeXt-Tiny Siamese U-Net",
    version="2.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
os.makedirs(STATIC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model Initialization (Loaded once at startup)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_checkpoint_path():
    candidates = [
        os.path.join(CHECKPOINT_DIR, "ConvNeXt_Siamese_best.pth"),
        os.path.join(BASE_DIR, "ConvNeXt_Siamese_best.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

CKPT_PATH = resolve_checkpoint_path()
MODEL = None
MODEL_PARAMS = 0

def load_model():
    global MODEL, MODEL_PARAMS
    if CKPT_PATH is None:
        raise RuntimeError("ConvNeXt_Siamese_best.pth not found in outputs/checkpoints/ or project root.")
    model = ConvNeXtSiameseBDA(num_classes=4, pretrained=False).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    MODEL = model
    MODEL_PARAMS = sum(p.numel() for p in model.parameters())
    print(f"[GeoSiam AI] Loaded ConvNeXtSiameseBDA from {CKPT_PATH} on {DEVICE} ({MODEL_PARAMS:,} params)")

@app.on_event("startup")
async def startup_event():
    load_model()

# ---------------------------------------------------------------------------
# Image Encoding & Post-Processing Helpers
# ---------------------------------------------------------------------------
def numpy_rgb_to_b64(img_uint8: np.ndarray) -> str:
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def preprocess_pil(pil_img: Image.Image, target_size=(512, 512)):
    resized = pil_img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    arr_uint8 = np.array(resized, dtype=np.uint8)
    arr_norm = arr_uint8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr_norm).permute(2, 0, 1).unsqueeze(0)
    return arr_uint8, tensor


def build_damage_overlay(post_uint8: np.ndarray, dam_mask: np.ndarray, loc_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = post_uint8.copy()
    colored = np.zeros_like(post_uint8)
    valid_any = np.zeros(loc_mask.shape, dtype=bool)
    for cls_idx, rgb in DAMAGE_COLORS.items():
        m = (loc_mask > 0) & (dam_mask == cls_idx)
        if np.any(m):
            colored[m] = rgb
            valid_any |= m
    if np.any(valid_any):
        blended = cv2.addWeighted(colored, alpha, overlay, 1.0 - alpha, 0)
        overlay[valid_any] = blended[valid_any]
    return overlay


def build_confidence_heatmap(conf_map: np.ndarray, loc_mask: np.ndarray) -> np.ndarray:
    conf_uint8 = np.clip(conf_map * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    # Dim background slightly to highlight building footprints
    bg_mask = loc_mask == 0
    heat_rgb[bg_mask] = (heat_rgb[bg_mask].astype(np.float32) * 0.35).astype(np.uint8)
    return heat_rgb


def build_bitemporal_change_map(pre_uint8: np.ndarray, post_uint8: np.ndarray) -> np.ndarray:
    diff = np.abs(pre_uint8.astype(np.float32) - post_uint8.astype(np.float32)).mean(axis=2)
    diff_norm = np.clip((diff / (diff.max() + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def analyze_buildings(post_uint8: np.ndarray, loc_mask: np.ndarray, dam_mask: np.ndarray, conf_map: np.ndarray, min_area: int = 50):
    """Extract individual building footprints using connected components."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(loc_mask.astype(np.uint8), connectivity=8)
    annotated = post_uint8.copy()
    buildings = []

    priority_map = {
        0: "SAFE",
        1: "MODERATE",
        2: "HIGH",
        3: "CRITICAL"
    }

    b_counter = 0
    for comp_id in range(1, num_labels):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        b_counter += 1
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        cx, cy = int(centroids[comp_id][0]), int(centroids[comp_id][1])

        comp_mask = (labels == comp_id)
        cls_vals = dam_mask[comp_mask]
        counts = np.bincount(cls_vals, minlength=4)
        dom_cls = int(np.argmax(counts))
        mean_conf = float(conf_map[comp_mask].mean() * 100.0)

        color = DAMAGE_COLORS[dom_cls]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label_txt = f"B-{b_counter:02d}"
        cv2.putText(annotated, label_txt, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        buildings.append({
            "id": label_txt,
            "damage_class": DAMAGE_CLASSES[dom_cls],
            "damage_code": dom_cls,
            "priority": priority_map[dom_cls],
            "area_px": area,
            "est_area_m2": round(area * 0.25, 1),  # ~0.5m GSD satellite resolution
            "confidence": round(mean_conf, 1),
            "bbox": [x, y, w, h],
            "centroid": [cx, cy]
        })

    # Sort by severity descending, then area descending
    buildings.sort(key=lambda b: (b["damage_code"], b["area_px"]), reverse=True)
    return annotated, buildings


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Frontend not found. Please ensure static/index.html exists.</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/health")
async def health_check():
    return {
        "status": "ready" if MODEL is not None else "error",
        "architecture": "ConvNeXt-Tiny Siamese Dual-Stream U-Net",
        "checkpoint": os.path.basename(CKPT_PATH) if CKPT_PATH else None,
        "parameters": MODEL_PARAMS,
        "device": str(DEVICE),
        "input_resolution": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "sample_available": os.path.exists(os.path.join(SAMPLES_DIR, "test_pre.jpg")) and os.path.exists(os.path.join(SAMPLES_DIR, "test_post.jpg"))
    }


@app.post("/api/analyze")
async def analyze_disaster(
    pre_image: UploadFile = File(None),
    post_image: UploadFile = File(None),
    use_sample: str = Form("false"),
    loc_threshold: float = Form(0.5),
    overlay_alpha: float = Form(0.55),
    min_area: int = Form(50),
):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    t_start = time.time()

    if use_sample.lower() == "true":
        pre_path = os.path.join(SAMPLES_DIR, "test_pre.jpg")
        post_path = os.path.join(SAMPLES_DIR, "test_post.jpg")
        if not (os.path.exists(pre_path) and os.path.exists(post_path)):
            raise HTTPException(status_code=404, detail="Sample images not found in samples/ folder.")
        pre_pil = Image.open(pre_path)
        post_pil = Image.open(post_path)
    else:
        if pre_image is None or post_image is None:
            raise HTTPException(status_code=400, detail="Please upload both Pre-Disaster and Post-Disaster images.")
        pre_bytes = await pre_image.read()
        post_bytes = await post_image.read()
        pre_pil = Image.open(io.BytesIO(pre_bytes))
        post_pil = Image.open(io.BytesIO(post_bytes))

    pre_uint8, pre_tensor = preprocess_pil(pre_pil, IMAGE_SIZE)
    post_uint8, post_tensor = preprocess_pil(post_pil, IMAGE_SIZE)

    with torch.no_grad():
        loc_logits, dam_logits = MODEL(pre_tensor.to(DEVICE), post_tensor.to(DEVICE))
        loc_probs = torch.sigmoid(loc_logits).squeeze().cpu().numpy()
        dam_probs = torch.softmax(dam_logits.squeeze(0), dim=0).cpu().numpy()

    loc_mask = (loc_probs > loc_threshold).astype(np.uint8)
    dam_mask = np.argmax(dam_probs, dim=0) if isinstance(dam_probs, torch.Tensor) else np.argmax(dam_probs, axis=0)
    conf_map = np.max(dam_probs, axis=0)

    # Generate all visual layers
    damage_overlay = build_damage_overlay(post_uint8, dam_mask, loc_mask, alpha=overlay_alpha)
    loc_vis = np.stack([loc_mask * 255] * 3, axis=-1).astype(np.uint8)
    conf_heatmap = build_confidence_heatmap(conf_map, loc_mask)
    change_map = build_bitemporal_change_map(pre_uint8, post_uint8)
    bbox_annotated, buildings = analyze_buildings(post_uint8, loc_mask, dam_mask, conf_map, min_area=min_area)

    # Pixel-level stats inside building footprints
    total_pixels = int(IMAGE_SIZE[0] * IMAGE_SIZE[1])
    bldg_pixels = int(loc_mask.sum())
    pixel_breakdown = {}
    for cls_idx, cls_name in DAMAGE_CLASSES.items():
        cnt = int(np.sum((loc_mask > 0) & (dam_mask == cls_idx)))
        pct = round((cnt / bldg_pixels * 100.0), 2) if bldg_pixels > 0 else 0.0
        pixel_breakdown[cls_name] = {"pixels": cnt, "percent": pct}

    # Building-level stats
    bldg_counts = {name: 0 for name in DAMAGE_CLASSES.values()}
    for b in buildings:
        bldg_counts[b["damage_class"]] += 1

    total_bldgs = len(buildings)
    damaged_bldgs = total_bldgs - bldg_counts["No Damage"]
    affected_pct = round((damaged_bldgs / total_bldgs * 100.0), 1) if total_bldgs > 0 else 0.0

    # Weighted Severity Index (0 - 100)
    if total_bldgs > 0:
        weighted_sum = (
            bldg_counts["No Damage"] * 0.0 +
            bldg_counts["Minor Damage"] * 33.3 +
            bldg_counts["Major Damage"] * 66.7 +
            bldg_counts["Destroyed"] * 100.0
        )
        severity_index = round(weighted_sum / total_bldgs, 1)
    else:
        severity_index = 0.0

    mean_confidence = round(float(conf_map[loc_mask > 0].mean() * 100.0), 1) if bldg_pixels > 0 else 0.0
    inference_ms = round((time.time() - t_start) * 1000.0, 1)

    return JSONResponse({
        "inference_ms": inference_ms,
        "summary": {
            "total_buildings": total_bldgs,
            "damaged_buildings": damaged_bldgs,
            "affected_percent": affected_pct,
            "building_coverage_percent": round((bldg_pixels / total_pixels) * 100.0, 2),
            "severity_index": severity_index,
            "mean_confidence": mean_confidence,
            "building_counts": bldg_counts,
            "pixel_breakdown": pixel_breakdown,
        },
        "buildings": buildings,
        "images": {
            "pre": numpy_rgb_to_b64(pre_uint8),
            "post": numpy_rgb_to_b64(post_uint8),
            "damage_overlay": numpy_rgb_to_b64(damage_overlay),
            "localization": numpy_rgb_to_b64(loc_vis),
            "confidence": numpy_rgb_to_b64(conf_heatmap),
            "change_map": numpy_rgb_to_b64(change_map),
            "bbox_triage": numpy_rgb_to_b64(bbox_annotated),
        }
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
