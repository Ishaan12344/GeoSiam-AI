# GeoSiam AI — Pre- & Post-Disaster Building Damage Assessment

Deep learning framework and interactive web command center for **Building Localization** and **4-Class Building Damage Assessment** from bi-temporal satellite imagery (Extended Building Damage / xBD dataset).

## Architecture
- **ConvNeXt-Tiny Siamese Dual-Stream U-Net** (`models/convnext_siamese.py`) — shared-weight Siamese encoder with absolute difference feature fusion $|F_{\text{pre}} - F_{\text{post}}|$ and dual localization + damage decoders.
- Also includes comparative baselines: **ResNeXt-50 Siamese** (`models/resnext_siamese.py`) and **HybridSiam-BDA Lite** (`models/hybrid_siambda.py`).

## Web Application (`app.py` + `static/index.html`)
- **Interactive Before/After Swipe Comparator** across Pre-Disaster, Post-Disaster, AI Damage Overlay, Numbered Triage Boxes, Confidence Heatmap, and Bi-Temporal Change Map.
- **Automated Emergency Triage & Analytics** with Severity Index (0–100), rescue priority alerts, and Chart.js visualizations.
- **Building-by-Building Inventory** using connected-component extraction (`B-01`, `B-02`, ...), area estimation, and JSON/CSV export.

## Quick Start

1. Place your trained `ConvNeXt_Siamese_best.pth` checkpoint in `outputs/checkpoints/`.
2. Install dependencies:
   ```bash
   pip install torch torchvision fastapi uvicorn opencv-python pillow numpy albumentations scikit-learn matplotlib
   ```
3. Launch the web app:
   ```bash
   python -m uvicorn app:app --host 127.0.0.1 --port 8000
   ```
   Or double-click `launch_webapp.bat` on Windows and open `http://127.0.0.1:8000`.

## Training
- **Kaggle (ConvNeXt-only)**: `ConvNeXt_Siamese_Kaggle_Training.ipynb`
- **Colab**: `EBD_Disaster_Damage_Assessment_Colab.ipynb`
- **CLI**: `python train.py` / `python evaluate.py`
