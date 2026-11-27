import os
import torch

# Base Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Kaggle Output Handling (Kaggle saves files from /kaggle/working)
if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working/outputs"
else:
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
VISUALS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Dataset Directories (Kaggle Input > Colab Local VM > Google Drive > Project Data)
KAGGLE_INPUT_DIR = "/kaggle/input"
COLAB_LOCAL_DIR = "/content/data"
EBD_DRIVE_DIR = "/content/drive/MyDrive/EBD_dataset"

if os.path.exists(KAGGLE_INPUT_DIR) and len(os.listdir(KAGGLE_INPUT_DIR)) > 0:
    cand_ebd = [os.path.join(KAGGLE_INPUT_DIR, d) for d in os.listdir(KAGGLE_INPUT_DIR) if any(k in d.lower() for k in ["ebd", "dataset", "damage", "data"])]
    ACTIVE_DATA_DIR = cand_ebd[0] if cand_ebd else KAGGLE_INPUT_DIR
elif os.path.exists(COLAB_LOCAL_DIR) and len(os.listdir(COLAB_LOCAL_DIR)) > 0:
    ACTIVE_DATA_DIR = COLAB_LOCAL_DIR
elif os.path.exists(EBD_DRIVE_DIR) and len(os.listdir(EBD_DRIVE_DIR)) > 0:
    ACTIVE_DATA_DIR = EBD_DRIVE_DIR
else:
    ACTIVE_DATA_DIR = DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

# Hardware Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30

# EBD Dataset Specifications
# 0: Background, 1: No Damage, 2: Minor Damage, 3: Major Damage, 4: Destroyed
EBD_RAW_CLASSES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed"
}

# Building Damage Assessment Target Classes (Evaluated on building footprints)
DAMAGE_CLASSES = {
    0: "No Damage",
    1: "Minor Damage",
    2: "Major Damage",
    3: "Destroyed"
}

# Color Palette for Visualization (RGB)
DAMAGE_COLORS = {
    0: (46, 204, 113),   # Green - No Damage
    1: (241, 196, 15),   # Yellow - Minor Damage
    2: (230, 126, 34),   # Orange - Major Damage
    3: (231, 76, 60)     # Red - Destroyed
}

# Class Weights to counteract Class Imbalance (No Damage, Minor, Major, Destroyed)
CLASS_WEIGHTS = torch.tensor([1.0, 3.5, 4.0, 5.0])

# Benchmark Architectures to Compare
BENCHMARK_MODELS = [
    "ResNeXt50_Siamese",
    "ConvNeXt_Siamese",
    "HybridSiam_BDA_Lite"
]
