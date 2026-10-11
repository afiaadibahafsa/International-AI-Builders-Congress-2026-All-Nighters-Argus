"""
model_loader.py
----------------
Loads your best.pt file.  Run inspect_model.py FIRST to find out which
case applies, then edit the CONFIG section below.

Supports two kinds of anomaly detector (set MODEL_TYPE):

  "classifier"    -> model outputs logits/probabilities over classes
                     (e.g. ["normal", "anomaly"] or multi-class).
                     Explained with Grad-CAM.

  "reconstruction" -> model outputs a reconstructed image (autoencoder);
                      anomaly = high pixel-wise reconstruction error.
                      Explained with a per-pixel error heatmap.

  "yolo"          -> ultralytics YOLOv8/v9/v10 detection or classification
                     model (best.pt saved by ultralytics trainer).
                     Explained with Grad-CAM on the backbone's last conv.
                     Requires: pip install ultralytics

NOTE: Everything runs fully offline once dependencies are installed.
      No internet connection is required at inference time.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# =============================================================================
# CONFIG — edit this section for your model
# =============================================================================

MODEL_PATH   = "best.pt"          # sits in the same folder as this script
VIDEO_PATH   = "co.mp4"          # sits in the same folder as this script
MODEL_TYPE   = "yolo"             # "classifier" | "reconstruction" | "yolo"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Only needed for classifier / reconstruction (NOT for yolo) -------------
# If inspect_model.py says FULL MODEL OBJECT -> leave MODEL_CLASS = None.
# If it says STATE_DICT or CHECKPOINT DICT -> define your architecture here.
#
# Example (uncomment and adapt):
#
# class YourModel(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         # ... your layers ...
#     def forward(self, x):
#         # ... your forward pass ...
#         return x
#
# MODEL_CLASS  = YourModel
# MODEL_KWARGS = {"num_classes": 2}

MODEL_CLASS  = None
MODEL_KWARGS = {}

# Key in the checkpoint dict that holds the actual state_dict.
# Common: "state_dict", "model_state_dict", "model".
CHECKPOINT_KEY = "state_dict"

# ---- Classifier settings ----------------------------------------------------
# Class names in output-index order.
CLASS_NAMES = ["normal", "anomaly"]

# Name of the last convolutional layer for Grad-CAM.
# None = auto-detect the last Conv2d (works for most plain CNNs).
GRAD_CAM_TARGET_LAYER = None

# ---- Preprocessing (classifier / reconstruction) ----------------------------
# MUST match the transforms used during training.
INPUT_SIZE = (224, 224)             # (H, W)
NORM_MEAN  = [0.485, 0.456, 0.406] # ImageNet defaults
NORM_STD   = [0.229, 0.224, 0.225]

# ---- Reconstruction threshold -----------------------------------------------
# Calibrate this against your validation set.
RECON_THRESHOLD = 0.5

# =============================================================================


def load_model():
    """Load the model per CONFIG.  Returns a model ready for inference."""

    # --- YOLO path -----------------------------------------------------------
    if MODEL_TYPE == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "MODEL_TYPE='yolo' requires the ultralytics package.\n"
                "Install it offline (no internet needed after the first install):\n"
                "  pip install ultralytics"
            )
        model = YOLO(MODEL_PATH)
        # Expose the underlying nn.Module so Grad-CAM can attach hooks later.
        model.pt_model = model.model   # ultralytics wraps it here
        return model

    # --- Torch path ----------------------------------------------------------
    raw = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(raw, nn.Module):
        model = raw

    elif isinstance(raw, dict):
        if MODEL_CLASS is None:
            raise RuntimeError(
                "Loaded a dict (state_dict or checkpoint dict), but MODEL_CLASS "
                "is not set.  Run inspect_model.py, then define your architecture "
                "class in the CONFIG section of model_loader.py."
            )
        state_dict = raw[CHECKPOINT_KEY] if CHECKPOINT_KEY in raw else raw
        model = MODEL_CLASS(**MODEL_KWARGS)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning — missing keys when loading weights: {missing}")
        if unexpected:
            print(f"Warning — unexpected keys when loading weights: {unexpected}")

    else:
        raise RuntimeError(f"Unrecognised object type in .pt file: {type(raw)}")

    model.to(DEVICE)
    model.eval()
    return model


def find_last_conv_layer(model: nn.Module):
    """Auto-detect the last Conv2d layer (used for Grad-CAM)."""
    last_conv, last_name = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv, last_name = module, name
    if last_conv is None:
        raise RuntimeError(
            "No Conv2d layer found automatically.  Set GRAD_CAM_TARGET_LAYER "
            "in model_loader.py to the name of your target conv layer."
        )
    return last_name, last_conv


def get_preprocess_transform():
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def load_and_preprocess_image(image_path: str):
    """
    Returns:
        input_tensor  — (1, C, H, W) on DEVICE
        display_image — HxWx3 uint8 RGB numpy array (resized to INPUT_SIZE)
    """
    pil_image = Image.open(image_path).convert("RGB")
    transform = get_preprocess_transform()
    tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    display_array = np.array(pil_image.resize((INPUT_SIZE[1], INPUT_SIZE[0])))
    return tensor, display_array


def get_target_layer(model: nn.Module):
    """Return the Conv2d layer to attach Grad-CAM hooks to."""
    if GRAD_CAM_TARGET_LAYER is not None:
        for name, module in model.named_modules():
            if name == GRAD_CAM_TARGET_LAYER:
                return module
        raise RuntimeError(f"Layer '{GRAD_CAM_TARGET_LAYER}' not found in model.")
    _, layer = find_last_conv_layer(model)
    return layer
