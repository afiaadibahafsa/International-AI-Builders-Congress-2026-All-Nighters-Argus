"""
infer.py
--------
Run anomaly detection + explanation on a single image or a folder of images.
Fully offline — no internet connection needed.

Usage
-----
  python infer.py --input path/to/image.jpg
  python infer.py --input path/to/folder/  --output results/
  python infer.py --input path/to/folder/  --output results/ --no-heatmap

Before running: edit the CONFIG section in model_loader.py
(MODEL_PATH, MODEL_TYPE, INPUT_SIZE, etc.)
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image as PILImage

import model_loader as cfg
from explain import GradCAM, reconstruction_error_map, overlay_heatmap, yolo_gradcam


# ---------------------------------------------------------------------------

def process_image(model, image_path: str, output_dir: str, save_heatmap: bool = True):
    filename = os.path.splitext(os.path.basename(image_path))[0]

    if cfg.MODEL_TYPE == "yolo":
        frame_rgb = np.array(PILImage.open(image_path).convert("RGB"))
        heatmap, label, score, results = yolo_gradcam(model, frame_rgb)
        print(f"[{filename}]  prediction: {label}  (score: {score:.3f})")

    elif cfg.MODEL_TYPE == "classifier":
        input_tensor, display_image = cfg.load_and_preprocess_image(image_path)
        target_layer = cfg.get_target_layer(model)
        cam = GradCAM(model, target_layer)
        heatmap, class_idx, raw_output = cam(input_tensor)
        cam.remove_hooks()

        probs  = raw_output.softmax(dim=1).squeeze().numpy()
        label  = (cfg.CLASS_NAMES[class_idx]
                   if class_idx < len(cfg.CLASS_NAMES) else str(class_idx))
        score  = float(probs[class_idx])
        frame_rgb = display_image
        print(f"[{filename}]  prediction: {label}  (confidence: {score:.3f})")

    elif cfg.MODEL_TYPE == "reconstruction":
        input_tensor, display_image = cfg.load_and_preprocess_image(image_path)
        heatmap, score, _ = reconstruction_error_map(model, input_tensor)
        label = "anomaly" if score > cfg.RECON_THRESHOLD else "normal"
        frame_rgb = display_image
        print(f"[{filename}]  anomaly score: {score:.4f}  → {label}")

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {cfg.MODEL_TYPE!r}")

    if save_heatmap:
        os.makedirs(output_dir, exist_ok=True)
        overlay_rgb = overlay_heatmap(frame_rgb, heatmap)
        out_path    = os.path.join(output_dir, f"{filename}_{label}_explained.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        print(f"  → saved explanation to {out_path}")

    return label, score


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline single-image / batch anomaly detection with XAI"
    )
    parser.add_argument("--input",  required=True,
                        help="Image file or folder of images")
    parser.add_argument("--output", default="results",
                        help="Output directory for explanation images (default: results/)")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip saving the heatmap overlay image")
    args = parser.parse_args()

    print(f"Loading model from: {cfg.MODEL_PATH}")
    model = cfg.load_model()
    print("Model loaded.\n")

    save_heatmap = not args.no_heatmap
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    if os.path.isdir(args.input):
        paths = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(exts)
        )
        if not paths:
            sys.exit(f"No supported images found in {args.input!r}")
        for p in paths:
            process_image(model, p, args.output, save_heatmap)
    elif os.path.isfile(args.input):
        process_image(model, args.input, args.output, save_heatmap)
    else:
        sys.exit(f"[ERROR] Input not found: {args.input!r}")


if __name__ == "__main__":
    main()
