# Offline Video Anomaly Detector with Explainable AI

Detects anomalies in a live webcam feed or video file and overlays a
Grad-CAM / reconstruction-error heatmap showing **why** the model flagged
each frame — all running **fully offline** on your local machine.

---

## Files

| File                | Purpose                                                              |
|---------------------|----------------------------------------------------------------------|
| `inspect_model.py`  | **Run this first** — tells you what is inside your `best.pt`        |
| `model_loader.py`   | Model loading + all config (edit the `CONFIG` block here)           |
| `explain.py`        | Grad-CAM, reconstruction error map, YOLO Grad-CAM, overlay helpers  |
| `video_app.py`      | **Main entry point** — live video + XAI overlay, OpenCV window      |
| `infer.py`          | Batch / single-image inference with saved explanation images         |
| `requirements.txt`  | Python dependencies                                                  |

---

## Setup

### 1 — Install dependencies (once, offline after first install)

```bash
pip install -r requirements.txt
```

If you are on a machine with **no internet**, download the wheels on another
machine and use:

```bash
pip install --no-index --find-links /path/to/wheels -r requirements.txt
```

> `torch` and `ultralytics` are the largest downloads (~2 GB total for CUDA).
> Everything else is small.

---

### 2 — Inspect your `best.pt`

```bash
python inspect_model.py best.pt
```

It will print one of three cases:

| Output                   | Meaning                                       |
|--------------------------|-----------------------------------------------|
| `FULL MODEL OBJECT`      | Ready to load — set `MODEL_CLASS = None`      |
| `STATE_DICT`             | You must define the architecture class        |
| `CHECKPOINT DICT`        | Dict wrapping weights — set `CHECKPOINT_KEY`  |

---

### 3 — Configure `model_loader.py`

Open `model_loader.py` and edit the `CONFIG` section:

```python
MODEL_PATH = "best.pt"      # path to your best.pt
MODEL_TYPE = "yolo"         # "yolo" | "classifier" | "reconstruction"
```

#### If MODEL_TYPE = `"yolo"` (most common for best.pt from ultralytics)
Nothing else to change. The loader handles ultralytics YOLOv8/v9/v10
automatically.

#### If MODEL_TYPE = `"classifier"`
- Set `CLASS_NAMES` to match your training labels in output-index order,
  e.g. `["normal", "anomaly"]`.
- If `inspect_model.py` said STATE_DICT or CHECKPOINT, uncomment
  `MODEL_CLASS` and define your architecture.
- Optionally set `GRAD_CAM_TARGET_LAYER` (auto-detected if left as `None`).
- Set `INPUT_SIZE`, `NORM_MEAN`, `NORM_STD` to match your training transforms.

#### If MODEL_TYPE = `"reconstruction"`
- Set `RECON_THRESHOLD` (default `0.5`) — calibrate this against your
  validation set (the value that best separates normal from anomalous
  reconstruction error).
- Set `INPUT_SIZE`, `NORM_MEAN`, `NORM_STD` to match your training transforms.

---

## Running

### Live webcam (default — first webcam)

```bash
python video_app.py
```

### Specific webcam index

```bash
python video_app.py --source 1
```

### Video file

```bash
python video_app.py --source path/to/video.mp4
```

### Video file + save annotated output

```bash
python video_app.py --source path/to/video.mp4 --save output.avi
```

### Single image or folder (no window — saves PNG files)

```bash
python infer.py --input image.jpg
python infer.py --input /path/to/folder/ --output results/
```

---

## Window controls

| Key       | Action                              |
|-----------|-------------------------------------|
| `Q` / ESC | Quit                                |
| `P`       | Pause / resume                      |
| `H`       | Toggle XAI heatmap overlay on / off |
| `S`       | Save current frame to disk          |

---

## How the XAI works

### YOLO models (`MODEL_TYPE = "yolo"`)
Grad-CAM is attached to the **last Conv2d layer in the backbone**.
Bright (red/yellow) regions in the heatmap are the pixels that most
activated the detection or classification head — i.e. what the model
actually looked at when it made its decision.

### Classifier models (`MODEL_TYPE = "classifier"`)
Standard **Grad-CAM** (Selvaraju et al., ICCV 2017) on the last conv layer
of your CNN.  Gradients of the predicted class score with respect to that
layer's feature maps reveal which spatial regions drove the prediction.

### Reconstruction / autoencoder models (`MODEL_TYPE = "reconstruction"`)
The per-pixel L2 reconstruction error **is** the explanation — wherever the
model's output differs most from the input is exactly where the anomaly is.
No gradient tricks are needed; this is more faithful for this model type.

---

## Heatmap colour scale

| Colour           | Meaning                            |
|------------------|------------------------------------|
| Red / bright     | High activation — anomaly region   |
| Blue / dark      | Low activation — normal region     |

The **bottom bar** in the video window shows:
- Predicted label and confidence
- Current FPS
- Whether heatmap and pause are active
- A colour strip (green = normal, red = anomaly)

---

## Troubleshooting

**`Cannot open video source`**
- Webcam: try `--source 1` or `--source 2` if index 0 fails.
- File: verify the path and that OpenCV was built with the codec your file uses
  (`pip install opencv-python` includes the most common ones).

**`No Conv2d layer found automatically`**
Set `GRAD_CAM_TARGET_LAYER` in `model_loader.py` to the name of a conv layer
in your model (run `python inspect_model.py best.pt` and read the architecture
printout to find a layer name).

**`MODEL_CLASS is not set`**
`inspect_model.py` told you the file is a state_dict or checkpoint.
Uncomment and fill in the `MODEL_CLASS` block in `model_loader.py` with the
same `nn.Module` subclass used when you trained the model.

**Heatmap is all blue / blank**
- For reconstruction mode: the model may reconstruct the input very well
  (low error everywhere) — your threshold may need lowering.
- For Grad-CAM: the target layer may be too early in the network.
  Set `GRAD_CAM_TARGET_LAYER` to a later layer name.

**Very low FPS**
- Use `--source` with a lower-resolution video.
- Set `DEVICE = "cuda"` in `model_loader.py` if you have a GPU.
- Grad-CAM requires a backward pass on every frame; this is slower than
  plain inference. Toggle `H` to turn off the heatmap for max speed.

---

## Citation

If you use Grad-CAM in published work:

> Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
> (2017). **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
> Localization.** *ICCV 2017.*
