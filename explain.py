"""
explain.py
----------
Explainability methods for the anomaly detector.

Methods
-------
GradCAM
    For classifier-style and YOLO models.  Looks at gradients flowing into
    the last conv layer to find which pixels drove the decision.
    Reference: Selvaraju et al., "Grad-CAM", ICCV 2017.

reconstruction_error_map
    For autoencoder-style models.  The per-pixel reconstruction error IS
    the anomaly heatmap — no gradient tricks needed.

yolo_gradcam
    Wraps GradCAM for an ultralytics YOLO model.  Attaches hooks to the
    backbone's last C2f/Conv block, runs a forward pass, and returns the
    same (heatmap, label, score) tuple used by the rest of the pipeline.

overlay_heatmap
    Blends a JET-colourmap heatmap onto an RGB frame.

All methods work fully offline — no network calls are made.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


# ---------------------------------------------------------------------------
# Grad-CAM (plain PyTorch models)
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients   = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor : (1, C, H, W), already preprocessed.
        class_idx    : class to explain; None → argmax (predicted class).
        Returns      : (heatmap HxW float[0,1], class_idx int, raw output tensor)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)   # (1, num_classes)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        output[0, class_idx].backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.ndim == 0:          # edge-case: single spatial cell
            cam = cam.reshape(1, 1)

        if cam.max() > 0:
            cam = cam / cam.max()

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam.astype(np.float32), (w, h))
        return cam, class_idx, output.detach().cpu()

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


# ---------------------------------------------------------------------------
# Reconstruction error map (autoencoder-style models)
# ---------------------------------------------------------------------------

def reconstruction_error_map(model, input_tensor: torch.Tensor):
    """
    Returns: (heatmap HxW float[0,1], scalar anomaly score float, reconstruction tensor)
    """
    with torch.no_grad():
        reconstruction = model(input_tensor)

    error = (input_tensor - reconstruction).pow(2).mean(dim=1, keepdim=True)
    error_map     = error.squeeze().cpu().numpy()
    anomaly_score = float(error_map.mean())

    if error_map.max() > 0:
        error_map = error_map / error_map.max()

    return error_map, anomaly_score, reconstruction


# ---------------------------------------------------------------------------
# YOLO Grad-CAM  (ultralytics models)
# ---------------------------------------------------------------------------

def _find_yolo_target_layer(pt_model):
    """
    Walk the ultralytics model graph and return the last Conv2d we find
    in the backbone/neck.  Works for YOLOv8/v9/v10 out of the box.
    """
    last_conv = None
    for name, module in pt_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("Could not find a Conv2d layer in the YOLO model.")
    return last_conv


def yolo_gradcam(yolo_model, frame_rgb: np.ndarray, input_size=(640, 640)):
    """
    Run Grad-CAM on one RGB frame using an ultralytics YOLO model.

    Parameters
    ----------
    yolo_model  : loaded ultralytics YOLO object (model.pt_model is the nn.Module)
    frame_rgb   : HxWx3 uint8 RGB numpy array (raw video frame)
    input_size  : (W, H) the model expects (default 640×640)

    Returns
    -------
    heatmap     : HxW float[0,1]  — Grad-CAM activation map
    label       : str             — top predicted class name or "anomaly"/"normal"
    score       : float           — confidence / anomaly score
    detections  : list[dict]      — raw YOLO result dicts (boxes, cls, conf)
    """
    import torchvision.transforms.functional as TF

    pt_model = yolo_model.pt_model
    pt_model.eval()

    # Pre-process frame → tensor
    h_orig, w_orig = frame_rgb.shape[:2]
    pil = TF.to_pil_image(frame_rgb)
    pil_resized = pil.resize(input_size)
    tensor = TF.to_tensor(pil_resized).unsqueeze(0)          # (1,3,H,W)
    tensor = tensor.to(next(pt_model.parameters()).device)
    tensor.requires_grad_(True)

    # Attach hooks to last conv
    target_layer = _find_yolo_target_layer(pt_model)
    activations_store = {}
    gradients_store   = {}

    def fwd_hook(m, inp, out):
        activations_store["a"] = out

    def bwd_hook(m, gin, gout):
        gradients_store["g"] = gout[0]

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    # Forward
    try:
        raw_out = pt_model(tensor)
    except Exception:
        # Some YOLO heads return structured objects; fall back to predict()
        fh.remove(); bh.remove()
        results = yolo_model.predict(frame_rgb, verbose=False)
        return _fallback_heatmap(frame_rgb), _yolo_label(results), _yolo_score(results), results

    # Pick the score to back-propagate through
    score_tensor = _yolo_score_tensor(raw_out)
    if score_tensor is None:
        fh.remove(); bh.remove()
        results = yolo_model.predict(frame_rgb, verbose=False)
        return _fallback_heatmap(frame_rgb), _yolo_label(results), _yolo_score(results), results

    pt_model.zero_grad()
    score_tensor.backward(retain_graph=True)

    fh.remove(); bh.remove()

    # Build Grad-CAM heatmap
    if "a" not in activations_store or "g" not in gradients_store:
        results = yolo_model.predict(frame_rgb, verbose=False)
        return _fallback_heatmap(frame_rgb), _yolo_label(results), _yolo_score(results), results

    acts = activations_store["a"].detach()     # (1, C, h, w)
    grads = gradients_store["g"].detach()      # (1, C, h, w)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = cam.squeeze().cpu().numpy()
    if cam.ndim == 0:
        cam = cam.reshape(1, 1)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam.astype(np.float32), (w_orig, h_orig))

    # Get readable label + score via predict (no hook overhead this time)
    results = yolo_model.predict(frame_rgb, verbose=False)
    label   = _yolo_label(results)
    score   = _yolo_score(results)

    return cam, label, score, results


# ---- helpers ---------------------------------------------------------------

def _yolo_score_tensor(raw_out):
    """
    Try to extract a differentiable scalar from whatever the YOLO head returns.
    Handles Detect, Classify, and raw tensor outputs.
    """
    if isinstance(raw_out, torch.Tensor):
        return raw_out.max()
    if isinstance(raw_out, (list, tuple)):
        for item in raw_out:
            t = _yolo_score_tensor(item)
            if t is not None:
                return t
    return None


def _yolo_label(results) -> str:
    try:
        r = results[0]
        if hasattr(r, "probs") and r.probs is not None:
            idx = int(r.probs.top1)
            return r.names[idx]
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes):
            idx = int(r.boxes.cls[0].item())
            return r.names.get(idx, str(idx))
    except Exception:
        pass
    return "unknown"


def _yolo_score(results) -> float:
    try:
        r = results[0]
        if hasattr(r, "probs") and r.probs is not None:
            return float(r.probs.top1conf.item())
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes):
            return float(r.boxes.conf[0].item())
    except Exception:
        pass
    return 0.0


def _fallback_heatmap(frame_rgb: np.ndarray) -> np.ndarray:
    """Return a blank (zero) heatmap when Grad-CAM cannot be computed."""
    return np.zeros(frame_rgb.shape[:2], dtype=np.float32)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def overlay_heatmap(original_image: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Blend a JET-colourmap heatmap over an RGB image.

    Parameters
    ----------
    original_image : HxWx3 uint8 RGB
    heatmap        : HxW float[0,1]
    alpha          : heatmap opacity (0 = only original, 1 = only heatmap)

    Returns
    -------
    HxWx3 uint8 RGB
    """
    if heatmap.shape[:2] != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap.astype(np.float32),
                             (original_image.shape[1], original_image.shape[0]))

    hm_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * hm_color.astype(np.float32) +
               (1 - alpha) * original_image.astype(np.float32))
    return np.clip(overlay, 0, 255).astype(np.uint8)
