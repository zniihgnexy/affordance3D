"""Monocular metric depth (Apple GPU or CPU by default, see affordance3d.device) and depth error metrics.

Weights download from the Hugging Face hub into the user's cache, not into the repository.
Licences differ by checkpoint (Depth Anything V2 Small: Apache-2.0; Base/Large: CC BY-NC 4.0;
Depth Pro: Apple's research licence); the mem branch only uses them for research.
"""

from __future__ import annotations

import numpy as np

from affordance3d.device import default_device

DELTA1 = 1.25


def depth_errors(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Standard depth metrics over pixels where both depths are valid (and `mask`, if given)."""
    valid = (gt > 0) & (pred > 0)
    if mask is not None:
        valid &= mask
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    if not len(g):
        return {"pixels": 0, "abs_rel": float("nan"), "rmse_m": float("nan"), "median_abs_m": float("nan"),
                "delta1": float("nan")}
    return {
        "pixels": int(valid.sum()),
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "median_abs_m": float(np.median(np.abs(p - g))),
        "delta1": float(np.mean(np.maximum(p / g, g / p) < DELTA1)),
    }


def with_true_focal(depth: np.ndarray, predicted_focal: float, true_focal: float) -> np.ndarray:
    """Depth Pro's metric depth is proportional to the focal length it assumed; use the known one."""
    return depth * (true_focal / predicted_focal)


class MonoDepth:
    """A Hugging Face depth-estimation checkpoint; `predict` returns metres at the image size."""

    def __init__(self, model_id: str, device: str | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.model_id = model_id
        self.device = device or default_device()
        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def predict(self, rgb: np.ndarray, focal_px: float) -> np.ndarray:
        inputs = self.processor(images=np.ascontiguousarray(rgb).copy(), return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_depth_estimation(outputs, target_sizes=[rgb.shape[:2]])[0]
        depth = result["predicted_depth"].float().cpu().numpy()
        predicted_focal = result.get("focal_length")
        if predicted_focal is not None:  # Depth Pro estimates the focal length; we know it from the device
            depth = with_true_focal(depth, float(predicted_focal), focal_px)
        return depth.astype(np.float32)
