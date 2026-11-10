"""Frozen DINOv2 ViT-S/14 patch features, pixel lookup and PCA colouring (E1).

The whole frame is resized (never cropped) to the model input, so patch (r, c) covers
the same fraction of the original image in both directions; `patch_rc` relies on this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from affordance3d.device import default_device

DINO_ID = "facebook/dinov2-small"  # ViT-S/14, Apache-2.0
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
PATCH_SIZE = 14
DEFAULT_INPUT_HW = (336, 448)  # 480x640 scaled by 0.7; 24x32 patches of ~20x20 original px
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PCA_COMPONENTS = 3
NORM_EPS = 1e-12


def l2_normalize(rows: np.ndarray) -> np.ndarray:
    """Scale each row to unit length; all-zero rows stay zero."""
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows / np.maximum(norms, NORM_EPS)


def patch_rc(uv: np.ndarray, image_hw: tuple[int, int], grid_hw: tuple[int, int]) -> np.ndarray:
    """Map original-image pixels (u, v) to the (row, col) of the patch that covers them."""
    height, width = image_hw
    grid_h, grid_w = grid_hw
    rows = np.clip(np.floor(uv[:, 1] * grid_h / height), 0, grid_h - 1)
    cols = np.clip(np.floor(uv[:, 0] * grid_w / width), 0, grid_w - 1)
    return np.stack([rows, cols], axis=1).astype(np.int64)


def tokens_to_grid(tokens: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """Turn (1, prefix + Gh*Gw, C) transformer tokens into a (Gh, Gw, C) patch grid.

    Prefix tokens (CLS, and registers in register variants) come first and are dropped.
    """
    grid_h, grid_w = grid_hw
    n_patches = grid_h * grid_w
    if tokens.shape[1] < n_patches:
        raise ValueError(f"{tokens.shape[1]} tokens cannot fill a {grid_h}x{grid_w} patch grid")
    return tokens[0, tokens.shape[1] - n_patches :].reshape(grid_h, grid_w, -1)


@dataclass(frozen=True)
class PatchFeatures:
    """Patch features of one frame and the original image size they cover."""

    grid: np.ndarray  # (Gh, Gw, C) float32
    image_hw: tuple[int, int]

    @property
    def grid_hw(self) -> tuple[int, int]:
        return self.grid.shape[0], self.grid.shape[1]

    def at_pixels(self, uv: np.ndarray) -> np.ndarray:
        rc = patch_rc(uv, self.image_hw, self.grid_hw)
        return self.grid[rc[:, 0], rc[:, 1]]


class DinoV2Extractor:
    """Frozen DINOv2 ViT-S/14 run one frame at a time (batch=1), on MPS when available."""

    def __init__(self, input_hw: tuple[int, int] = DEFAULT_INPUT_HW, device: str | None = None) -> None:
        from transformers import AutoModel

        if input_hw[0] % PATCH_SIZE or input_hw[1] % PATCH_SIZE:
            raise ValueError(f"input size {input_hw} must be a multiple of {PATCH_SIZE}")
        self.input_hw = input_hw
        self.device = device or default_device()
        self._model = AutoModel.from_pretrained(DINO_ID, revision=DINO_REVISION).eval().to(self.device)

    @property
    def grid_hw(self) -> tuple[int, int]:
        return self.input_hw[0] // PATCH_SIZE, self.input_hw[1] // PATCH_SIZE

    def __call__(self, rgb: np.ndarray) -> PatchFeatures:
        import torch

        with torch.inference_mode():
            tokens = self._model(pixel_values=self._preprocess(rgb)).last_hidden_state
        grid = tokens_to_grid(tokens.float().cpu().numpy(), self.grid_hw)
        return PatchFeatures(grid, (rgb.shape[0], rgb.shape[1]))

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:  # noqa: F821
        import torch
        import torch.nn.functional as F

        image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
        image = F.interpolate(image, size=self.input_hw, mode="bilinear", antialias=True, align_corners=False)
        mean = torch.from_numpy(IMAGENET_MEAN)[:, None, None]
        std = torch.from_numpy(IMAGENET_STD)[:, None, None]
        return ((image - mean) / std).to(self.device)


@dataclass(frozen=True)
class PcaColoring:
    """Fixed 3-component PCA of unit-normalised features mapped to RGB in [0, 1]."""

    mean: np.ndarray  # (C,)
    components: np.ndarray  # (3, C)
    low: np.ndarray  # (3,)
    high: np.ndarray  # (3,)

    def to_rgb(self, features: np.ndarray) -> np.ndarray:
        projected = (l2_normalize(features) - self.mean) @ self.components.T
        spread = np.maximum(self.high - self.low, NORM_EPS)  # a constant component must not give NaN
        return np.clip((projected - self.low) / spread, 0.0, 1.0)


def fit_pca_coloring(rows: np.ndarray, percentiles: tuple[float, float] = (1.0, 99.0)) -> PcaColoring:
    """Fit one colouring on features from all frames so colours are comparable across views."""
    unit = l2_normalize(rows)
    mean = unit.mean(axis=0)
    _, _, vt = np.linalg.svd(unit - mean, full_matrices=False)
    components = vt[:PCA_COMPONENTS]
    # SVD signs are arbitrary; fix them so reruns give the same colours.
    signs = np.sign(components[np.arange(PCA_COMPONENTS), np.abs(components).argmax(axis=1)])
    components = components * signs[:, None]
    projected = (unit - mean) @ components.T
    low, high = np.percentile(projected, percentiles, axis=0)
    return PcaColoring(mean, components, low, high)
