"""Class-agnostic regions of a keyframe (SAM, automatic mask generation): the "generic regions" write policy.

The segmenter knows nothing about affordance. It cuts every keyframe into object and part segments; their
boxes are the pool from which policy P3 draws its tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from affordance3d.device import default_device
from affordance3d.mem.encoders import model_source

SAM_ID = "facebook/sam-vit-large"  # Apache-2.0
SAM_REVISION = "6851e0441005b0fb96f2cc4dfac472f3d1b14af1"
MIN_PIXELS = 10
MAX_IMAGE_SHARE = 0.3  # larger segments are walls, floors and whole pieces of furniture
POINTS_PER_BATCH = 64

Box = tuple[int, int, int, int]  # left, top, right, bottom (exclusive)


def mask_boxes(masks: Sequence[np.ndarray], min_pixels: int = MIN_PIXELS, max_share: float = MAX_IMAGE_SHARE) -> list[Box]:
    """Bounding boxes of the masks that are neither tiny nor a large share of the image; duplicates removed."""
    boxes = []
    for mask in masks:
        rows_cols = np.argwhere(mask)
        if not min_pixels <= len(rows_cols) <= max_share * mask.size:
            continue
        (top, left), (bottom, right) = rows_cols.min(axis=0), rows_cols.max(axis=0) + 1
        boxes.append((int(left), int(top), int(right), int(bottom)))
    return list(dict.fromkeys(boxes))


class RegionSegmenter:
    """SAM run in its "segment everything" mode on one image at a time."""

    def __init__(self, device: str | None = None) -> None:
        from transformers import pipeline

        source = model_source(SAM_ID, SAM_REVISION)
        self._pipeline = pipeline("mask-generation", model=source["pretrained_model_name_or_path"],
                                  revision=source.get("revision"), device=device or default_device())

    def boxes(self, rgb: np.ndarray) -> list[Box]:
        from PIL import Image

        output = self._pipeline(Image.fromarray(rgb), points_per_batch=POINTS_PER_BATCH)
        return mask_boxes([np.asarray(mask, dtype=bool) for mask in output["masks"]])
