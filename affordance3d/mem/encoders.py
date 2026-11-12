"""Frozen models read by the L2 memory: an image-text encoder (SigLIP 2) and a small vision-language model.

Both are used as they come, without training. A verified local copy under $AFFORDANCE3D_MODELS is preferred
(see scripts/remote/fetch_models.py); otherwise the hub is used at the pinned revision.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from affordance3d.device import default_device
from affordance3d.features import l2_normalize

MODELS_ENV = "AFFORDANCE3D_MODELS"
VERIFIED_MARKER = ".verified"  # written by scripts/remote/fetch_models.py once every file has passed its hash check
SIGLIP_ID = "google/siglip2-base-patch16-384"  # Apache-2.0
SIGLIP_REVISION = "f775b65a79762255128c981547af89addcfe0f88"
SIGLIP_TEXT_TOKENS = 64
VLM_ID = "Qwen/Qwen3-VL-4B-Instruct"  # Apache-2.0
VLM_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
IMAGE_BATCH = 16


def model_source(repo: str, revision: str) -> dict[str, str]:
    """`from_pretrained` arguments for a model: the verified local copy of `revision` if there is one, else the hub."""
    root = os.environ.get(MODELS_ENV)
    marker = Path(root) / repo / VERIFIED_MARKER if root else None
    if marker is not None and marker.is_file() and marker.read_text().strip() == revision:
        return {"pretrained_model_name_or_path": str(marker.parent)}
    return {"pretrained_model_name_or_path": repo, "revision": revision}


def _pooled(output: object) -> np.ndarray:
    tensor = output if hasattr(output, "detach") else output.pooler_output
    return l2_normalize(tensor.detach().float().cpu().numpy())


def _gpu_precision(device: str) -> dict:
    """Half precision on a GPU (a 4B model then needs about 9 GB instead of 17); full precision on the CPU."""
    if not device.startswith("cuda"):
        return {}
    import torch

    return {"torch_dtype": torch.bfloat16}


class ImageTextEncoder:
    """SigLIP 2: unit-length vectors for images and for texts in one shared space."""

    def __init__(self, device: str | None = None) -> None:
        from transformers import AutoModel, AutoProcessor

        source = model_source(SIGLIP_ID, SIGLIP_REVISION)
        self.device = device or default_device()
        self._processor = AutoProcessor.from_pretrained(**source)
        self._model = AutoModel.from_pretrained(**source).eval().to(self.device)

    def images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        rows = []
        with torch.inference_mode():
            for start in range(0, len(images), IMAGE_BATCH):
                batch = [Image.fromarray(image) for image in images[start:start + IMAGE_BATCH]]
                inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
                rows.append(_pooled(self._model.get_image_features(**inputs)))
        return np.concatenate(rows)

    def texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        lowered = [text.lower() for text in texts]  # SigLIP 2 was trained on lower-cased text
        with torch.inference_mode():
            inputs = self._processor(text=lowered, padding="max_length", max_length=SIGLIP_TEXT_TOKENS,
                                     truncation=True, return_tensors="pt").to(self.device)
            return _pooled(self._model.get_text_features(**inputs))


class VisionLanguageModel:
    """A small open vision-language model that answers one question about a few images."""

    def __init__(self, device: str | None = None) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        source = model_source(VLM_ID, VLM_REVISION)
        self.device = device or default_device()
        self._processor = AutoProcessor.from_pretrained(**source)
        self._model = AutoModelForImageTextToText.from_pretrained(**source, **_gpu_precision(self.device)).eval().to(self.device)

    def ask(self, images: Sequence[np.ndarray], prompt: str, max_new_tokens: int = 16) -> str:
        import torch
        from PIL import Image

        content = [{"type": "image", "image": Image.fromarray(image)} for image in images]
        messages = [{"role": "user", "content": [*content, {"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = output[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(answer, skip_special_tokens=True)[0].strip()
