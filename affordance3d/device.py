"""Which torch device small models run on."""

from __future__ import annotations

import os

ENV_VAR = "AFFORDANCE3D_DEVICE"


def default_device() -> str:
    """`AFFORDANCE3D_DEVICE` if set, otherwise Apple's GPU when present, otherwise the CPU.

    CUDA is never chosen automatically: on the shared server, GPU work must go through the queue,
    whose job scripts set AFFORDANCE3D_DEVICE=cuda themselves.
    """
    chosen = os.environ.get(ENV_VAR)
    if chosen:
        return chosen
    import torch  # imported late so that modules using this stay cheap to import

    return "mps" if torch.backends.mps.is_available() else "cpu"
