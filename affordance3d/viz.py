"""Visual helpers for the demo notebook: Plotly 3D views and Matplotlib image rows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from affordance3d.frames import Frame
from affordance3d.geometry import transform_points

MAX_POINTS = 20_000  # keeps notebook outputs and HTML files light
POINT_SIZE = 2
FRUSTUM_DEPTH_M = 0.25
# The Redwood world frame is the first camera's frame: +y points down, +z forward.
DEFAULT_VIEW = {"up": {"x": 0, "y": -1, "z": 0}, "eye": {"x": 0.25, "y": -0.35, "z": -1.35}}
# Gravity-aligned scans (e.g. the SceneFun3D Faro frame): +z up, seen from above at an angle.
Z_UP_VIEW = {"up": {"x": 0, "y": 0, "z": 1}, "eye": {"x": 1.1, "y": 1.1, "z": 1.0}}


def subsample(n_points: int, max_points: int = MAX_POINTS, seed: int = 0) -> np.ndarray:
    """Deterministic subset of indices so repeated runs draw the same points."""
    if n_points <= max_points:
        return np.arange(n_points)
    return np.sort(np.random.default_rng(seed).choice(n_points, max_points, replace=False))


def points_trace(xyz: np.ndarray, colours: np.ndarray | str | None = None, values: np.ndarray | None = None,
                 name: str = "points", colorscale: str = "Viridis", size: float = POINT_SIZE,
                 max_points: int = MAX_POINTS, cmin: float | None = None, cmax: float | None = None,
                 visible: bool | str = True) -> go.Scatter3d:
    """Point cloud coloured by RGB in [0, 1] (array), one CSS colour (str), or a scalar (`values`).

    `visible="legendonly"` starts the trace hidden; clicking its legend entry shows it.
    """
    keep = subsample(len(xyz), max_points)
    marker: dict = {"size": size}
    if values is not None:
        marker.update(color=values[keep].astype(np.float32), colorscale=colorscale, cmin=cmin, cmax=cmax,
                      colorbar={"thickness": 12, "len": 0.6})
    elif isinstance(colours, str):
        marker.update(color=colours)
    elif colours is not None:
        rgb = (np.clip(colours[keep], 0.0, 1.0) * 255).astype(int)
        marker.update(color=[f"rgb({r},{g},{b})" for r, g, b in rgb])
    pts = xyz[keep].astype(np.float32)  # float32 halves the embedded (base64) coordinate payload
    return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", marker=marker, name=name,
                        visible=visible)


def frustum_trace(frame: Frame, colour: str = "black", depth: float = FRUSTUM_DEPTH_M) -> go.Scatter3d:
    """Wireframe pyramid showing a camera's position and viewing direction."""
    height, width = frame.image_hw
    corners_px = np.array([[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]], float)
    corners = (np.linalg.inv(frame.intrinsics) @ corners_px.T).T * depth
    world = transform_points(frame.cam_to_world, np.vstack([np.zeros(3), corners]))
    order = [0, 1, 2, 0, 3, 4, 0, 1, 4, 3, 2]  # apex-to-corner edges plus the image rectangle
    path = world[order]
    return go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines",
                        line={"color": colour, "width": 4}, name=f"camera {frame.index}")


def scene_figure(traces: Sequence[go.BaseTraceType], title: str, height: int = 560,
                 view: dict | None = None) -> go.Figure:
    figure = go.Figure(list(traces))
    figure.update_layout(
        title=title, height=height, margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={"aspectmode": "data", "camera": view or DEFAULT_VIEW,
               "xaxis": {"visible": False}, "yaxis": {"visible": False}, "zaxis": {"visible": False}},
        legend={"itemsizing": "constant"},
    )
    return figure


def image_row(images: Sequence[np.ndarray], titles: Sequence[str], panel_height: float = 3.0,
              **imshow_kwargs: object) -> tuple[plt.Figure, np.ndarray]:
    """One row of images with titles and no axes; returns the axes for overlays.

    `imshow_kwargs` (cmap, vmin, vmax, interpolation, ...) apply to every panel, so
    single-channel panels in one row share a colour scale.
    """
    height, width = images[0].shape[:2]
    figure, axes = plt.subplots(1, len(images), figsize=(panel_height * width / height * len(images), panel_height),
                                squeeze=False)
    for axis, image, title in zip(axes[0], images, titles, strict=True):
        axis.imshow(image, **imshow_kwargs)
        axis.set_title(title, fontsize=10)
        axis.set_axis_off()
    figure.tight_layout()
    return figure, axes[0]


def save_html(figure: go.Figure, path: Path) -> Path:
    """Write a standalone page; plotly.min.js is stored once beside it so pages open offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="directory")
    return path
