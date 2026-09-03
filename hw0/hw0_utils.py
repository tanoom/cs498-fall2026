"""Small provided utilities shared by the three HW0 task scripts.

The readable task pipelines live in each task file's ``main()`` function.
Students do not need to edit this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


COURT_XY = np.array(
    [[0.0, 15.24], [28.65, 15.24], [28.65, 0.0], [0.0, 0.0]],
    dtype=float,
)
COURT_UV = np.array(
    [
        [642.893784, 589.797136],
        [1715.313542, 773.807048],
        [1087.515019, 1049.405604],
        [74.211568, 637.256706],
    ],
    dtype=float,
)

LANDMARK_XYZ = np.array(
    [
        [0.0, 15.24, 0.0],
        [28.65, 15.24, 0.0],
        [28.65, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.22, 6.705, 3.965],
        [1.22, 8.535, 3.965],
        [1.22, 8.535, 2.745],
        [1.22, 6.705, 2.745],
    ],
    dtype=float,
)
LANDMARK_UV = np.array(
    [
        [642.893784, 589.797136],
        [1715.313542, 773.807048],
        [1087.515019, 1049.405604],
        [74.211568, 637.256706],
        [375.621468, 464.070907],
        [439.733519, 462.405659],
        [441.398767, 496.543244],
        [376.454092, 499.873740],
    ],
    dtype=float,
)


def float_image(path: Path) -> np.ndarray:
    image = iio.imread(path)
    if image.dtype.kind in "ui":
        return image.astype(float) / np.iinfo(image.dtype).max
    return image.astype(float)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = np.uint8(np.round(np.clip(image, 0.0, 1.0) * 255.0))
    iio.imwrite(path, image)


def write_json(path: Path, values: dict[str, float | int | str | None]) -> None:
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def read_video_frame(video_path: Path, frame_index: int = 0) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def project_planar(xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points_h = np.column_stack((xy, np.ones(len(xy))))
    uv_h = (homography @ points_h.T).T
    return uv_h[:, :2] / uv_h[:, 2:3]


def warp_rgba(
    logo_rgba: np.ndarray,
    logo_to_image: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Bilinear inverse warp used by both basketball and tennis."""
    height, width = output_shape
    source_height, source_width, channels = logo_rgba.shape
    yy, xx = np.indices((height, width), dtype=float)
    target = np.stack((xx.ravel(), yy.ravel(), np.ones(height * width)))
    source_h = np.einsum("ij,jk->ik", np.linalg.inv(logo_to_image), target)
    finite = np.abs(source_h[2]) > 1e-12
    source_h[:, finite] /= source_h[2:3, finite]
    x, y = source_h[0], source_h[1]
    valid = (
        finite
        & (x >= 0.0)
        & (x <= source_width - 1)
        & (y >= 0.0)
        & (y <= source_height - 1)
    )
    output = np.zeros((height * width, channels), dtype=float)
    if np.any(valid):
        xv, yv = x[valid], y[valid]
        x0, y0 = np.floor(xv).astype(int), np.floor(yv).astype(int)
        x1 = np.minimum(x0 + 1, source_width - 1)
        y1 = np.minimum(y0 + 1, source_height - 1)
        wx, wy = (xv - x0)[:, None], (yv - y0)[:, None]
        top = (1.0 - wx) * logo_rgba[y0, x0] + wx * logo_rgba[y0, x1]
        bottom = (1.0 - wx) * logo_rgba[y1, x0] + wx * logo_rgba[y1, x1]
        output[valid] = (1.0 - wy) * top + wy * bottom
    return output.reshape(height, width, channels)


def project_points(xyz: np.ndarray, projection: np.ndarray) -> np.ndarray:
    points_h = np.column_stack((xyz, np.ones(len(xyz))))
    uv_h = np.einsum("ij,nj->ni", projection, points_h)
    return uv_h[:, :2] / uv_h[:, 2:3]


def decompose_projection(projection: np.ndarray) -> dict[str, np.ndarray]:
    intrinsic, rotation, center_h, *_ = cv2.decomposeProjectionMatrix(projection)
    intrinsic /= intrinsic[2, 2]
    center = center_h[:3, 0] / center_h[3, 0]
    translation = -rotation @ center
    return {"K": intrinsic, "R": rotation, "t": translation, "C": center}


def load_bunny(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line:
                raise ValueError("Invalid bunny PLY")
            header.extend(line)
        text = header.decode("ascii")
        vertex_line = next(line for line in text.splitlines() if line.startswith("element vertex "))
        count = int(vertex_line.split()[-1])
        vertices = np.fromfile(stream, dtype="<f8", count=count * 3)
    return vertices.reshape(count, 3)


def symmetric_epipolar_distance(matches: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    first = np.column_stack((matches[:, :2], np.ones(len(matches))))
    second = np.column_stack((matches[:, 2:], np.ones(len(matches))))
    lines_second = (fundamental @ first.T).T
    lines_first = (fundamental.T @ second.T).T
    residual = np.abs(np.sum(second * lines_second, axis=1))
    first_norm = np.linalg.norm(lines_first[:, :2], axis=1)
    second_norm = np.linalg.norm(lines_second[:, :2], axis=1)
    return 0.5 * (residual / first_norm + residual / second_norm)


def _draw_line(axis: plt.Axes, line: np.ndarray, width: int, height: int, color: np.ndarray) -> None:
    a, b, c = line
    candidates: list[tuple[float, float]] = []
    if abs(b) > 1e-12:
        candidates.extend([(0.0, -c / b), (width - 1.0, -(a * (width - 1.0) + c) / b)])
    if abs(a) > 1e-12:
        candidates.extend([(-c / a, 0.0), (-(b * (height - 1.0) + c) / a, height - 1.0)])
    visible = [(x, y) for x, y in candidates if -1 <= x <= width and -1 <= y <= height]
    if len(visible) >= 2:
        axis.plot(
            [visible[0][0], visible[1][0]],
            [visible[0][1], visible[1][1]],
            color=color,
            linewidth=1.2,
        )


def plot_two_views(
    images: list[np.ndarray],
    matches: np.ndarray,
    output_path: Path,
    fundamental: np.ndarray | None = None,
) -> None:
    selected = matches[np.linspace(0, len(matches) - 1, 10, dtype=int)]
    palette = plt.colormaps["tab10"](np.arange(10))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    first_h = np.column_stack((selected[:, :2], np.ones(len(selected))))
    second_h = np.column_stack((selected[:, 2:], np.ones(len(selected))))
    lines = None
    if fundamental is not None:
        lines = [(fundamental.T @ second_h.T).T, (fundamental @ first_h.T).T]
    for view, axis in enumerate(axes):
        points = selected[:, :2] if view == 0 else selected[:, 2:]
        axis.imshow(images[view])
        for index, (point, color) in enumerate(zip(points, palette, strict=True)):
            axis.scatter(point[0], point[1], color=color, s=30, edgecolor="white", linewidth=0.5)
            axis.text(point[0] + 5, point[1] - 5, str(index + 1), color=color, weight="bold")
            if lines is not None:
                _draw_line(
                    axis,
                    lines[view][index],
                    images[view].shape[1],
                    images[view].shape[0],
                    color,
                )
        axis.set_title(f"View {view + 1}")
        axis.axis("off")
    title = "Corresponding points" if fundamental is None else "Corresponding points and epipolar lines"
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
