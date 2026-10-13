"""HW0 optional bonus: foreground-preserving logo insertion in tennis video.

Run from this folder:

    python bonus_tennis_video.py

Pipeline for every clip:
  1. track the small residual camera drift: Lucas-Kanade tracks of reference-
     frame corners plus RANSAC give a homography reference -> frame t (two of
     the "fixed" cameras drift by 15-30 px, enough to smear a naive median);
  2. align evenly spaced frames to the reference and take the per-pixel
     temporal median, which reveals the clean court because a moving player
     covers any one pixel for only a minority of the samples;
  3. estimate the court-to-image homography from the clip's A--H anchors and
     the metric coordinates derived in Task 1; per frame, chain it with the
     drift homography and warp the logo onto the court;
  4. per frame, turn the color distance to the (drift-warped) clean background
     into a soft foreground alpha (noise-adaptive smoothstep, morphology,
     feathering);
  5. lower the logo's alpha wherever foreground is present and blend it over
     the live frame, so players stay in front of the virtual logo.

Only Task 1 geometry and alpha blending are reused. Video loading, camera
tracking, background estimation, masking, logo warping, composition, and
writing live in this file.
"""

from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from task1_homography import (
    TENNIS_XY,
    alpha_blend,
    estimate_homography,
    logo_to_image_homography,
)


CLIPS = ("match001", "match005", "match028", "match091", "match125")
VIDEO_CLIPS = ("match005",)  # only this clip needs a complete MP4

# Same 6 m x 3 m rectangle as Task 1C: centered, just inside the near baseline,
# where the near player keeps walking across it.
LOGO_LOWER_LEFT_XY = (-3.0, 0.5)
LOGO_SIZE_XY = (6.0, 3.0)

BACKGROUND_SAMPLES = 45  # frames used by the temporal median
ROI_PADDING = 24  # pixels around the logo so blur and morphology see context
NOISE_MARGIN = 64  # extra reference-frame margin so drifted ROIs still have noise data
STATIC_DRIFT = 0.25  # px; smaller camera motion is treated as exactly none
MIN_TRACKS = 40  # fewer RANSAC inliers than this keeps the previous drift estimate
NOISE_SIGMAS = 4.0  # distances below this many noise sigmas are background
MIN_LOW_THRESHOLD = 14.0  # floor for the lower threshold, 8-bit RGB distance
THRESHOLD_WIDTH = 26.0  # soft ramp from "background" to "foreground"


# ------------------------------ video input/output ------------------------------


def open_video(path: Path) -> tuple[cv2.VideoCapture, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return capture, count, fps


def iterate_frames(path: Path, first: int, last: int):
    """Yield (index, RGB uint8 frame) for first <= index < last, decoding sequentially."""
    capture, count, _ = open_video(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    for index in range(first, min(last, count)):
        ok, frame = capture.read()
        if not ok:
            break
        yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    capture.release()


def read_frame(path: Path, index: int) -> np.ndarray:
    return next(frame for _, frame in iterate_frames(path, index, index + 1))


def open_writer(path: Path, fps: float, size_wh: tuple[int, int]) -> cv2.VideoWriter:
    """Prefer H.264 so the MP4 plays everywhere; fall back to MPEG-4 Part 2."""
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size_wh)
        if writer.isOpened():
            return writer
        writer.release()
    raise ValueError(f"Could not open a video writer for {path}")


# -------------------------------- camera tracking --------------------------------


def corner_shift(drift: np.ndarray, shape_hw: tuple[int, int]) -> float:
    """Largest displacement, in pixels, that a homography applies to the image corners."""
    height, width = shape_hw
    corners = np.float32([[[0, 0]], [[width - 1, 0]], [[0, height - 1]], [[width - 1, height - 1]]])
    return float(np.max(np.linalg.norm(cv2.perspectiveTransform(corners, drift) - corners, axis=2)))


def track_camera(
    reference_gray: np.ndarray,
    reference_points: np.ndarray,
    frame: np.ndarray,
    previous: np.ndarray,
) -> np.ndarray:
    """Homography mapping reference-frame pixels to this frame's pixels.

    Reference corners are followed with pyramidal Lucas-Kanade, started from the
    previous frame's estimate. RANSAC then keeps the dominant (static scene)
    motion and rejects tracks on players, the ball, and broadcast overlays.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    guess = cv2.perspectiveTransform(reference_points, previous)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        reference_gray, gray, reference_points, guess,
        winSize=(31, 31), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    keep = status.ravel() == 1
    if keep.sum() < MIN_TRACKS:
        return previous
    drift, inliers = cv2.findHomography(reference_points[keep], tracked[keep], cv2.RANSAC, 1.0)
    if drift is None or inliers.sum() < MIN_TRACKS:
        return previous
    return np.eye(3) if corner_shift(drift, gray.shape) < STATIC_DRIFT else drift


def warp_to_frame(reference_image: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """Resample a reference-coordinate image into the coordinates of a drifted frame."""
    height, width = reference_image.shape[:2]
    return cv2.warpPerspective(reference_image, drift, (width, height), borderMode=cv2.BORDER_REPLICATE)


def warp_to_reference(frame: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """Undo the drift so the frame lines up with the reference frame."""
    height, width = frame.shape[:2]
    return cv2.warpPerspective(
        frame, drift, (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
    )


# ----------------------------- background and logo ------------------------------


def estimate_background(stack: np.ndarray) -> np.ndarray:
    """Per-pixel temporal median of the sampled frames, as float32 RGB in [0, 255]."""
    return np.median(stack, axis=0).astype(np.float32)


def estimate_noise(stack: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Robust per-pixel sigma of the RGB distance to the background.

    The median absolute deviation ignores the minority of samples in which a
    player covers the pixel, so it measures compression and texture noise only.
    """
    distance = np.linalg.norm(stack.astype(np.float32) - background, axis=-1)
    sigma = 1.4826 * np.median(distance, axis=0)
    return cv2.GaussianBlur(sigma, (0, 0), 3.0)


def warp_logo(logo_rgba: np.ndarray, logo_to_image: np.ndarray, roi: tuple[slice, slice]) -> np.ndarray:
    """Inverse-warp the RGBA logo into the image ROI with bilinear sampling.

    Color is premultiplied by alpha before interpolation and divided out
    afterwards so transparent logo pixels do not bleed dark fringes.
    """
    height, width = roi[0].stop - roi[0].start, roi[1].stop - roi[1].start
    source_height, source_width = logo_rgba.shape[:2]
    vv, uu = np.indices((height, width), dtype=float)
    image_h = np.stack((uu.ravel() + roi[1].start, vv.ravel() + roi[0].start, np.ones(height * width)))
    logo_h = np.linalg.inv(logo_to_image) @ image_h
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = logo_h[0] / logo_h[2], logo_h[1] / logo_h[2]
    valid = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= 0.0) & (x <= source_width - 1) & (y >= 0.0) & (y <= source_height - 1)
    )
    xv, yv = x[valid], y[valid]
    x0, y0 = np.floor(xv).astype(int), np.floor(yv).astype(int)
    x1, y1 = np.minimum(x0 + 1, source_width - 1), np.minimum(y0 + 1, source_height - 1)
    wx, wy = (xv - x0)[:, None], (yv - y0)[:, None]

    premultiplied = logo_rgba.copy()
    premultiplied[..., :3] *= premultiplied[..., 3:4]
    top = (1.0 - wx) * premultiplied[y0, x0] + wx * premultiplied[y0, x1]
    bottom = (1.0 - wx) * premultiplied[y1, x0] + wx * premultiplied[y1, x1]
    sampled = (1.0 - wy) * top + wy * bottom
    sampled[:, :3] /= np.maximum(sampled[:, 3:4], 1e-6)

    warped = np.zeros((height * width, 4), dtype=float)
    warped[valid] = sampled
    return warped.reshape(height, width, 4)


def logo_roi(
    logo_to_image: np.ndarray,
    logo_shape: tuple[int, ...],
    image_shape: tuple[int, ...],
    padding: int,
) -> tuple[slice, slice]:
    """Padded image bounding box of the projected logo; nothing outside it ever changes."""
    source_height, source_width = logo_shape[:2]
    corners = np.float64([[[0, 0]], [[source_width - 1, 0]], [[0, source_height - 1]], [[source_width - 1, source_height - 1]]])
    uv = cv2.perspectiveTransform(corners, logo_to_image).reshape(-1, 2)
    (u_min, v_min), (u_max, v_max) = np.floor(uv.min(axis=0)).astype(int), np.ceil(uv.max(axis=0)).astype(int)
    height, width = image_shape[:2]
    return (
        slice(max(v_min - padding, 0), min(v_max + padding + 1, height)),
        slice(max(u_min - padding, 0), min(u_max + padding + 1, width)),
    )


# ---------------------------- foreground and blending ---------------------------


def foreground_alpha(frame: np.ndarray, background: np.ndarray, noise_sigma: np.ndarray) -> np.ndarray:
    """Soft alpha in [0, 1]: 0 where the frame matches the clean court, 1 on players."""
    smoothed = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 1.0)
    distance = np.linalg.norm(smoothed - background, axis=-1)
    low = np.maximum(MIN_LOW_THRESHOLD, NOISE_SIGMAS * noise_sigma)
    t = np.clip((distance - low) / THRESHOLD_WIDTH, 0.0, 1.0)
    alpha = t * t * (3.0 - 2.0 * t)  # smoothstep

    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, small)  # drop isolated speckles
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, large)  # fill holes inside the player
    alpha = cv2.dilate(alpha, small)  # recover the thin edge lost to thresholding
    return cv2.GaussianBlur(alpha, (0, 0), 1.2)  # feather the matte


def composite(frame: np.ndarray, logo_rgba: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Blend the logo over the live frame, hidden wherever foreground is present.

    With logo alpha a_l and foreground alpha a_f this equals placing the
    foreground back over a logo-inserted background:
        out = a_f * frame + (1 - a_f) * (a_l * logo + (1 - a_l) * frame).
    """
    occluded = logo_rgba.copy()
    occluded[..., 3] *= 1.0 - foreground
    blended = alpha_blend(occluded, frame.astype(float) / 255.0)
    return np.uint8(np.round(np.clip(blended, 0.0, 1.0) * 255.0))


# ------------------------------------ driver ------------------------------------


def process_clip(root: Path, output_dir: Path, name: str, logo: np.ndarray, write_video: bool) -> dict:
    video_path = root / f"data/tennis/{name}.mp4"
    meta = np.load(root / f"data/tennis/{name}.npz", allow_pickle=False)
    anchors_uv = meta["court_anchor_image_xy"].astype(float)
    first, last = (int(v) for v in meta["valid_frame_range"])
    _, _, fps = open_video(video_path)

    # Pass 1: track the drift of every frame relative to the annotated reference
    # frame, and collect drift-corrected samples for the temporal median.
    reference = read_frame(video_path, int(meta["reference_frame"]))
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    reference_points = cv2.goodFeaturesToTrack(reference_gray, maxCorners=1500, qualityLevel=0.01, minDistance=10)
    wanted = set(np.linspace(first, last - 1, BACKGROUND_SAMPLES, dtype=int).tolist())
    drifts, samples = {}, []
    drift = np.eye(3)
    for index, frame in iterate_frames(video_path, first, last):
        drift = drifts[index] = track_camera(reference_gray, reference_points, frame, drift)
        if index in wanted:
            samples.append(warp_to_reference(frame, drift))
    stack = np.stack(samples)
    del samples

    # Clean court and its noise level, both in reference-frame coordinates.
    background = estimate_background(stack)
    iio.imwrite(output_dir / f"{name}_background.png", np.uint8(np.round(background)))

    # Task 1 geometry with this clip's anchors (annotated in the reference frame).
    court_to_image = estimate_homography(TENNIS_XY, anchors_uv)
    logo_to_image = logo_to_image_homography(court_to_image, logo.shape, LOGO_LOWER_LEFT_XY, LOGO_SIZE_XY)
    wide = logo_roi(logo_to_image, logo.shape, background.shape, ROI_PADDING + NOISE_MARGIN)
    wide_sigma = estimate_noise(stack[(slice(None), *wide)], background[wide])
    noise_sigma = np.full(background.shape[:2], np.median(wide_sigma), dtype=np.float32)
    noise_sigma[wide] = wide_sigma
    court_model = np.dstack((background, noise_sigma))  # RGB + sigma, warped together per frame
    del stack

    static_roi = logo_roi(logo_to_image, logo.shape, background.shape, ROI_PADDING)
    static_logo = warp_logo(logo, logo_to_image, static_roi)
    identity = np.eye(3)

    writer = None
    if write_video:
        height, width = background.shape[:2]
        writer = open_writer(output_dir / f"{name}_occlusion_aware.mp4", fps, (width, height))

    # Pass 2: composite. The representative frame is the one where the most logo
    # is hidden by foreground.
    best_score, best_index, best_frame = -1.0, first, None
    for index, frame in iterate_frames(video_path, first, last):
        drift = drifts[index]
        if np.array_equal(drift, identity):
            roi, logo_crop, model_crop = static_roi, static_logo, court_model[static_roi]
        else:
            logo_to_frame = drift @ logo_to_image
            roi = logo_roi(logo_to_frame, logo.shape, frame.shape, ROI_PADDING)
            logo_crop = warp_logo(logo, logo_to_frame, roi)
            model_crop = warp_to_frame(court_model, drift)[roi]
        foreground = foreground_alpha(frame[roi], model_crop[..., :3], model_crop[..., 3])
        output = frame.copy()
        output[roi] = composite(frame[roi], logo_crop, foreground)
        score = float(np.sum(foreground * logo_crop[..., 3]))
        if score > best_score:
            best_score, best_index, best_frame = score, index, output
        if writer is not None:
            writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    if writer is not None:
        writer.release()

    iio.imwrite(output_dir / f"{name}_representative.png", best_frame)
    occluded_fraction = best_score / float(np.sum(static_logo[..., 3]))
    max_drift = max(corner_shift(d, background.shape[:2]) for d in drifts.values())
    print(
        f"{name}: representative frame {best_index}, {100 * occluded_fraction:.1f}% of the logo occluded, "
        f"max camera drift {max_drift:.1f} px"
    )
    return {"frame": best_index, "occluded_fraction": occluded_fraction, "max_camera_drift_px": max_drift}


def main() -> dict[str, dict]:
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/bonus"
    output_dir.mkdir(parents=True, exist_ok=True)
    logo = iio.imread(root / "data/court/logo.png").astype(float) / 255.0
    return {
        name: process_clip(root, output_dir, name, logo, write_video=name in VIDEO_CLIPS)
        for name in CLIPS
    }


if __name__ == "__main__":
    print("Bonus complete:", main())
