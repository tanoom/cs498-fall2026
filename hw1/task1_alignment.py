"""HW1 Task 1: rigid fitting and nearest correspondences.

Run directly:

    python task1_alignment.py
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
from scipy.spatial import cKDTree

from hw1_utils import (
    project_rigid_transform,
    rotation_error_degrees,
    save_alignment_visuals,
    to_homogeneous,
    transform_points,
    write_json,
)


# ------------------------ Task 1: Modify your code below ------------------------

def fit_rigid_svd(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit T_source_to_target for paired source and target points.

    Center both point sets, form their 3x3 cross-covariance, solve for rotation
    with SVD, correct a reflection if necessary, and recover translation.

    ``source`` and ``target`` are homogeneous [4, N] point matrices. Lecture
    notation is used directly: P_target = T_source_to_target @ P_source.
    """
    # we first drop the homogeneous row to get points [3,N]
    P = source[:3]
    Q = target[:3]

    #1. get the centroids, and subtract each point with them
    # we keep dim here because of the subtraction
    p_bar = P.mean(axis=1, keepdims=True)
    q_bar = Q.mean(axis=1, keepdims=True)
    P_hat= P - p_bar
    Q_hat = Q - q_bar

    #2. find the best R matrix with center-normalized points
    U, S, Vt = np.linalg.svd(P_hat @ Q_hat.T)
    V = Vt.T

    #3. ensure det != -1 to get rotation instead of refection
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ U.T

    #4. R is derived, we can have closed form solution of t
    t = q_bar - R @ p_bar

    #5. restore the result back to the homogeneous form
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t[:, 0]
    return T




def nearest_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source indices, nearest target indices, and accepted distances.

    Transpose xyz to [N, 3] only for the SciPy cKDTree boundary, query one
    neighbor per source column, and retain distances <= ``max_distance``.
    """
    # we first drop the homogeneous row to get points [3,N]
    # and then transpose it as instructed above
    tree = cKDTree(target[:3].T)
    distances, nearest_ids = tree.query(source[:3].T, k=1)
    is_keep = distances <= max_distance
    source_ids = np.nonzero(is_keep)[0]
    target_ids = nearest_ids[is_keep]
    return source_ids, target_ids, distances[is_keep]



# ------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK --------------------


def main(render: bool = True) -> dict[str, float | int | str | None]:
    """Fit real pairs, evaluate the full fragments, and save Task 1 outputs."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task1"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data/registration_demo.npz"
    if not data_path.exists():
        raise FileNotFoundError("Missing data/registration_demo.npz")
    with np.load(data_path, allow_pickle=False) as data:
        source = to_homogeneous(data["source"].astype(np.float64))
        target = to_homogeneous(data["target"].astype(np.float64))
        paired_source = to_homogeneous(data["task1_source"].astype(np.float64))
        paired_target = to_homogeneous(data["task1_target"].astype(np.float64))
        reference = project_rigid_transform(data["reference_transform"])

    # Checkpoint 1A: recover a rigid transform from known point pairs.
    estimate = fit_rigid_svd(paired_source, paired_target)
    aligned = transform_points(estimate, source)
    translation_error = float(np.linalg.norm(estimate[:3, 3] - reference[:3, 3]))
    rotation_error = rotation_error_degrees(estimate, reference)
    np.savetxt(output_dir / "estimated_transform.txt", estimate, fmt="%.8f")
    print(
        f"Checkpoint 1A: translation error={translation_error:.4f} m, "
        f"rotation error={rotation_error:.3f} deg"
    )

    # Checkpoint 1B: evaluate nearest-neighbor matches and render before/after.
    source_ids, _, distances = nearest_correspondences(aligned, target, 0.05)
    rmse = float(np.sqrt(np.mean(distances**2))) if len(distances) else None
    metrics = {
        "translation_error_m": translation_error,
        "rotation_error_deg": rotation_error,
        "known_pair_count": paired_source.shape[1],
        "nearest_correspondence_count": len(source_ids),
        "inlier_ratio": len(source_ids) / source.shape[1],
        "inlier_rmse_m": rmse,
        "status": "complete" if len(distances) else "starter_placeholder",
    }
    write_json(output_dir / "svd_metrics.json", metrics)
    if render:
        save_alignment_visuals(
            output_dir / "svd_alignment.png", source, target,
            np.stack([np.eye(4), estimate]),
        )
    print("Checkpoint 1B saved: svd_metrics.json")
    print("Alignment image saved" if render else "Alignment rendering skipped")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rendering", action="store_true")
    print("Task 1 complete:", main(render=not parser.parse_args().skip_rendering))
