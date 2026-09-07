"""HW0 Task 3: two-view geometry and epipolar lines.

Work from Checkpoint 3A through 3C, then run:

    python task3_epipolar.py
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from hw0_utils import plot_two_views, symmetric_epipolar_distance, write_json


# ------------------------ Task 3: Modify your code below ------------------------

def estimate_fundamental_matrix(matches: np.ndarray) -> np.ndarray:
    """Estimate a rank-two F from rows (u1, v1, u2, v2), N >= 8.

    Replace the runnable placeholder with the normalized eight-point method:
      1. normalize the points in each view independently;
      2. build the N x 9 design matrix from (u1, v1, u2, v2);
      3. solve its right null space with SVD;
      4. enforce rank two by zeroing the smallest singular value; and
      5. denormalize and choose a stable scale.
    """
    def normalize_2d(contours):
        centroid = contours.mean(axis=0)
        mean_distance = np.linalg.norm(contours - centroid,axis=1).mean()
        scale = np.sqrt(2) / mean_distance
        norm_matrix = np.array(
            [[scale, 0.0, -scale*centroid[0]],
             [0.0 ,scale, -scale*centroid[1]],
             [0.0, 0.0, 1.0]
             ]
        )
        return (contours - centroid) * scale, norm_matrix
    norm_p1, norm_matrix_1 = normalize_2d(matches[:, 0:2]) #view 1
    norm_p2, norm_matrix_2 = normalize_2d(matches[:, 2:4]) #view 2
    #build the N x 9 design matrix, one row per match from x2^T F x1 = 0
    u1, v1 = norm_p1[:,0], norm_p1[:,1]
    u2, v2 = norm_p2[:,0], norm_p2[:,1]
    ones = np.ones_like(u1)
    desgin_matrix = np.stack([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, ones], axis=1)
    U, S, Vt = np.linalg.svd(desgin_matrix)
    norm_F = Vt[-1].reshape(3,3)
    Uf, Sf, Vtf = np.linalg.svd(norm_F)
    Sf[-1] = 0.0
    norm_F = Uf @ np.diag(Sf) @ Vtf
    F = norm_matrix_2.T @ norm_F @ norm_matrix_1
    return F / np.linalg.norm(F)


# ------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK --------------------


def main() -> dict[str, float | int]:
    """Run the three Task 3 checkpoints in the same order as the handout."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task3"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_matches = np.load(root / "data/epipolar/all_good_matches.npy")
    eight_matches = np.load(root / "data/epipolar/eight_good_matches.npy")
    images = [
        iio.imread(root / "data/epipolar/images/0000.png"),
        iio.imread(root / "data/epipolar/images/0005.png"),
    ]

    # Checkpoint 3A: estimate F from the supplied eight correspondences.
    fundamental = estimate_fundamental_matrix(eight_matches)
    np.savetxt(
        output_dir / "task3a_fundamental_matrix.txt",
        fundamental,
        fmt="%.10e",
        header="Estimated fundamental matrix F",
    )
    print(f"Checkpoint 3A saved; rank(F) = {np.linalg.matrix_rank(fundamental, tol=1e-8)}")

    # Checkpoint 3B: visualize matches first, then the epipolar lines from F.
    plot_two_views(images, all_matches, output_dir / "task3b_correspondences.png")
    plot_two_views(images, all_matches, output_dir / "task3b_epipolar_lines.png", fundamental)
    print("Checkpoint 3B saved: correspondences and epipolar lines")

    # Checkpoint 3C: evaluate F on all 200 correspondences.
    errors = symmetric_epipolar_distance(all_matches, fundamental)
    metrics: dict[str, float | int] = {
        "fundamental_rank": int(np.linalg.matrix_rank(fundamental, tol=1e-8)),
        "mean_symmetric_epipolar_error_px": float(np.mean(errors)),
        "median_symmetric_epipolar_error_px": float(np.median(errors)),
    }
    write_json(output_dir / "task3c_metrics.json", metrics)
    np.savez(output_dir / "task3_results.npz", fundamental=fundamental)
    print("Checkpoint 3C saved: task3c_metrics.json")
    return metrics


if __name__ == "__main__":
    print("Task 3 complete:", main())
