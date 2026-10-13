import numpy as np

data = np.load("outputs/task2/task2_results.npz")
print("Saved keys:", data.files)

np.set_printoptions(precision=6, suppress=True)

K, R = data["K"], data["R"]
t, C = data["t"].reshape(3), data["C"].reshape(3)

for name, value in [("K", K), ("R", R), ("t", t), ("C", C)]:
    print(f"{name} =\n{value}")

C_check = -R.T @ t
print("-R.T @ t =", C_check)
print("Verification error =", np.linalg.norm(C - C_check))
print("Verified:", np.allclose(C, C_check, rtol=1e-6, atol=1e-8))
print("Optical-center height (m) =", C[2])