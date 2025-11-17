import numpy as np
import matplotlib.pyplot as plt
import os


GT_PATH = "/home/ho/INHA/ML/dataset/poses/10.txt"
EST_PATH = "/home/ho/INHA/ML/estimated_kitti_10.txt"


def load_kitti_pose_file(path):
    poses = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 12:
                continue
            nums = list(map(float, parts))
            T = np.array(nums).reshape(3,4)
            poses.append(T[:,3])
    return np.array(poses)


def umeyama(src, dst):
    assert src.shape == dst.shape

    mu_src = src.mean(0)
    mu_dst = dst.mean(0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = np.sum(S) / np.sum(src_c**2)
    t = mu_dst - scale * R @ mu_src

    return scale, R, t


gt = load_kitti_pose_file(GT_PATH)
est = load_kitti_pose_file(EST_PATH)

N = min(len(gt), len(est))
gt = gt[:N]
est = est[:N]

print(f"GT poses: {len(gt)}, EST poses: {len(est)}")



scale, R, t = umeyama(est, gt)
est_aligned = (scale * (R @ est.T)).T + t



ape = np.linalg.norm(gt - est_aligned, axis=1)
ape_rmse = np.sqrt(np.mean(ape**2))

rpe = np.linalg.norm(np.diff(gt - est_aligned, axis=0), axis=1)
rpe_rmse = np.sqrt(np.mean(rpe**2))

print("\n=== Evaluation Results ===")
print(f"APE RMSE: {ape_rmse:.3f} m")
print(f"RPE RMSE: {rpe_rmse:.3f} m")


plt.figure(figsize=(10,8))
plt.plot(gt[:,0], gt[:,2], label="GT", linewidth=2)
plt.plot(est_aligned[:,0], est_aligned[:,2], label="Estimate_10 (Aligned)", linewidth=2)
plt.xlabel("X (meters)")
plt.ylabel("Z (meters)")
plt.title("Trajectory Comparison_10 (Aligned)")
plt.grid()
plt.legend()
plt.axis("equal")
plt.show()
