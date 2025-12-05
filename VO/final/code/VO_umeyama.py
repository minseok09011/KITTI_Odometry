import numpy as np
import matplotlib.pyplot as plt

SEQ = "09"

vo = np.load(f"traj_seq_{SEQ}_gt_scale.npy") 

poses = np.loadtxt(
    f"/home/ho/Desktop/dataset/data_odometry_poses/dataset/poses/{SEQ}.txt"
).reshape(-1, 3, 4)
gt = poses[:, :, 3]   # (N, 3)

N = min(len(vo), len(gt))
vo = vo[:N]
gt = gt[:N]

def umeyama_alignment(X, Y):
    """
    X: (N,3) VO trajectory
    Y: (N,3) GT trajectory
    Y ≈ s * R @ X + t  꼴의 s,R,t를 찾는 함수
    """
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    Xc = X - mu_X
    Yc = Y - mu_Y

    Sxy = (Xc.T @ Yc) / X.shape[0]

    U, D, Vt = np.linalg.svd(Sxy)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_X = np.sum(np.square(Xc)) / X.shape[0]
    scale = np.sum(D) / var_X

    t = mu_Y - scale * (R @ mu_X)

    return scale, R, t

s, R, t = umeyama_alignment(vo, gt)
print("[INFO] Umeyama scale =", s)

vo_aligned = (s * (R @ vo.T)).T + t

plt.figure(figsize=(7,6))
plt.plot(gt[:,0], gt[:,2], 'r', label="GT")
plt.plot(vo_aligned[:,0], vo_aligned[:,2], 'b', label="VO")
plt.xlabel("X [m]")
plt.ylabel("Z [m]")
plt.title(f"VO vs GT (Seq {SEQ})")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
