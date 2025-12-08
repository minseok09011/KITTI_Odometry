import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import datetime

SEQ = "09"

POSE_PATH = "/home/ho/Desktop/dataset/data_odometry_poses/dataset/poses/09.txt"

RAW_ROOT = "/home/ho/Desktop/kitti_raw/2011_09_30_drive_0033_sync/2011_09_30/2011_09_30_drive_0033_sync"
OXTS_DIR = os.path.join(RAW_ROOT, "oxts")
OXTS_DATA_DIR = os.path.join(OXTS_DIR, "data")

VO_IMG_DIR = f"/home/ho/Desktop/dataset/sequences/{SEQ}/image_0"

K = np.array([
    [7.070912e+02, 0.0,        6.018873e+02],
    [0.0,          7.070912e+02, 1.831104e+02],
    [0.0,          0.0,        1.0]
])

orb = cv.ORB_create(3000)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)


def wrap_angle(a):
    """[-pi, pi] 범위로 angle wrapping"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_gt_poses(pose_path):
    """
    GT pose 로드: (N, 3, 4) + x,z 궤적
    """
    poses = np.loadtxt(pose_path).reshape(-1, 3, 4)
    gt_pos = poses[:, :, 3]
    gt_x = gt_pos[:, 0]
    gt_z = gt_pos[:, 2]
    return poses, gt_x, gt_z


def load_oxts(oxts_dir, max_frames=None):
    """
    OXTS에서 yaw(heading), speed, dt 배열 로드
    - yaw: rad, 길이 N
    - speed: m/s, 길이 N
    - dt: 초, 길이 N
    """
    data_dir = os.path.join(oxts_dir, "data")
    files = sorted(os.listdir(data_dir))
    if max_frames is not None:
        files = files[:max_frames]
    N = len(files)

    yaw = np.zeros(N)
    speed = np.zeros(N)

    for i, fname in enumerate(files):
        vals = np.loadtxt(os.path.join(data_dir, fname))
        yaw[i] = vals[5]          
        vn, ve = vals[6], vals[7]    
        speed[i] = np.hypot(vn, ve)

    ts_path = os.path.join(oxts_dir, "timestamps.txt")
    dt = np.full(N, 0.1, dtype=float)  

    try:
        with open(ts_path, "r") as f:
            lines = [line.strip() for line in f.readlines()[:N]]

        if len(lines) == N:
            t_list = []
            t0 = None
            for line in lines:
                if not line:
                    t_list.append(None)
                    continue
                dt_obj = datetime.datetime.fromisoformat(line)
                if t0 is None:
                    t0 = dt_obj
                t_sec = (dt_obj - t0).total_seconds()
                t_list.append(t_sec)

            t_arr = np.array(t_list, dtype=float)
            dt[1:] = np.diff(t_arr)
            dt = np.clip(dt, 0.01, 0.2)
            dt[0] = dt[1]
    except Exception:
        pass

    return yaw, speed, dt


def detect_and_describe(img):
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def match_descriptors(des1, des2, ratio=0.75):
    if des1 is None or des2 is None:
        return []
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def estimate_R(kp1, kp2, matches):
    """
    Essential → R 추정 (상대 회전)
    """
    if len(matches) < 30:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    E, mask = cv.findEssentialMat(
        pts1, pts2, K,
        method=cv.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None

    _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K, mask=mask)
    inliers = int(mask_pose.sum())
    if inliers < 30:
        return None

    return R


def umeyama_alignment(X, Y):
    """
    Umeyama Similarity Alignment
    X, Y: (N,3),  Y ≈ s * R @ X + t
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


def run_vio():
    gt_poses, gt_x, gt_z = load_gt_poses(POSE_PATH)
    N_gt = gt_poses.shape[0]

    yaw_imu, speed_imu, dt_imu = load_oxts(OXTS_DIR, max_frames=N_gt)
    N_imu = len(yaw_imu)

    vo_files = sorted(f for f in os.listdir(VO_IMG_DIR) if f.endswith(".png"))
    N_vo = len(vo_files)

    N = min(N_gt, N_imu, N_vo)
    print(f"[INFO] GT: {N_gt}, IMG: {N_vo}, OXTS: {N_imu} → 사용 프레임: {N}")

    yaw_imu = yaw_imu[:N]
    speed_imu = speed_imu[:N]
    dt_imu = dt_imu[:N]
    gt_pos = gt_poses[:N, :, 3]

    yaw_vo_rel = np.zeros(N)
    vo_valid = np.zeros(N, dtype=bool)

    img_prev = cv.imread(os.path.join(VO_IMG_DIR, vo_files[0]), cv.IMREAD_GRAYSCALE)
    kp_prev, des_prev = detect_and_describe(img_prev)

    for i in range(1, N):
        img = cv.imread(os.path.join(VO_IMG_DIR, vo_files[i]), cv.IMREAD_GRAYSCALE)
        kp, des = detect_and_describe(img)

        matches = match_descriptors(des_prev, des, ratio=0.75)
        R = estimate_R(kp_prev, kp, matches)

        if R is not None:
            delta_yaw = np.arctan2(R[0, 2], R[2, 2])
            yaw_vo_rel[i] = yaw_vo_rel[i-1] + delta_yaw
            vo_valid[i] = True
        else:
            yaw_vo_rel[i] = yaw_vo_rel[i-1]

        kp_prev, des_prev = kp, des

        if i % 100 == 0:
            print(f"[INFO] VO frame {i}/{N}")

    yaw_vo_abs = wrap_angle(yaw_vo_rel + yaw_imu[0])

    vio_pos = np.zeros((N, 2))
    vo_pos = np.zeros((N, 2))

    yaw_est = np.zeros(N)
    yaw_est[0] = yaw_imu[0]

    yaw_vo_used = np.zeros(N)
    yaw_vo_used[0] = yaw_vo_abs[0]

    alpha = 0.15

    for i in range(1, N):
        if vo_valid[i]:
            yaw_fused = wrap_angle((1.0 - alpha) * yaw_imu[i] + alpha * yaw_vo_abs[i])
        else:
            yaw_fused = yaw_imu[i]

        yaw_est[i] = yaw_fused

        yaw_vo_used[i] = yaw_vo_abs[i] if vo_valid[i] else yaw_vo_used[i-1]

        v = speed_imu[i]
        dt = dt_imu[i]

        vio_pos[i, 0] = vio_pos[i-1, 0] + v * dt * np.cos(yaw_est[i])
        vio_pos[i, 1] = vio_pos[i-1, 1] + v * dt * np.sin(yaw_est[i])

        vo_pos[i, 0] = vo_pos[i-1, 0] + v * dt * np.cos(yaw_vo_used[i])
        vo_pos[i, 1] = vo_pos[i-1, 1] + v * dt * np.sin(yaw_vo_used[i])

    vio_traj3 = np.column_stack([vio_pos[:, 0], np.zeros(N), vio_pos[:, 1]])
    vo_traj3  = np.column_stack([vo_pos[:, 0], np.zeros(N), vo_pos[:, 1]])
    gt_traj3  = gt_pos[:, :3]

    s_vio, R_vio, t_vio = umeyama_alignment(vio_traj3, gt_traj3)
    s_vo,  R_vo,  t_vo  = umeyama_alignment(vo_traj3,  gt_traj3)

    vio_aligned = (s_vio * (R_vio @ vio_traj3.T)).T + t_vio
    vo_aligned  = (s_vo  * (R_vo  @ vo_traj3.T)).T  + t_vo

    print(f"[INFO] Umeyama scale VIO = {s_vio:.3f}, VO-only = {s_vo:.3f}")

    plt.figure(figsize=(7, 6))
    plt.plot(gt_pos[:, 0], gt_pos[:, 2], 'k', label="GT")
    plt.plot(vio_aligned[:, 0], vio_aligned[:, 2], 'b', label="VIO (IMU speed + yaw fusion)")
    plt.plot(vo_aligned[:, 0], vo_aligned[:, 2], 'g--', label="VO-only (IMU speed + VO yaw)")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title(f"Seq {SEQ} - VIO vs VO vs GT")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_vio()
