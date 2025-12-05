import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

SEQ = "07"
IMG_DIR = f"/home/ho/Desktop/dataset/sequences/{SEQ}/image_0"
POSE_PATH = f"/home/ho/Desktop/dataset/data_odometry_poses/dataset/poses/{SEQ}.txt"

K = np.array([
    [7.070912e+02, 0.0,        6.018873e+02],
    [0.0,          7.070912e+02, 1.831104e+02],
    [0.0,          0.0,        1.0]
])

orb = cv.ORB_create(3000)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

poses = np.loadtxt(POSE_PATH).reshape(-1, 3, 4)
gt_pos = poses[:, :, 3]   # (N,3)

def get_absolute_scale(frame_id: int) -> float:
    """
    GT에서 frame_id-1 -> frame_id 의 이동 거리 [m]
    VO 속도에 곱할 scale factor로 사용.
    """
    if frame_id == 0:
        return 1.0
    p_prev = gt_pos[frame_id - 1]
    p_curr = gt_pos[frame_id]
    return float(np.linalg.norm(p_curr - p_prev))

def detect_and_describe(img):
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def match_descriptors(des1, des2, ratio=0.75):
    """
    Lowe ratio test + knnMatch 로 outlier 줄이기
    """
    if des1 is None or des2 is None:
        return []

    knn_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def estimate_motion(kp1, kp2, matches):
    """
    Essential -> R, t 추정
    """
    if len(matches) < 30:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    E, mask = cv.findEssentialMat(
        pts1, pts2, K,
        method=cv.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None, None

    _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K, mask=mask)

    inliers = int(mask_pose.sum())
    if inliers < 30:
        return None, None

    return R, t

def run_vo():
    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".png"))
    num_frames = len(files)
    assert num_frames > 1, "이미지가 너무 적음"

    print(f"[INFO] Seq {SEQ}, frames = {num_frames}")

    img_prev = cv.imread(os.path.join(IMG_DIR, files[0]), cv.IMREAD_GRAYSCALE)
    kp_prev, des_prev = detect_and_describe(img_prev)

    T = np.eye(4)
    traj = np.zeros((num_frames, 3))
    traj[0] = T[:3, 3]

    for i in range(1, num_frames):
        img = cv.imread(os.path.join(IMG_DIR, files[i]), cv.IMREAD_GRAYSCALE)
        kp, des = detect_and_describe(img)

        matches = match_descriptors(des_prev, des, ratio=0.75)
        R, t = estimate_motion(kp_prev, kp, matches)

        if R is not None and t is not None:
            scale = get_absolute_scale(i)
            t_scaled = t * scale

            angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
            if angle < np.deg2rad(20):
                T_inc = np.eye(4)
                T_inc[:3, :3] = R
                T_inc[:3, 3] = t_scaled.ravel()
                T = T @ T_inc

        traj[i] = T[:3, 3]

        kp_prev, des_prev = kp, des

        if i % 100 == 0:
            print(f"[INFO] frame {i}/{num_frames}")

    np.save(f"traj_seq_{SEQ}_gt_scale.npy", traj)
    print("[INFO] Trajectory saved:", f"traj_seq_{SEQ}_gt_scale.npy")

    plt.figure(figsize=(7, 6))
    plt.plot(traj[:, 0], traj[:, 2], label=f"Seq {SEQ} VO+GTscale")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_vo()