import cv2
import numpy as np
import glob
import os

IMG_DIR = "/home/ho/INHA/ML/gray/10/image_0"
CALIB_PATH = "/home/ho/INHA/ML/gray/10/calib.txt"
SAVE_PATH = "estimated_kitti_10.txt"

def load_calib(filepath):
    with open(filepath) as f:
        P0 = [float(x) for x in f.readline().split()[1:]]
    P0 = np.array(P0).reshape(3,4)
    K = P0[:3,:3]
    return K

K = load_calib(CALIB_PATH)

img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
print(f"Loaded {len(img_paths)} images.")

orb = cv2.ORB_create(2500)

prev_img = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
prev_kp = orb.detect(prev_img, None)
prev_pts = np.array([kp.pt for kp in prev_kp], dtype=np.float32)

R_f = np.eye(3)
t_f = np.zeros((3,1))

poses = []

for i in range(1, len(img_paths)):

    curr_img = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None)
    good_prev = prev_pts[status.flatten()==1]
    good_curr = curr_pts[status.flatten()==1]

    if len(good_prev) < 8:
        prev_kp = orb.detect(curr_img, None)
        prev_pts = np.array([kp.pt for kp in prev_kp], dtype=np.float32)
        prev_img = curr_img
        continue

    E, mask = cv2.findEssentialMat(good_curr, good_prev, K, cv2.RANSAC, 0.999, 1.0)

    if E is None:
        prev_img = curr_img
        prev_pts = good_curr
        continue

    _, R, t, mask_pose = cv2.recoverPose(E, good_curr, good_prev, K)

    pts4d_h = cv2.triangulatePoints(
        np.hstack((R_f, t_f)),
        np.hstack((R_f @ R.T, t_f + R_f @ t)),
        good_prev.T, good_curr.T
    )
    pts4d = pts4d_h[:3] / pts4d_h[3]

    scale = np.median(np.abs(pts4d[2]))

    if scale > 50 or scale < 0.1:
        scale = 1.0

    t_f = t_f + R_f.dot(t) * scale
    R_f = R.dot(R_f)

    T = np.hstack((R_f, t_f))
    poses.append(" ".join(f"{v:.6e}" for v in T.flatten()))

    prev_img = curr_img
    prev_pts = good_curr


with open(SAVE_PATH, "w") as f:
    f.write("\n".join(poses))

print(f"Saved → {SAVE_PATH}")
