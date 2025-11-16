import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#########################################################
directory_path = '/home/minseok/Desktop/KITTI dataset/data_odometry_gray/04/image_0'
file_list = len(os.listdir(directory_path))
poses_df = pd.read_csv('/home/minseok/Desktop/KITTI dataset/data_odometry_poses/dataset/poses/04.txt', delimiter=' ', header=None)
annotations = poses_df.values

def getAbsoluteScale(frame_id):
    ss = annotations[frame_id-1]
    x_prev = float(ss[3])
    y_prev = float(ss[7])
    z_prev = float(ss[11])
    ss = annotations[frame_id]
    x = float(ss[3])
    y = float(ss[7])
    z = float(ss[11])
    return np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)
#########################################################

camera_matrix = np.array([[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02], 
                         [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02],
                         [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])

first_pose = np.array([[1.000000e+00, 1.197625e-11, 1.704638e-10, -5.551115e-17], 
                      [1.197625e-11, 1.000000e+00, 3.562503e-10, 0.000000e+00], 
                      [1.704638e-10, 3.562503e-10, 1.000000e+00, 2.220446e-16],
                      [0,0,0,1]])

first_rotation = np.array([[1.000000e+00, 1.197625e-11, 1.704638e-10],
                          [1.197625e-11, 1.000000e+00, 3.562503e-10],
                          [1.704638e-10, 3.562503e-10, 1.000000e+0]])

X_Y_Z = np.zeros((3, file_list))
X_Y_Z[0][0] = -5.551115e-17
X_Y_Z[1][0] = 0
X_Y_Z[2][0] = 2.220446e-16

#########################################################
orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

def orb_detect(img):
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def bfmatcher(des, des_next):
    matches = bf.match(des, des_next)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def find_R_t(kp, kp_next, matches):
    kp_pts = np.float32([ kp[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
    src_pts = np.float32([ kp_next[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)
    essential, mask = cv.findEssentialMat(kp_pts, src_pts, camera_matrix, method= cv.RANSAC, prob = 0.999, threshold= 1.0)
    
    if essential is None:
        return np.eye(3), np.zeros(3, 1)

    rtv, R, t, mask = cv.recoverPose(essential, kp_pts, src_pts, camera_matrix, mask) 
    return R, t

for i in range(file_list - 1):
    img_path1 = os.path.join(directory_path, str(i).zfill(6) + '.png')
    img_path2 = os.path.join(directory_path, str(i+1).zfill(6) + '.png')
    img = cv.imread(img_path1)
    img_next = cv.imread(img_path2)
    kp, des = orb_detect(img)
    kp_next, des_next= orb_detect(img_next)
    matches = bfmatcher(des, des_next)
    R, t = find_R_t(kp, kp_next, matches)
    scale = getAbsoluteScale(i + 1)
    t_scale = scale * t
    pose_matrix = np.eye(4)
    pose_matrix[0:3, 0:3] = R.T
    pose_matrix[0:3, 3] = (-R.T @ t_scale).ravel()
    first_pose = first_pose @ pose_matrix
    X_Y_Z[0][i + 1] = first_pose[0][3] #x
    X_Y_Z[1][i + 1] = first_pose[1][3] #y
    X_Y_Z[2][i + 1] = first_pose[2][3] #z

fig = plt.figure(figsize = (7,6))
ax = fig.add_subplot(111)
ax.plot(X_Y_Z[0][:], X_Y_Z[2][:])
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.axis('equal')
plt.grid(True)
plt.show()
