import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

poses = pd.read_csv('/home/minseok/Desktop/KITTI dataset/data_odometry_poses/dataset/poses/11.txt', delimiter = ' ', header = None)
print('Shape of position dataframe: ', poses.shape)
print(poses.head())

print('First position: ')
first_pose = np.array(poses.iloc[0]).reshape((3,4)).round(2)
print(first_pose)

gt = np.zeros((len(poses), 3,4))
for i in range(len(poses)):
    gt[i] = np.array(poses.iloc[i]).reshape((3,4))
gt[1].dot(np.array([0,0,0,1]))

fig = plt.figure(figsize = (7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt[:, :, 3][:, 0], gt[:, :, 3][:, 1], gt[:, :, 3][:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=-40, azim=270)

plt.show()