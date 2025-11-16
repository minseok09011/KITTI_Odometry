import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

poses = pd.read_csv('/home/minseok/Desktop/KITTI dataset/data_odometry_poses/dataset/poses/04.txt', delimiter = ' ', header = None)
# ... (데이터 로드 코드 생략) ...
gt = np.zeros((len(poses), 3,4))
for i in range(len(poses)):
    gt[i] = np.array(poses.iloc[i]).reshape((3,4))

fig = plt.figure(figsize = (7,6))
ax = fig.add_subplot(111)

# --- [수정된 부분] ---
# Y축 데이터 (gt[:, :, 3][:, 1])를 Z축 데이터 (gt[:, :, 3][:, 2])로 변경합니다.
ax.plot(gt[:, :, 3][:, 0], gt[:, :, 3][:, 2]) 

ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)') # 라벨도 y에서 z로 변경

# X축과 Z축의 스케일을 동일하게 맞춰 실제 궤적 비율을 봅니다. (권장)
ax.axis('equal') 
plt.grid(True)
ax.set_title('Ground Truth Trajectory (Top-Down View - Seq 04)')
# --- [수정 완료] ---

plt.show()