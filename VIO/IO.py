import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_sci

# ---------------------------------------------------------
# 1. 설정 및 데이터 로드
# ---------------------------------------------------------
base_path = '/home/minseok/Desktop/KITTI dataset' # 경로 확인 필요
seq = '07'
img_dir = os.path.join(base_path, f'data_odometry_gray/{seq}/image_0')
imu_dir = os.path.join(base_path, f'imu/{seq}/oxts')
pose_path = os.path.join(base_path, f'data_odometry_poses/dataset/poses/{seq}.txt')

file_list = sorted(os.listdir(img_dir))
file_count = len(file_list)

# Ground Truth 로드
poses_df = pd.read_csv(pose_path, delimiter=' ', header=None)
gt_vals = poses_df.values
pose_np = poses_df.to_numpy()
first_pose = np.reshape(pose_np[0,:], (3,-1))
print(first_pose)
gt = np.zeros((len(gt_vals), 3, 4))
for i in range(len(gt_vals)):
    gt[i] = np.array(gt_vals[i]).reshape((3, 4))

def load_imu_data(imu_dir):
    timestamp_path = os.path.join(imu_dir, 'timestamps.txt')
    data_path = os.path.join(imu_dir, 'data')
    df_time = pd.read_csv(timestamp_path, header=None, sep='\s+')
    timestamps = pd.to_datetime(df_time[0] + ' ' + df_time[1])
    time_sec = (timestamps - timestamps[0]).dt.total_seconds().values
    
    # KITTI OXTS 데이터 컬럼 (Raw Data 사용)
    cols = ['lat', 'lon','alt','roll','pitch','yaw','vn', 've','vf','vl','vu',
            'ax','ay','az','af','al','au','wx', 'wy', 'wz', 'wf', 'wl', 'wu', 
            'pos_accuracy', 'vel_accuracy', 'navstat', 'numstats', 'posmode', 'velmode', 'orimode']
    
    imu_files = sorted(os.listdir(data_path))
    imu_list = []
    for f in imu_files:
        p = os.path.join(data_path, f)
        df = pd.read_csv(p, header=None, sep='\s+', names=cols)
        imu_list.append(df)
        
    imu_df = pd.concat(imu_list, ignore_index=True)
    imu_df['timestamp'] = time_sec
    print("IMU 로딩 완료!")
    return imu_df

imu_df = load_imu_data(imu_dir)

#