import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_sci

# ---------------------------------------------------------
# 1. 설정 및 데이터 로드
# ---------------------------------------------------------
base_path = '/home/minseok/Desktop/KITTI dataset'
seq = '04'

# 경로 설정 (Unsynced Raw Data 구조 가정)
# KITTI Raw 데이터는 보통 image_02/data 와 oxts/data로 나뉨
# 사용자의 경로에 맞춰 설정합니다.
img_dir = os.path.join(base_path, f'data_odometry_gray/{seq}/image_0')
# imu_dir는 unsynced 데이터가 있는 곳 (많은 양의 IMU 파일)
imu_dir = os.path.join(base_path, f'imu/{seq}/unsynced') 
pose_path = os.path.join(base_path, f'data_odometry_poses/dataset/poses/{seq}.txt')

# [중요] 이미지의 정확한 타임스탬프 로드
# KITTI Odometry 데이터셋의 times.txt 사용
times_path = os.path.join(base_path, f'data_odometry_gray/{seq}/times.txt')
image_times = pd.read_csv(times_path, header=None, sep='\s+').values.flatten()

file_list = sorted(os.listdir(img_dir))
file_count = len(file_list)

# Ground Truth 로드
poses_df = pd.read_csv(pose_path, delimiter=' ', header=None)
gt = np.zeros((len(poses_df), 3, 4))
for i in range(len(poses_df)):
    gt[i] = np.array(poses_df.iloc[i]).reshape((3, 4))
first_pose = gt[0]

# ---------------------------------------------------------
# 2. IMU 데이터 로드 (전체 로드)
# ---------------------------------------------------------
def load_imu_data_unsynced(imu_dir):
    # 타임스탬프 파일 로드
    timestamp_path = os.path.join(imu_dir, 'timestamps.txt')
    data_path = os.path.join(imu_dir, 'data')
    
    # 타임스탬프 파싱
    df_time = pd.read_csv(timestamp_path, header=None, sep='\s+')
    # 날짜+시간 형식을 초(second) 단위 float으로 변환
    timestamps = pd.to_datetime(df_time[0] + ' ' + df_time[1])
    # 첫 번째 이미지 시간 기준으로 0초부터 시작하도록 맞춤 (상대 시간)
    # 주의: 이미지 타임스탬프와 기준(0초)을 맞춰야 함. 
    # 여기서는 간단히 첫 IMU 시간을 0으로 잡고, 나중에 이미지 시간과 동기화 로직 필요
    # KITTI Odometry 'times.txt'는 0.0부터 시작하므로, IMU도 첫 데이터 기준으로 0.0 정렬
    base_time = timestamps[0]
    time_sec = (timestamps - base_time).dt.total_seconds().values
    
    cols = ['lat', 'lon','alt','roll','pitch','yaw','vn', 've','vf','vl','vu',
            'ax','ay','az','af','al','au','wx', 'wy', 'wz', 'wf', 'wl', 'wu', 
            'pos_accuracy', 'vel_accuracy', 'navstat', 'numstats', 'posmode', 'velmode', 'orimode']
    
    # 파일이 매우 많으므로(수천 개), 정렬 후 읽기
    imu_files = sorted(os.listdir(data_path))
    
    # 데이터 로딩 (pandas로 한 번에 읽는 게 빠름)
    imu_list = []
    print(f"Loading {len(imu_files)} IMU files... (This might take a moment)")
    
    # 성능을 위해 파일 하나하나 읽는 대신 덩어리로 처리 추천하지만,
    # 기존 코드 스타일 유지하되 리스트 컴프리헨션 사용
    # (실제로는 pandas read_csv가 많으면 느릴 수 있음. 여기서는 일단 진행)
    for f in imu_files:
        p = os.path.join(data_path, f)
        df = pd.read_csv(p, header=None, sep='\s+', names=cols)
        imu_list.append(df)

    imu_df = pd.concat(imu_list, ignore_index=True)
    imu_df['timestamp'] = time_sec
    print("IMU Unsynced Loading Done!")
    return imu_df

imu_df = load_imu_data_unsynced(imu_dir)
imu_timestamps = imu_df['timestamp'].values

# ---------------------------------------------------------
# 3. 초기화 (Camera Frame 기준: X=Right, Y=Down, Z=Fwd)
# ---------------------------------------------------------
# 초기 자세 계산을 위해 초반 10개의 IMU 데이터를 사용
init_ax_cam = np.mean(-imu_df['ay'][:10].values) # Right
init_ay_cam = np.mean(-imu_df['az'][:10].values) # Down
init_az_cam = np.mean(imu_df['ax'][:10].values)  # Fwd

def get_initial_attitude_cam(ax_mean, ay_mean, az_mean):
    norm = np.sqrt(ax_mean**2 + ay_mean**2 + az_mean**2)
    ax = ax_mean / norm
    ay = ay_mean / norm
    az = az_mean / norm
    
    # Camera Frame 기준 (Y-Down, Z-Fwd)
    pitch = -np.arcsin(az)       # 전진 가속도(Z) 기반 Pitch
    roll = np.arctan2(-ax, -ay)  # 횡방향(X), 수직(Y) 기반 Roll
    yaw = 0 
    return roll, pitch, yaw

r0, p0, y0 = get_initial_attitude_cam(init_ax_cam, init_ay_cam, init_az_cam)
print(f"Init Attitude -> Roll: {np.degrees(r0):.2f}, Pitch: {np.degrees(p0):.2f}")


# ---------------------------------------------------------
# 4. EKF 클래스 (9-State, Scale Factor 적용)
# ---------------------------------------------------------
class ekf():
    def __init__(self, px, py, pz, roll, pitch, yaw):
        self.x = np.array([px, py, pz, 0, 0, 0, roll, pitch, yaw])
        self.P = np.eye(9)
        self.Q = np.eye(9) * 0.01 
        self.g = np.array([0, 9.81, 0]) # Camera Frame Gravity (+Y is Down)

        self.VO_first_pose = np.eye(4)
        self.VO_first_pose[0:3, 3] = [px, py, pz]
        self.VO_cov = np.eye(6) * 0.1 

    def predict(self, ax, ay, az, wx, wy, wz, dt):
        p = self.x[0:3]
        v = self.x[3:6]
        euler = self.x[6:9]

        # 1. 자세 적분
        euler_new = euler + np.array([wx, wy, wz]) * dt
        
        # 2. 가속도 변환 (Camera Frame 기준 회전 + 중력 보정)
        R_mat = R_sci.from_euler('xyz', euler).as_matrix()
        acc_body = np.array([ax, ay, az])
        
        # [중요] 중력 제거 로직 (성공했던 로직 유지)
        # Acc_World = R * Acc_Body + Gravity_Vector
        acc_world = (R_mat @ acc_body) + self.g 

        # 3. 위치/속도 적분
        p_new = p + v * dt + 0.5 * acc_world * dt**2
        v_new = v + acc_world * dt
        
        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:9] = euler_new
        
        # 공분산 예측 (간략화)
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x

    def update(self, R_vo, t_vo, dt):
        current_speed = np.linalg.norm(self.x[3:6])
        if current_speed < 0.1: return 

        # Scale Factor (IMU 적분 정확도가 올라가면 1.0에 가까워도 됨)
        # 하지만 여전히 Bias 등이 있으므로 약간의 보정(1.0~1.1)이 필요할 수 있음
        scale_factor = 1.2
        scale = current_speed * dt * scale_factor
        t_metric = t_vo * scale
        
        # VO Update 로직 (기존 유지)
        T_local = np.eye(4)
        T_local[0:3, 0:3] = R_vo
        T_local[0:3, 3] = t_metric.ravel()
        
        self.VO_first_pose = self.VO_first_pose @ np.linalg.inv(T_local)
        
        z_meas = np.zeros(6)
        z_meas[0:3] = self.VO_first_pose[0:3, 3]
        
        residual = np.zeros(6)
        residual[0:3] = z_meas[0:3] - self.x[0:3]
        
        H = np.zeros((6,9))
        H[0:3, 0:3] = np.eye(3)
        
        S = H @ self.P @ H.T + self.VO_cov
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ residual
        self.P = (np.eye(9) - K @ H) @ self.P

# ---------------------------------------------------------
# 5. VO 설정
# ---------------------------------------------------------
orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
class VO():
    def __init__(self):
        self.camera_matrix = np.array([[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02], 
                         [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02],
                         [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
    
    def orb_detect(self, img):
        return orb.detectAndCompute(img, None)
    
    def bfmatcher(self, des, des_next):
        matches = bf.match(des, des_next)
        return sorted(matches, key = lambda x:x.distance)
    
    def find_R_t(self, kp, kp_next, matches):
        kp_pts = np.float32([ kp[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
        src_pts = np.float32([ kp_next[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)
        essential, mask = cv.findEssentialMat(kp_pts, src_pts, self.camera_matrix, method= cv.RANSAC, prob = 0.999, threshold= 0.9)
        if essential is None: return np.eye(3), np.zeros(3, 1)
        _, R, t, _ = cv.recoverPose(essential, kp_pts, src_pts, self.camera_matrix, mask) 
        return R, t

# ---------------------------------------------------------
# 6. 메인 루프 (Multi-Step Propagation 적용)
# ---------------------------------------------------------
ekf_estimator = ekf(first_pose[0][3], first_pose[1][3], first_pose[2][3], r0, p0, y0)
vo_estimator = VO()

trajectory = []
trajectory.append(ekf_estimator.x[0:3].copy())

imu_ptr = 0 # IMU 데이터 인덱스 포인터 (검색 효율화)
print("Processing with Multi-Step IMU Propagation...")

for i in range(file_count - 1):
    # 1. 이미지 시간 정보 가져오기
    # image_times는 0.0초부터 시작하는 상대 시간이라고 가정 (위 load 함수에서 맞춰줘야 함)
    # 만약 KITTI times.txt가 그냥 float 초 단위라면 그대로 쓰면 됩니다.
    t_curr_img = image_times[i+1]
    
    # 2. VO 수행 (이전 이미지와 현재 이미지)
    img_path1 = os.path.join(img_dir, str(i).zfill(6) + '.png')
    img_path2 = os.path.join(img_dir, str(i+1).zfill(6) + '.png')
    
    img1 = cv.imread(img_path1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img_path2, cv.IMREAD_GRAYSCALE)
    
    R_vo, t_vo = np.eye(3), np.zeros((3,1))
    vo_valid = False
    
    if img1 is not None and img2 is not None:
        kp1, des1 = vo_estimator.orb_detect(img1)
        kp2, des2 = vo_estimator.orb_detect(img2)
        if des1 is not None and des2 is not None:
            matches = vo_estimator.bfmatcher(des1, des2)
            R_vo, t_vo = vo_estimator.find_R_t(kp1, kp2, matches)
            if not np.array_equal(R_vo, np.eye(3)):
                vo_valid = True

    # 3. IMU Propagation (이미지 사이의 모든 IMU 데이터 적분)
    # 현재 이미지 시간(t_curr_img)보다 작거나 같은 모든 IMU 데이터를 처리
    
    propagation_count = 0
    while imu_ptr < len(imu_timestamps) and imu_timestamps[imu_ptr] <= t_curr_img:
        # dt 계산
        if imu_ptr == 0:
            dt_imu = 0.01 # 첫 프레임은 대략 100Hz 가정
        else:
            dt_imu = imu_timestamps[imu_ptr] - imu_timestamps[imu_ptr-1]
            if dt_imu > 0.1: dt_imu = 0.01 # 비정상적으로 큰 점프 예외처리
        
        # 데이터 매핑 (Camera Frame: X=Right, Y=Down, Z=Fwd)
        # DataFrame에서 현재 포인터의 값을 가져옴
        raw_ax = imu_df.at[imu_ptr, 'ax']
        raw_ay = imu_df.at[imu_ptr, 'ay']
        raw_az = imu_df.at[imu_ptr, 'az']
        raw_wx = imu_df.at[imu_ptr, 'wx']
        raw_wy = imu_df.at[imu_ptr, 'wy']
        raw_wz = imu_df.at[imu_ptr, 'wz']

        # Camera Frame으로 변환해서 입력
        # ax_cam(Right) = -ay_imu
        # ay_cam(Down)  = -az_imu
        # az_cam(Fwd)   =  ax_imu
        in_ax = -raw_ay
        in_ay = -raw_az
        in_az =  raw_ax
        
        in_wx = -raw_wy
        in_wy = -raw_wz
        in_wz =  raw_wx
        
        # EKF 예측 실행
        ekf_estimator.predict(in_ax, in_ay, in_az, in_wx, in_wy, in_wz, dt_imu)
        
        imu_ptr += 1
        propagation_count += 1
    
    # 4. VO Update (IMU 적분이 끝난 시점에서 수행)
    # dt는 이미지 간의 시간 차이
    dt_img = image_times[i+1] - image_times[i]
    if vo_valid:
        ekf_estimator.update(R_vo, t_vo, dt_img)
    
    # 결과 저장
    trajectory.append(ekf_estimator.x[0:3].copy())
    
    if i % 100 == 0:
        print(f"Frame {i}/{file_count} - Propagated {propagation_count} IMU steps")

est_traj = np.array(trajectory).T

# 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.plot(gt[:, 0, 3], gt[:, 2, 3], label='Ground Truth', color='r')
ax.plot(est_traj[0, :], est_traj[2, :], label='VIO Estimate', color='b')
ax.set_xlabel('x (m) - Right')
ax.set_ylabel('z (m) - Forward')
ax.legend()
ax.axis('equal')
plt.grid()
plt.show()