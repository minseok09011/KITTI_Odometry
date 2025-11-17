< 결과 >
00
Loaded 4541 images.
estimated_kitti.txt
GT poses: 4497, EST poses: 4497
=== Evaluation Results ===
APE RMSE: 180.976 m
RPE RMSE: 0.973 m


01
Loaded 1101 images.
estimated_kitti_01.txt
GT poses: 1096, EST poses: 1096
=== Evaluation Results ===
APE RMSE: 416.537 m
RPE RMSE: 10.527 m

02
Loaded 4661 images.
estimated_kitti_02.txt
GT poses: 4611, EST poses: 4611
=== Evaluation Results ===
APE RMSE: 289.725 m
RPE RMSE: 1.559 m

03
Loaded 801 images.
estimated_kitti_03.txt
GT poses: 796, EST poses: 796
=== Evaluation Results ===
APE RMSE: 98.888 m
RPE RMSE: 3.713 m

04
Loaded 271 images.
estimated_kitti_04.txt
GT poses: 270, EST poses: 270
=== Evaluation Results ===
APE RMSE: 22.450 m
RPE RMSE: 4.509 m

05
Loaded 2761 images.
Saved → estimated_kitti_05.txt
GT poses: 2741, EST poses: 2741
=== Evaluation Results ===
APE RMSE: 136.457 m
RPE RMSE: 1.222 m

06
Loaded 1101 images.
Saved → estimated_kitti_06.txt
GT poses: 1093, EST poses: 1093
=== Evaluation Results ===
APE RMSE: 127.516 m
RPE RMSE: 1.682 m

07
Loaded 1101 images.
Saved → estimated_kitti_07.txt
GT poses: 1091, EST poses: 1091
=== Evaluation Results ===
APE RMSE: 69.268 m
RPE RMSE: 1.858 m

08
Loaded 4071 images.
Saved → estimated_kitti_08.txt
GT poses: 4034, EST poses: 4034
=== Evaluation Results ===
APE RMSE: 201.669 m
RPE RMSE: 2.279 m

09
Loaded 1591 images.
Saved → estimated_kitti_09.txt
GT poses: 1575, EST poses: 1575
=== Evaluation Results ===
APE RMSE: 163.478 m
RPE RMSE: 2.137 m

10
Loaded 1201 images.
Saved → estimated_kitti_10.txt
GT poses: 1189, EST poses: 1189
=== Evaluation Results ===
APE RMSE: 125.976 m
RPE RMSE: 2.832 m


-----------------------------------------------------------
APE 수치가 높은거 같아서 논문 찾아보니까
ORB-SLAM2 Monocular VO: APE 100~300m (KITTI Seq 00)
DeepVO Monocular: APE 150~350m (KITTI Seq 00)
VISO2-Mono: APE 200m 이상
은 괜찮은 수치라고 하더라고 -> 정상범위

< if VO 성능 개선하려면 >
(Keyframe 기반 VO / Optical flow 기반 VO)
1. Scale Drift 줄이기 : Translation Filtering / Smooth Scaling
  - Translation smoothing
  - Dynamic scale estimation (Pseudo scale)
  - Use 5-point algorithm + Essential decomposition carefully
2. Rotation Drift 줄이기: Keyframe-based VO
 - 매 프레임 VO -> Keyframe 기반 VO
 - Keyframe-to-keyframe Essential
