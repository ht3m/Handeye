# Handeye Calibration 使用说明

本项目用于 UR 机械臂与 Intel RealSense D405 的手眼标定。当前主流程面向 Eye-on-Hand 场景：相机安装在机械臂末端，使用 ArUco 标记板作为标定目标。

## 环境依赖

建议使用 Python 3.10 或更新版本，并在独立虚拟环境中安装依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

硬件和软件要求：

- UR 机械臂，电脑需要能通过 TCP 连接到控制器，默认端口是 `30003`。
- Intel RealSense D405。
- Intel RealSense SDK，以及 Python 包 `pyrealsense2`。
- OpenCV ArUco 模块。若出现 `ModuleNotFoundError: cv2.aruco`，需要安装 `opencv-contrib-python`，不能只安装 `opencv-python`。
- 标定目标：ArUco 标记板，字典、ID、边长需要和 [handeye/config.py](handeye/config.py) 中的配置一致。

## 基础配置

运行前先修改 [handeye/config.py](handeye/config.py)。

主要配置项：

```python
UR5_CONFIG = {
    'tcp_host_ip': '169.254.162.111',
    'tcp_port': 30003,
}
```

- `UR5_CONFIG['tcp_host_ip']`：UR 控制器 IP。
- `UR5_CONFIG['tcp_port']`：UR TCP 端口，通常是 `30003`。

```python
REALSENSE_CONFIG = {
    'device_id': None,
    'width': 1280,
    'height': 720,
    'fps': 30,
}
```

- `device_id`：为 `None` 时自动选择第一台 RealSense。
- `width`、`height`、`fps`：相机采集分辨率和帧率。
- 相机内参不需要手动写进 `config.py`。运行 `main.py` 时，程序会从 RealSense 自动读取当前真实内参和畸变参数。

```python
ARUCO_CONFIG = {
    'dictionary': 'DICT_ARUCO_ORIGINAL',
    'marker_id': 159,
    'marker_size': 0.15,
}
```

- `dictionary`：ArUco 字典类型，必须和打印标记时使用的字典一致。
- `marker_id`：目标 ArUco ID。
- `marker_size`：ArUco 标记边长，单位是米。例如 15 cm 写 `0.15`。

```python
CALIBRATION_MODE = 'eye_on_hand'
CALIBRATION_BACKEND = 'aruco'
```

当前主流程使用 `eye_on_hand + aruco`。

## 在线标定工作流

在项目根目录运行：

```powershell
python scripts\main.py
```

程序会连接机器人和 RealSense。RealSense 连接成功后会打印当前相机内参、畸变参数和深度缩放，这些参数是运行时自动读取的。

采集界面按键：

- `Space`：检测当前画面中的 ArUco 标记。
- `Enter`：保存当前已检测到的样本。
- `Backspace`：取消当前待保存检测。
- `Esc`：结束采集。结束后程序会自动读取已保存样本并计算手眼标定矩阵。

每个有效样本会保存机器人 TCP 位姿、ArUco 位姿、角点和图像数据。默认位置：

```text
data/eye_on_hand/
  poses/
    tcp_001.txt
    tag_pose_001.txt
    tag_corners_001.txt
  images/
    rgb_001.png
    depth_001.npy
```

重要：运行 `main.py` 不会自动删除 `data/` 里的老数据。新采集的数据会继续在已有编号后面递增保存。例如已有 `001-014`，新数据会从 `015` 开始。计算矩阵时，如果旧的 `poses/tcp_XXX.txt` 和 `poses/tag_pose_XXX.txt` 仍然存在，它们可能继续参与计算。

如果要重新做一组干净标定，建议先清空：

```text
data/eye_on_hand/poses/
data/eye_on_hand/images/
```

如果只想删除某些旧样本，至少要删除对应编号的：

```text
tcp_XXX.txt
tag_pose_XXX.txt
```

只删除 `tag_corners_XXX.txt` 或只删除图片，并不能保证旧样本不参与在线标定计算。

## 标定结果

在线标定或离线标定完成后，结果写入：

```text
results/eye_on_hand/
  handeye_transform.txt
  depth_scale.txt
  calibration_info.txt
```

- `handeye_transform.txt`：4x4 手眼变换矩阵。Eye-on-Hand 下解释为 `T_tcp_camera`，即相机坐标系在 TCP 坐标系下的位姿。
- `depth_scale.txt`：优化得到的深度缩放因子。
- `calibration_info.txt`：标定模式、深度缩放和矩阵摘要。

当前矩阵保存格式示例：

```python
[
    [0.351230, -0.714777, 0.604677, -0.052565],
    [-0.053744, 0.627255, 0.776953, -0.036955],
    [-0.934741, -0.305386, 0.181896, 0.067569],
    [0.0, 0.0, 0.0, 1.0],
]
```

## 评估标定结果

运行：

```powershell
python scripts\evaluate_calibration.py
```

该脚本会读取：

```text
results/<mode>/handeye_transform.txt
results/<mode>/depth_scale.txt
```

然后重新采集一组独立验证样本，并输出一致性误差。

评估方法概念：

- 对每个验证样本，程序读取当前机器人 TCP 位姿和相机观测到的 ArUco 位姿。
- 使用已标定的手眼矩阵把不同样本转换到同一坐标关系中。
- 如果标定准确，不同姿态下推算出的目标位置和姿态应该一致。

输出含义：

- `Position consistency error`：位置一致性误差，单位是米。越小越好。
- `Rotation consistency error`：旋转一致性误差，单位是度。越小越好。
- 报告中的均值、最大值、标准差用于判断整体稳定性和是否存在离群样本。

评估时需要重新把 ArUco 标记放在相机视野内，并按照提示采集验证样本。验证样本应尽量和标定样本不同，不要只重复同一姿态。

## 直接查看当前 ArUco 中心坐标

运行：

```powershell
python tests\compute_eye_on_hand_tag_base_xyz.py
```

该脚本用于在 Eye-on-Hand 场景下计算当前 ArUco 中心点在机器人基座坐标系下的 XYZ。

默认行为：

- 从机器人读取当前 TCP 位姿。
- 从 RealSense 当前画面检测 `ARUCO_CONFIG` 中配置的 ArUco。
- 读取 `results/eye_on_hand/handeye_transform.txt`。
- 输出 ArUco 中心在 base 坐标系下的位置。

运行要求：

- 机器人连接正常。
- RealSense 连接正常。
- ArUco 目标必须在相机画面内，并且 ID、字典、尺寸和 `ARUCO_CONFIG` 一致。
- 如果目标不在画面内，会报类似 `Failed to detect ArUco id=...` 的错误。

常用参数：

```powershell
python tests\compute_eye_on_hand_tag_base_xyz.py --camera-frames 120
python tests\compute_eye_on_hand_tag_base_xyz.py --quiet
python tests\compute_eye_on_hand_tag_base_xyz.py --tcp-pose 0.1 0.2 0.3 0 0 0
python tests\compute_eye_on_hand_tag_base_xyz.py --tag-camera 0.01 0.02 0.30
python tests\compute_eye_on_hand_tag_base_xyz.py --handeye-file results\eye_on_hand\handeye_transform.txt
```

- `--camera-frames`：最多尝试多少帧检测 ArUco。
- `--quiet`：只输出三个数字，便于其他程序读取。
- `--tcp-pose`：手动指定 TCP 位姿，不从机器人读取。
- `--tag-camera`：手动指定 tag 中心在相机坐标系下的位置，不进行实时检测。
- `--handeye-file`：指定要使用的手眼矩阵文件。

## 离线标定和评估

如果已经采集过数据，可以不连接相机和机器人，直接离线读取 `data/<mode>/` 重新计算矩阵：

```powershell
python scripts\offline_calibrate_and_evaluate.py
```

该脚本读取：

```text
data/eye_on_hand/poses/
data/eye_on_hand/images/
```

然后执行和在线流程相同的求解与误差计算，并把结果写入 `results/eye_on_hand/`。

注意：离线脚本当前会跳过缺少对应 RGB 图片的样本。因此如果你删除了旧图片，旧样本不会被离线脚本使用。但在线 `main.py` 的数据读取逻辑不等同于离线脚本，重新在线标定前仍建议清理旧 `poses`。

## 可视化手眼坐标系

运行：

```powershell
python scripts\visualize_handeye_frames.py --mode eye_on_hand
```

该脚本读取 `handeye_transform.txt`，显示 TCP 坐标系和相机光学坐标系的相对位置，用于检查相机安装方向和矩阵方向是否合理。

常用参数：

```powershell
python scripts\visualize_handeye_frames.py --transform results\eye_on_hand\handeye_transform.txt
python scripts\visualize_handeye_frames.py --mode eye_on_hand --save results\eye_on_hand\frames.png --no-show
python scripts\visualize_handeye_frames.py --mode eye_on_hand --inverse
```

`--inverse` 只建议作为诊断使用，用来判断是否把矩阵方向理解反了。

## 项目结构

```text
handeye/
  config.py                    # 全局配置：机器人 IP、相机分辨率、ArUco 参数、标定模式
  calibration_solver.py        # 标定主求解流程，保存 handeye_transform/depth_scale/calibration_info
  data_collector.py            # 在线采集逻辑，保存 TCP、tag_pose、图像和角点
  device_manager.py            # 机器人和相机连接管理
  error_calculator.py          # 评估误差计算
  result_visualizer.py         # 结果和误差可视化工具
  transform_io.py              # 手眼矩阵读写工具

handeye/calibration/
  feature_extractor.py         # ArUco 和棋盘格检测
  optimizer.py                 # 非线性优化
  solver_axxb.py               # AX=XB 手眼求解
  transforms.py                # 位姿和变换矩阵工具
  svd.py                       # SVD 实验相关工具

handeye/camera/
  realsense.py                 # RealSense D405 封装，读取图像、深度、内参和畸变

handeye/robot/
  ur_robot.py                  # UR 机器人 TCP 位姿读取和位姿转换

scripts/
  main.py                      # 在线采集并标定
  evaluate_calibration.py      # 在线采集验证样本并评估已有矩阵
  offline_calibrate_and_evaluate.py
                               # 从 data 离线读取样本，重新计算矩阵并评估
  visualize_handeye_frames.py  # 可视化 TCP 和相机坐标系

tests/
  compute_eye_on_hand_tag_base_xyz.py
                               # 计算当前 ArUco 中心在机器人 base 坐标系下的 XYZ
  move_tcp_eye_to_hand_apriltag.py
                               # Eye-to-Hand/AprilTag 目标定位测试工具
  capture_svd_data.py          # SVD 实验数据采集

data/
  eye_on_hand/
    poses/                     # tcp/tag_pose/tag_corners
    images/                    # rgb/depth

results/
  eye_on_hand/
    handeye_transform.txt      # 标定矩阵
    depth_scale.txt            # 深度缩放
    calibration_info.txt       # 标定摘要
```

## 常见问题

- 检测不到 ArUco：检查 `dictionary`、`marker_id`、`marker_size`，并确认标记在画面内、对焦和曝光正常。
- 改了 ArUco 板尺寸：修改 `ARUCO_CONFIG['marker_size']`，单位是米。已保存的旧 `tag_pose` 不会自动按新尺寸重算。
- 旧数据混入计算：清理 `data/eye_on_hand/poses` 和 `data/eye_on_hand/images`，或者至少删除旧样本对应的 `tcp_XXX.txt` 和 `tag_pose_XXX.txt`。
- RealSense 连接失败：检查 USB、RealSense SDK、是否被其他程序占用。
- 机器人连接失败：检查机器人 IP、网络配置和端口 `30003`。
