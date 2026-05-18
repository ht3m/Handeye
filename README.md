# Handeye Calibration

UR robot and Intel RealSense D405 hand-eye calibration toolkit. The current workflow targets an Eye-on-Hand setup with an ArUco marker, and keeps Eye-to-Hand utilities and offline evaluation scripts available for experiments.

## Features

- Collect synchronized robot TCP poses and RealSense RGB-D frames.
- Detect an ArUco calibration target with OpenCV.
- Solve hand-eye calibration with an AX=XB based solver and optional nonlinear refinement.
- Save calibration outputs under `results/<mode>/`.
- Evaluate a saved calibration with independent validation samples.
- Visualize TCP and camera optical frames before running detailed error measurements.

## Hardware

- Robot: UR series robot reachable through TCP port `30003`.
- Camera: Intel RealSense D405.
- Target: ArUco marker configured in [handeye/config.py](handeye/config.py).

## Project Layout

```text
handeye/
  calibration/                 # solvers, optimizer, transforms, feature extraction
  camera/                      # RealSense camera wrapper
  robot/                       # UR TCP pose reader
  calibration_solver.py        # calibration pipeline
  data_collector.py            # capture and persistence workflow
  device_manager.py            # robot/camera connection manager
  error_calculator.py          # validation metrics
  result_visualizer.py         # trajectory and error plots
  config.py                    # project configuration

scripts/
  main.py                      # online data collection and calibration
  evaluate_calibration.py      # collect validation samples and evaluate result
  offline_calibrate_and_evaluate.py
  visualize_handeye_frames.py  # local TCP/camera frame sanity check

tests/
  capture_svd_data.py
  move_tcp_eye_to_hand_apriltag.py

data/                          # captured samples, ignored by git
results/                       # calibration results
```

## Installation

Use Python 3.10 or newer. A virtual environment is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `pyrealsense2` installation fails, install the Intel RealSense SDK for your platform first, then reinstall the Python requirements.

## Configuration

Edit [handeye/config.py](handeye/config.py) before collecting data.

Important fields:

- `UR5_CONFIG['tcp_host_ip']`: robot controller IP.
- `UR5_CONFIG['tcp_port']`: robot TCP port, usually `30003`.
- `REALSENSE_CONFIG`: camera stream size and FPS.
- `ARUCO_CONFIG['dictionary']`: OpenCV ArUco dictionary name.
- `ARUCO_CONFIG['marker_id']`: target marker ID.
- `ARUCO_CONFIG['marker_size']`: marker side length in meters.
- `CALIBRATION_MODE`: `eye_on_hand` or `eye_to_hand`.
- `CALIBRATION_BACKEND`: currently `aruco` for the main workflow.

## Usage

Run commands from the repository root.

### 1. Collect Data and Calibrate

```powershell
python scripts\main.py
```

Interactive keys during capture:

- `Space`: detect the target in the current frame.
- `Enter`: save the current accepted detection.
- `Backspace`: cancel the pending detection.
- `Esc`: finish collection.

At least `CALIBRATION_CONFIG['min_calibration_points']` valid samples are required.

### 2. Inspect the TCP-Camera Relationship

Use this before detailed error measurement to check whether the transform matches the physical mount.

```powershell
python scripts\visualize_handeye_frames.py --mode eye_on_hand
```

Useful options:

```powershell
python scripts\visualize_handeye_frames.py --mode eye_on_hand --inverse
python scripts\visualize_handeye_frames.py --transform results\eye_on_hand\handeye_transform.txt
python scripts\visualize_handeye_frames.py --mode eye_on_hand --save results\eye_on_hand\frames.png --no-show
```

In the plot:

- TCP axes show the robot tool coordinate system.
- Camera axes show the OpenCV optical frame.
- Camera Z is the optical forward direction.
- The dashed line shows the offset from TCP origin to camera optical center.

### 3. Evaluate a Saved Calibration

```powershell
python scripts\evaluate_calibration.py
```

This loads `results/<mode>/handeye_transform.txt`, collects independent validation samples, and prints position and rotation consistency errors.

### 4. Offline Calibration and Evaluation

```powershell
python scripts\offline_calibrate_and_evaluate.py
```

This reads already saved samples from `data/<mode>/` and writes results to `results/<mode>/`.

### 5. SVD Experiment Utilities

Capture RGB-D samples:

```powershell
python tests\capture_svd_data.py
```

Analyze saved SVD samples:

```powershell
python -m handeye.calibration.svd
```

## Outputs

Calibration results are written to:

```text
results/<mode>/
  handeye_transform.txt   # 4x4 homogeneous transform
  depth_scale.txt         # optimized depth scale
  calibration_info.txt    # summary
```

For Eye-on-Hand, `handeye_transform.txt` is interpreted by the visualization script as `T_tcp_camera`, the camera pose in the TCP frame.

## Data Format

Captured samples are stored under:

```text
data/<mode>/
  poses/
    tcp_001.txt
    tag_pose_001.txt
    tag_corners_001.txt
  images/
    rgb_001.png
    depth_001.npy
```

`data/` is ignored by git because it may contain large capture data.

## Development

Run a syntax check without writing `__pycache__`:

```powershell
python -c "import ast, pathlib; files=list(pathlib.Path('scripts').glob('*.py'))+list(pathlib.Path('handeye').rglob('*.py'))+list(pathlib.Path('tests').glob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print(f'AST syntax check passed: {len(files)} files')"
```

Run type checking:

```powershell
mypy handeye scripts tests
```

## Troubleshooting

- `ModuleNotFoundError: cv2.aruco`: install `opencv-contrib-python`, not only `opencv-python`.
- RealSense connection fails: check USB connection, RealSense SDK installation, and whether another process is using the camera.
- Robot connection fails: check `UR5_CONFIG['tcp_host_ip']`, network route, and controller port `30003`.
- Plot shows the camera behind or flipped relative to the mount: verify whether the transform direction is `T_tcp_camera`; try `--inverse` only as a diagnostic.
- Changed code but behavior looks old: Python normally refreshes `.pyc` automatically. Delete `__pycache__` only when diagnosing unusual import behavior.
