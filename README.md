# se3kit

Lightweight Python library for 3D rigid-body transforms, rotations, and simple robot kinematics.

Key components:
- `transformation.Transformation` — 4×4 homogeneous transforms (`src/transformation.py`)
- `rotation.Rotation` — 3×3 rotation matrices and Euler/quaternion helpers (`src/rotation.py`)
- `translation.Translation` — 3D translation vectors and unit helpers (`src/translation.py`)
- `hpoint.HPoint` — homogeneous point (4×1) wrapper (`src/hpoint.py`)
- `robot.Robot` — simple serial manipulator models and forward kinematics (`src/robot.py`)
- Utility helpers in `src/utils.py` and angle convenience class `degrees.Degrees` (`src/degrees.py`)

Repository files:
- `src/__init__.py`
- `src/transformation.py`
- `src/rotation.py`
- `src/translation.py`
- `src/hpoint.py`
- `src/robot.py`
- `src/utils.py`
- `src/degrees.py`
- `src/ros_compat.py`
- `src/tests.py`
- `LICENSE`

Overview
--------
se3kit implements core SE(3) building blocks and a minimal robot FK example:
- Homogeneous transforms follow the standard block form
	$$
T=\begin{bmatrix}R & t\\[4pt]0 & 1\end{bmatrix}
$$
	where $R\in SO(3)$ and $t\in\mathbb{R}^3$.
- Rotations are stored as 3×3 matrices in `rotation.Rotation`.
- Translations are stored as 3-vectors in `translation.Translation`.
- `Robot` provides factory methods for common robot models and a space-frame forward kinematics method: `robot.Robot.FK_space`.

Installation
------------
This library is pure Python with a small NumPy/Scipy dependency for robot exponentials.

- Recommended (editable):
```sh
pip install -e .
pip install numpy scipy
```

If you use ROS message conversions, the package will detect ROS1 or ROS2 via `ros_compat.ROS_VERSION` (`src/ros_compat.py`). Geometry message types are wrapped in `ros_compat.get_ros_geometry_msgs`.

Quick usage examples
--------------------
Create transforms and compose them:

```python
from transformation import Transformation
from rotation import Rotation
from translation import Translation

t = Transformation(Translation([0, 0, 1]), Rotation())  # 1 m up, identity rotation
```

Transform a homogeneous point:

```python
from hpoint import HPoint
p = HPoint(0.1, 0.0, 0.0)
pt = t.transform_hpoint(p)  # uses Transformation.transform_hpoint in `src/transformation.py`
```

Robot forward kinematics (KUKA iiwa example):

```python
from robot import Robot
r = Robot.create_iiwa()
ja_deg = [-0.01, -35.10, 47.58, 24.17, 0.00, 0.00, 0.00]
import numpy as np
tf = r.FK_space(np.deg2rad(ja_deg))  # returns a Transformation (`src/robot.py`)
```

API documentation
-----------------
See the bundled API reference: `docs/API.md`.

Testing
-------
Run the unit tests with Python's unittest:

```sh
python -m unittest src.tests
```

If using ROS2, `src/tests.py` will initialize `rclpy` when run as `__main__`.

Contributing
------------
- Keep changes small and well tested.
- Follow NumPy-style docstrings as in existing files.
- Add unit tests in `src/tests.py`.

License
-------
MIT — see `LICENSE`

Contact
-------
Repository authored by Daniyal Maroufi.

---

## se3kit API Reference

This document summarizes the main public classes and utilities in the library and points to their implementations.

Core classes
------------

- `transformation.Transformation` (`src/transformation.py`)
	- Represents a 4×4 homogeneous transform.
	- Constructors:
		- `Transformation(np.ndarray(4x4))` — full matrix
		- `Transformation(Pose)` — from ROS Pose if available (see `ros_compat.get_ros_geometry_msgs`)
		- `Transformation(Translation)` — translation only (rotation = identity)
		- `Transformation(Translation, Rotation)` — explicit parts
	- Key methods and properties:
		- `matrix` — returns 4×4 matrix
		- `rotation`, `rotation = ...` — Rotation part (uses `rotation.Rotation`)
		- `translation`, `translation = ...` — Translation part (uses `translation.Translation`)
		- `inv` — inverse transform
		- `transform_hpoint(HPoint)` — transform a homogeneous point (`src/hpoint.py`)
		- `as_geometry_pose()` — returns ROS `Pose` if geometry messages available

- `rotation.Rotation` (`src/rotation.py`)
	- Stores a 3×3 rotation matrix.
	- Constructors accept:
		- None (identity)
		- `np.quaternion` (numpy-quaternion) or ROS `Quaternion` (via `ros_compat`)
		- 3×3 `np.ndarray`
		- another `Rotation`
	- Converters and helpers:
		- `from_zyx(euler, degrees=False)` — create from ZYX Euler angles
		- `from_rpy(rpy, degrees=False)` — RPY alias
		- `as_zyx(degrees=False)` / `as_rpy(degrees=False)` — extract Euler angles
		- `as_quat()` — returns `np.quaternion`
		- `as_geometry_orientation()` — ROS `Quaternion`
		- `as_axisangle()` — axis-angle decomposition
		- `x_axis, y_axis, z_axis` — column axes of rotation matrix

- `translation.Translation` (`src/translation.py`)
	- Simple 3-vector wrapper with convenience:
		- init from list/np.ndarray, `HPoint`, ROS `Point`/`Vector3`, or another `Translation`
		- arithmetic ops `+`, `-`, scalar `*`, `/`
		- `norm()` — Euclidean norm
		- `convert_m_to_mm()`, `convert_mm_to_m()` and their non-destructive variants
		- `as_geometry_point()` — ROS `Point` if available

- `hpoint.HPoint` (`src/hpoint.py`)
	- Homogeneous 4×1 point container.
	- Construct with `HPoint(x,y,z)` or `HPoint(np.array([x,y,z]))` or `HPoint(np.array([x,y,z,w]))`
	- `x,y,z` properties and `xyz` accessor for Cartesian vector

- `robot.Robot` (`src/robot.py`)
	- Factory methods:
		- `Robot.create_iiwa()` — KUKA iiwa R14 model (millimeters)
		- `Robot.create_franka_fp3()` — Franka Panda approximation (meters)
	- Kinematics:
		- `FK_space(joint_angles)` — computes space-frame forward kinematics by exponentiating screws. Uses `scipy.linalg.expm` and `utils.vector_to_skew`.

Utilities
---------
- `src/utils.py`
	- `deg2rad`, `rad2deg`
	- `is_near`, `is_identity`
	- `vector_to_skew(v)` and `skew_to_vector(sk)`

ROS compatibility
-----------------
- `src/ros_compat.py`
	- Detects ROS1/ROS2 availability and exposes:
		- `ROS_VERSION` — `0` (none), `1` (ROS1), `2` (ROS2)
		- `get_ros_geometry_msgs()` — returns `(Point, Quaternion, Pose, Vector3)` or `(None, None, None, None)`

Examples
--------
See the README for short examples or inspect:
- Forward kinematics usage: `src/robot.py`
- Transform/rotation examples: `src/transformation.py`, `src/rotation.py`

Files
-----
- `src/__init__.py`
- `src/transformation.py`
- `src/rotation.py`
- `src/translation.py`
- `src/hpoint.py`
- `src/robot.py`
- `src/utils.py`
- `src/degrees.py`
- `src/ros_compat.py`
- `src/tests.py`

