# SE3kit

[![Documentation Status](https://github.com/daniyalmaroufi/se3kit/actions/workflows/deploy_docs.yml/badge.svg)](https://daniyalmaroufi.github.io/se3kit/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**SE3kit** is a lightweight Python library designed for 3D rigid-body transformations and rotations. It provides intuitive wrappers for homogeneous transformations, rotation representations (matrices, Euler angles, quaternions), and geometric primitives.

> [!NOTE]
> This library includes optional ROS compatibility, automatically detecting and converting to/from ROS 1 or ROS 2 geometry messages if available.

---

## 🚀 Features

- **Rigid Body Transformations**: Easy-to-use 4x4 homogenous transformation matrices (`Transformation`).
- **Rotations**: Comprehensive 3x3 rotation matrix support with conversions to/from Euler angles, Quaternions, and Axis-Angles (`Rotation`).
- **Translations**: Vector arithmetic and unit management (`Translation`).
- **Homogeneous Points**: 4D point representation for SE(3) operations (`HPoint`).
- **ROS Integration**: Seamless conversion between SE3kit objects and ROS `geometry_msgs`.

## 📦 Installation

This library is pure Python with minimal dependencies (`numpy`, `numpy-quaternion`).

### PyPI (Recommended)

To install the latest stable version from PyPI:

```bash
pip install se3kit
```

### From Source (Development)

To install the library in editable mode from source:

```bash
git clone https://github.com/daniyalmaroufi/se3kit.git
cd se3kit
pip install -e .
```

### Development Setup

For contributors, install the development dependencies (testing, linting, docs):

```bash
pip install -e '.[dev]'
pre-commit install
```

## 🛠 SE3kit Applications

### Rigid Body Transformations

Create and compose transformations intuitively:

```python
from se3kit.transformation import Transformation
from se3kit.rotation import Rotation
from se3kit.translation import Translation

# Create a transformation: 1 meter up in Z, with identity rotation
t1 = Transformation(Translation([0, 0, 1]), Rotation())

# Compose transformations
t2 = Transformation(Translation([0.5, 0, 0]), Rotation.from_rpy([0, 0, 1.57])) # Rotate 90° around Z
t_combined = t1 * t2
```

### 3D Point Transformation

Transform homogeneous points efficiently:

```python
from se3kit.hpoint import HPoint

p = HPoint(0.1, 0.5, 0.0)
p_transformed = t_combined.transform_hpoint(p)

print(p_transformed.xyz) # Access as standard 3D vector
```

### Angle Conversion

Convert between degrees and radians effectively:

```python
from se3kit.degrees import Degrees

# Create an angle in degrees
theta = Degrees(90)

print(theta.deg)  # 90.0
print(theta.rad)  # 1.57079632679 (π/2)

# Update the angle in radians
theta.rad = 3.14159  # About π
print(theta.deg)     # ≈ 180.0
```

### Homogeneous Point (HPoint) Representation

Store and manipulate 3D points in either Cartesian or Full Homogeneous Form

```python
from se3kit.hpoint import HPoint

# Create from Cartesian coordinates
p1 = HPoint(0.2, 0.4, 0.1)

# Create from a NumPy array
import numpy as np
p2 = HPoint(np.array([1.0, 2.0, 3.0]))

# Create from a homogeneous vector
p3 = HPoint(np.array([0.5, 0.0, 1.0, 1.0]))

print(p1.xyz)     # [0.2 0.4 0.1]
print(p2.as_array())  # Full 4×1 homogeneous vector
```


### Robot's End-Effector Point Transformation

Transform points attached to a robot’s tool through the end-effector pose.

```python
from se3kit.transformation import Transformation
from se3kit.rotation import Rotation
from se3kit.translation import Translation
from se3kit.hpoint import HPoint

# A tool on the robot’s end effector
tool_point = HPoint(0.1, 0.0, 0.0)

# Robot end-effector pose in the world frame
T_world_ee = Transformation(
    Translation([0.5, 0.2, 1.0]),
    Rotation.from_rpy([0, 0, 1.57])
)

p_world = T_world_ee.transform_hpoint(tool_point)
print(p_world.xyz)
```


### 3D Point Cloud Data to Homogeneous Coordinate Conversion

Convert large sets of 3D points to homogeneous coordinates for batch processing.

```python
import numpy as np
from se3kit.hpoint import HPoint

point_cloud = np.random.rand(100, 3)  # N × 3 point cloud

hpoints = [HPoint(p) for p in point_cloud]
```


### Full Kinematic Chain Representation for Robot Arms

Compose multiple transformations to represent an entire robot arm’s kinematic chain.

```python
from se3kit.transformation import Transformation
from se3kit.translation import Translation
from se3kit.rotation import Rotation

# Example 3-link arm
T1 = Transformation(Translation([0, 0, 0.4]), Rotation.from_rpy([0, 0, 0.5]))
T2 = Transformation(Translation([0, 0, 0.3]), Rotation.from_rpy([0, 0.2, 0]))
T3 = Transformation(Translation([0.1, 0, 0]), Rotation.from_rpy([0.1, 0, 0]))

T_end_effector = T1 * T2 * T3
print(T_end_effector.as_geometry_pose())
```


### Scaling and Unit Conversions

Seamlessly convert between millimeters and meters for transformations.

```python
from se3kit.transformation import Transformation

T_mm = Transformation.convert_m_to_mm(T_end_effector)
T_m = Transformation.convert_mm_to_m(T_mm)

print(T_mm.translation.xyz)
```









## 📚 Documentation

Full API documentation is available at:
👉 **[https://daniyalmaroufi.github.io/se3kit/](https://daniyalmaroufi.github.io/se3kit/)**

### Building Docs Locally

You can build the Sphinx documentation locally to preview changes:

```bash
cd docs
make html
open _build/html/index.html
```

## 🤝 Contributing

We welcome contributions! Please follow these steps to ensure a smooth workflow:

1.  **Install Hooks**: Run `pre-commit install` to set up linting hooks.
2.  **Test Locally**: Run `python -m unittest discover -v` to ensure all tests pass.
3.  **Lint**: Code is automatically formatted with `black` and `ruff` on commit.

## 📄 License

Distributed under the **Apache 2.0 License**. See `LICENSE` for more information.

## ✍️ Authors

- **Daniyal Maroufi**
- **Omid Rezayof**
- **Sai Peram**
