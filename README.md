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

## 📚 Documentation

Full API documentation is available at:
👉 **[https://daniyalmaroufi.github.io/se3kit/](https://daniyalmaroufi.github.io/se3kit/)**

## 🛠 Usage

### Rigid Body Transformations

Create and compose transformations intuitively:

```python
import se3kit as se3

# Create a transformation: 1 meter up in Z, identity rotation
t1 = se3.Transformation(se3.Translation([0, 0, 1]), se3.Rotation())

# Compose transformations
t2 = se3.Transformation(
    se3.Translation([0.5, 0, 0]),
    se3.Rotation.from_rpy([0, 0, 1.57])  # Rotate 90° around Z
)

t_combined = t1 * t2
```

### 3D Point Transformation

Transform homogeneous points efficiently:

```python
import se3kit as se3

p = se3.HPoint(0.1, 0.5, 0.0)
p_transformed = t_combined.transform_hpoint(p)

print(p_transformed.xyz)   # Access as standard 3D vector
```

### Homogeneous Point (HPoint) Representation

Store and manipulate 3D points in either Cartesian or Full Homogeneous Form

```python
import se3kit as se3
import numpy as np

# Cartesian coordinates
p1 = se3.HPoint(0.2, 0.4, 0.1)

# From NumPy array
p2 = se3.HPoint(np.array([1.0, 2.0, 3.0]))

# From full homogeneous vector
p3 = se3.HPoint(np.array([0.5, 0.0, 1.0, 1.0]))

print(p1.xyz)          # [0.2, 0.4, 0.1]
print(p2.as_array())   # Full 4×1 homogeneous vector
```


### Transform a Homogeneous Point (HPoint)

Transform points attached to a robot’s tool through the end-effector pose.

```python
import se3kit as se3

# A tool point on the robot’s end effector
tool_point = se3.HPoint(0.1, 0.0, 0.0)

# End-effector pose in world frame
T_world_ee = se3.Transformation(
    se3.Translation([0.5, 0.2, 1.0]),
    se3.Rotation.from_rpy([0, 0, 1.57])
)

p_world = T_world_ee.transform_hpoint(tool_point)
print(p_world.xyz)

```



### Kinematic Chain Representation

Compose multiple transformations to represent an entire robot arm’s kinematic chain.

```python
import se3kit as se3

# Example arm links
T1 = se3.Transformation(se3.Translation([0, 0, 0.4]), se3.Rotation.from_rpy([0, 0, 0.5]))
T2 = se3.Transformation(se3.Translation([0, 0, 0.3]), se3.Rotation.from_rpy([0, 0.2, 0]))
T3 = se3.Transformation(se3.Translation([0.1, 0, 0]), se3.Rotation.from_rpy([0.1, 0, 0]))

T_end_effector = T1 * T2 * T3

print(T_end_effector.as_geometry_pose())

```


### Scaling and Unit Conversions

Seamlessly convert between millimeters and meters for transformations.

```python
import se3kit as se3

T_mm = se3.Transformation.convert_m_to_mm(T_end_effector)
T_m  = se3.Transformation.convert_mm_to_m(T_mm)

print(T_mm.translation.xyz)
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
