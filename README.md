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

## 🛠 Usage

### Rigid Body Transformations

Create and compose transformations intuitively:

```python
from se3kit.transformation import Transformation
from se3kit.rotation import Rotation
from se3kit.translation import Translation

# Create a transformation: 1 meter up in Z, with identity rotation
t1 = Transformation(Translation([0, 0, 1]), Rotation())

# Compose transformations
t2 = Transformation(Translation([0.5, 0, 0]), Rotation.from_rpy([0, 0, 1.57])) # Rotate 90 deg around Z
t_combined = t1 * t2
```

### Point Transformation

Transform homogeneous points efficiently:

```python
from se3kit.hpoint import HPoint

p = HPoint(0.1, 0.5, 0.0)
p_transformed = t_combined.transform_hpoint(p)

print(p_transformed.xyz) # Access as standard 3D vector
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
