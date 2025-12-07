SE3kit documentation
====================

Lightweight Python library for 3D rigid-body transforms, rotations, and simple robot kinematics.

Overview
--------

SE3kit implements core SE(3) building blocks and a minimal robot FK example:

*   Homogeneous transforms follow the standard block form :math:`T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}` where :math:`R \in SO(3)` and :math:`t \in \mathbb{R}^3`.
*   Rotations are stored as 3x3 matrices in ``se3kit.rotation.Rotation``.
*   Translations are stored as 3-vectors in ``se3kit.translation.Translation``.

Installation
------------

This library is pure Python with a small NumPy/Scipy dependency for robot exponentials.

**Recommended (editable):**

.. code-block:: bash

    pip install .

If you use ROS message conversions, the package will detect ROS1 or ROS2 via ``se3kit.ros_compat``. Geometry message types are wrapped in ``get_ros_geometry_msgs``.

Quick usage examples
--------------------

**Create transforms and compose them:**

.. code-block:: python

    from se3kit.transformation import Transformation
    from se3kit.rotation import Rotation
    from se3kit.translation import Translation

    t = Transformation(Translation([0, 0, 1]), Rotation())  # 1 m up, identity rotation

**Transform a homogeneous point:**

.. code-block:: python

    from se3kit.hpoint import HPoint
    p = HPoint(0.1, 0.0, 0.0)
    pt = t.transform_hpoint(p)  # uses Transformation.transform_hpoint

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   se3kit
