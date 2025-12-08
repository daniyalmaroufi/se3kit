SE3kit documentation
====================

Lightweight Python library for 3D rigid-body transformations and rotations.

Overview
--------

SE3kit implements core SE(3) building blocks and a minimal robot FK example:

*   Homogeneous transforms follow the standard block form :math:`T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}` where :math:`R \in SO(3)` and :math:`t \in \mathbb{R}^3`.
*   Rotations are stored as 3x3 matrices in ``se3kit.rotation.Rotation``.
*   Translations are stored as 3-vectors in ``se3kit.translation.Translation``.

Installation
------------

Install using pip:

.. code-block:: bash

    pip install se3kit

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

**Convert between radians and degrees effectively:**

.. code-block:: python

    from se3kit.degrees import Degrees
    # Create an angle in degrees
    theta = Degrees(90)

    print(theta.deg)  # 90.0
    print(theta.rad)  # 1.57079632679 (π/2)

    # Update the angle in radians
    theta.rad = 3.14159  # About π
    print(theta.deg)     # ≈ 180.0

**Store and manipulate 3D points in either Cartesian or Full Homogeneous Form:**

.. code-block:: python

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


**Transform points attached to a robot’s tool through the end-effector pose:**

.. code-block:: python

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

**Convert large sets of 3D points to homogeneous coordinates for batch processing:**

.. code-block:: python

    import numpy as np
    from se3kit.hpoint import HPoint

    point_cloud = np.random.rand(100, 3)  # N × 3 point cloud

    hpoints = [HPoint(p) for p in point_cloud]

**Compose multiple transformations to represent an entire robot arm’s kinematic chain.**

.. code-block:: python

    from se3kit.transformation import Transformation
    from se3kit.translation import Translation
    from se3kit.rotation import Rotation

    # Example 3-link arm
    T1 = Transformation(Translation([0, 0, 0.4]), Rotation.from_rpy([0, 0, 0.5]))
    T2 = Transformation(Translation([0, 0, 0.3]), Rotation.from_rpy([0, 0.2, 0]))
    T3 = Transformation(Translation([0.1, 0, 0]), Rotation.from_rpy([0.1, 0, 0]))

    T_end_effector = T1 * T2 * T3
    print(T_end_effector.as_geometry_pose())

**Seamlessly convert between millimeters and meters for transformations.**

.. code-block:: python

    from se3kit.transformation import Transformation

    T_mm = Transformation.convert_m_to_mm(T_end_effector)
    T_m = Transformation.convert_mm_to_m(T_mm)

    print(T_mm.translation.xyz)

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   se3kit
