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

    import se3kit as se3

    # Create a transformation: 1 meter up in Z, identity rotation
    t1 = se3.Transformation(se3.Translation([0, 0, 1]), se3.Rotation())

    # Compose transformations
    t2 = se3.Transformation(
        se3.Translation([0.5, 0, 0]),
        se3.Rotation.from_rpy([0, 0, 1.57])  # Rotate 90° around Z
    )

    t_combined = t1 * t2

**Transform a homogeneous point:**

.. code-block:: python

    import se3kit as se3

    p = se3.HPoint(0.1, 0.5, 0.0)
    p_transformed = t_combined.transform_hpoint(p)

    print(p_transformed.xyz)   # Access as standard 3D vector


**Store and manipulate 3D points in either Cartesian or Full Homogeneous Form:**

.. code-block:: python

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


**Transform points attached to a robot’s tool through the end-effector pose:**

.. code-block:: python

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


**Compose multiple transformations to represent an full kinematic chain.**

.. code-block:: python

    import se3kit as se3

    # Example arm links
    T1 = se3.Transformation(se3.Translation([0, 0, 0.4]), se3.Rotation.from_rpy([0, 0, 0.5]))
    T2 = se3.Transformation(se3.Translation([0, 0, 0.3]), se3.Rotation.from_rpy([0, 0.2, 0]))
    T3 = se3.Transformation(se3.Translation([0.1, 0, 0]), se3.Rotation.from_rpy([0.1, 0, 0]))

    T_end_effector = T1 * T2 * T3

    print(T_end_effector.as_geometry_pose())

**Seamlessly convert between millimeters and meters for transformations.**

.. code-block:: python

    import se3kit as se3

    T_mm = se3.Transformation.convert_m_to_mm(T_end_effector)
    T_m  = se3.Transformation.convert_mm_to_m(T_mm)

    print(T_mm.translation.xyz)

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   se3kit
