import numpy as np
from scipy import linalg
from utils import vector_to_skew
from transformation import Transformation  # Ensure these exist in ros_compat.py


class Robot:
    """
    Represents a serial-link manipulator robot with screw axes and forward kinematics computation.

    This class stores essential kinematic information such as:
    - Screw axes (in the space frame)
    - Home configuration (pose of the end-effector at zero joint angles)
    - Degrees of freedom (DOF)
    - Link offsets and joint axes

    It provides static constructors for common robot models (e.g., KUKA iiwa R14, Franka Panda)
    and methods to compute forward kinematics.
    """

    def __str__(self):
        """
        Returns the robot's name as a string.

        :return: Name of the robot.
        :rtype: str
        """
        return self.name

    # -------------------------- Factory Methods --------------------------

    @staticmethod
    def create_iiwa():
        """
        Creates a KUKA iiwa R14 robot model with home configuration and screw axes.

        The model is defined in millimeters, consistent with many ROS2 datasets.

        :return: Instance of Robot configured as a KUKA iiwa R14.
        :rtype: Robot
        """
        r = Robot()
        r.name = 'Kuka iiwa R14'

        # Home pose (tool pointing downward, base at origin)
        r.home = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1306],
            [0, 0, 0, 1]
        ])

        # Joint axes (all revolute)
        r.axes = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, -1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # Link offsets (joint origins in base frame)
        r.offset = np.array([
            [0, 0, 170],
            [0, 0, 360],
            [0, 0, 600],
            [0, 0, 780],
            [0, 0, 1000],
            [0, -60, 1180],
            [0, 0, 1271]
        ])

        r.dof = r.axes.shape[0]

        # Compute screw axes (6 × DOF matrix)
        r.screw = np.empty([6, r.dof])
        for i in range(r.dof):
            r.screw[0:3, i] = r.axes[i]
            r.screw[3:6, i] = -np.cross(r.axes[i], r.offset[i])

        return r

    @staticmethod
    def create_franka_fp3():
        """
        Creates a Franka Emika Panda (FP3) robot model with home configuration and screw axes.

        The home matrix, axes, and link offsets are approximated from the standard Panda URDF.
        Screw axes are expressed in the space (base) frame.

        :return: Instance of Robot configured as a Franka Emika Panda (FP3).
        :rtype: Robot
        """
        r = Robot()
        r.name = 'Franka Emika Panda (FP3)'

        # Home configuration (tool forward, z-up, standard home pose)
        r.home = np.array([
            [1, 0, 0, 0.0],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.590],
            [0, 0, 0, 1]
        ])

        # Joint axes (all revolute, in space frame)
        r.axes = np.array([
            [0, 0, 1],  # J1 (base yaw)
            [0, 1, 0],  # J2 (shoulder pitch)
            [0, 0, 1],  # J3 (shoulder yaw)
            [0, 1, 0],  # J4 (elbow pitch)
            [0, 0, 1],  # J5 (wrist yaw)
            [0, 1, 0],  # J6 (wrist pitch)
            [0, 0, 1]   # J7 (flange rotate)
        ])

        # Link offsets (approx. from URDF distances in meters)
        r.offset = np.array([
            [0, 0, 0.333],
            [0, 0, 0.333],
            [0, 0, 0.649],
            [0, 0, 0.649],
            [0, 0, 0.649],
            [0, 0, 0.649],
            [0, 0, 0.649]
        ])

        r.dof = r.axes.shape[0]

        # Compute screw axes (6 × DOF matrix)
        r.screw = np.empty([6, r.dof])
        for i in range(r.dof):
            r.screw[0:3, i] = r.axes[i]
            r.screw[3:6, i] = -np.cross(r.axes[i], r.offset[i])

        return r

    # -------------------------- Kinematics --------------------------

    def FK_space(self, joint_angles):
        """
        Computes the forward kinematics of the robot in the space (base) frame.

        :param joint_angles: Joint angles (array-like of length equal to DOF)
        :type joint_angles: list | np.ndarray
        :return: End-effector pose as a Transformation object.
        :rtype: Transformation
        """
        if len(joint_angles) != self.dof:
            raise ValueError(f"Expected {self.dof} joint angles, got {len(joint_angles)}")

        cum_t = np.eye(4)
        for i in range(self.dof):
            # Matrix exponential of screw axis * joint angle
            cum_t = cum_t @ linalg.expm(vector_to_skew(self.screw[:, i]) * joint_angles[i])

        return Transformation(cum_t @ self.home)
