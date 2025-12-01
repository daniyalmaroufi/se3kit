"""
rotation.py

Defines a Rotation class representing a 3x3 rotation matrix
with constructors from quaternions, Euler angles, and utility
methods for axis-angle, ZYX Euler angles, and ROS geometry types.
"""


import logging
from math import atan2, cos, pi, sin, sqrt

import numpy as np
import quaternion  # Requires numpy-quaternion package

from se3kit.ros_compat import get_ros_geometry_msgs
from se3kit.utils import deg2rad, is_identity, rad2deg, skew_to_vector

# Retrieve the ROS geometry message types (Point, Quaternion, Pose, Vector3)
Point, Quaternion, Pose, Vector3 = get_ros_geometry_msgs()
use_geomsg = Quaternion is not None

# module logger
logger = logging.getLogger(__name__)

TOLERANCE = 1e-14


class Rotation:
    """
    Represents a 3x3 rotation matrix used for rotating vectors in 3D space.
    """

    def __init__(self, init_value=None):
        """
        Initializes rotation from quaternion, matrix, ROS Quaternion, or another Rotation.

        :param init_value: Quaternion, 3x3 ndarray, ROS Quaternion, or another Rotation
        :type init_value: np.ndarray | np.quaternion | Rotation | Quaternion | None
        """
        if init_value is None:
            # Case 1: No input provided
            # Default to the identity rotation (3x3 identity matrix)
            self.m = np.eye(3)

        elif isinstance(init_value, quaternion.quaternion):
            # Case 2: Input is a numpy quaternion
            # Convert quaternion to a 3x3 rotation matrix
            self.m = quaternion.as_rotation_matrix(init_value)

        elif use_geomsg and isinstance(init_value, Quaternion):
            # Case 3: Input is a ROS geometry_msgs Quaternion (ROS1 or ROS2)
            # Convert ROS Quaternion to numpy quaternion first, then to rotation matrix
            q = np.quaternion(init_value.w, init_value.x, init_value.y, init_value.z)
            self.m = np.quaternion.as_rotation_matrix(q)

        elif isinstance(init_value, np.ndarray):
            # Case 4: Input is a numpy array
            # Expecting a 3x3 rotation matrix directly
            if not Rotation.is_valid(init_value):
                raise ValueError("Rotation matrix is invalid.")
            self.m = init_value

        elif isinstance(init_value, Rotation):
            # Case 5: Input is another Rotation object
            # Copy its internal rotation matrix
            self.m = np.copy(init_value.m)

        else:
            # Case 6: Input type is not supported
            raise TypeError(f"Cannot initialize Rotation from {type(init_value)}")

    def __mul__(self, other):
        """
        Multiplies two rotations and returns the product.

        :param other: Another Rotation object
        :type other: Rotation
        :return: Product rotation
        :rtype: Rotation
        """
        return Rotation(np.matmul(self.m, other.m))

    @staticmethod
    def eye():
        """
        Returns identity rotation matrix.

        This is a 3x3 identity rotation matrix, representing no rotation.

        :return: Identity rotation as a Rotation object.
        :rtype: Rotation
        """
        return Rotation()

    @staticmethod
    def from_zyx(euler, degrees=False):
        """
        Creates a Rotation from ZYX Euler angles.

        The input angles are applied in Z (yaw), Y (pitch), X (roll) order to generate
        the corresponding 3x3 rotation matrix.

        :param euler: Euler angles as [z, y, x]
        :type euler: array-like of 3 floats
        :param degrees: If True, input is in degrees. Defaults to False (radians)
        :type degrees: bool
        :return: Rotation object representing the specified rotation.
        :rtype: Rotation
        """

        # Convert input Euler angles to radians if they are in degrees, else just convert to NumPy array
        e = deg2rad(euler) if degrees else np.array(euler)

        # Unpack angles into separate components: a = Z (yaw), b = Y (pitch), g = X (roll)
        a, b, g = e

        # Precompute cosines and sines of each angle for matrix construction
        ca, sa = cos(a), sin(a)
        cb, sb = cos(b), sin(b)
        cg, sg = cos(g), sin(g)

        # Construct the 3x3 rotation matrix using ZYX (yaw-pitch-roll) convention
        # Rows correspond to new x, y, z axes after rotation
        return Rotation(
            np.array(
                [
                    [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
                    [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
                    [-sb, cb * sg, cb * cg],
                ]
            )
        )

    # # Create Rotation from Euler angles in degrees
    # from_zyx_degrees = lambda euler: Rotation.from_zyx(euler, degrees=True)

    # # Alias for from_zyx, using ABC notation (same as ZYX)
    # from_ABC = lambda abc, degrees=False: Rotation.from_zyx(abc, degrees=degrees)

    # # ABC notation with degrees
    # from_ABC_degrees = lambda abc: Rotation.from_ABC(abc, degrees=True)

    # # Create Rotation from roll-pitch-yaw sequence (XYZ order), flipping to ZYX internally
    # from_rpy = lambda rpy, degrees=False: Rotation.from_zyx(np.flip(rpy), degrees=degrees)

    @staticmethod
    def from_zyx_degrees(euler):
        """
        Creates a Rotation object from ZYX Euler angles in degrees.

        :param euler: Euler angles [Z, Y, X] in degrees
        :type euler: list, tuple, or np.ndarray
        :return: Rotation object representing the rotation
        :rtype: Rotation
        """
        return Rotation.from_zyx(euler, degrees=True)

    # legacy mixed-case name removed in favor of lowercase alias below

    @staticmethod
    def from_abc(abc, degrees=False):
        """
        Lowercase alias for creating a Rotation from ABC angles (ZYX order).
        """
        return Rotation.from_zyx(abc, degrees=degrees)

    @staticmethod
    def from_abc_degrees(abc):
        """
        Lowercase alias for creating a Rotation from ABC angles (degrees).
        """
        return Rotation.from_abc(abc, degrees=True)

    # Backwards-compatible aliases (legacy mixed-case names)
    from_ABC = from_abc  # noqa: N815
    from_ABC_degrees = from_abc_degrees  # noqa: N815

    @staticmethod
    def from_rpy(rpy, degrees=False):
        """
        Creates a Rotation object from roll-pitch-yaw angles (XYZ order),
        flipping them internally to ZYX.

        :param rpy: Roll-Pitch-Yaw angles [X, Y, Z]
        :type rpy: list, tuple, or np.ndarray
        :param degrees: If True, angles are in degrees; otherwise radians
        :type degrees: bool
        :return: Rotation object representing the rotation
        :rtype: Rotation
        """
        return Rotation.from_zyx(np.flip(rpy), degrees=degrees)

    def is_identity(self):
        """
        Checks whether the rotation matrix represents the identity rotation.

        This means the rotation matrix is effectively the 3x3 identity matrix,
        with no rotation applied.

        :return: True if rotation is identity, False otherwise.
        :rtype: bool
        """
        return is_identity(self.m)

    def as_zyx(self, degrees=False):
        """
        Converts the rotation matrix to ZYX Euler angles (yaw-pitch-roll).

        The ZYX convention represents rotations applied in order:
        1. Rotation about Z-axis (yaw)
        2. Rotation about Y-axis (pitch)
        3. Rotation about X-axis (roll)

        Handles the singularity cases when the rotation is identity or near 180 degrees.

        :param degrees: If True, returns angles in degrees; otherwise in radians.
        :type degrees: bool, optional (default=False)
        :return: Euler angles as a 3-element array [z, y, x].
        :rtype: np.ndarray
        """

        if self.is_identity():
            # If the rotation is the identity matrix (no rotation), return zero angles
            return np.zeros(3)

        # Compute ZYX Euler angles from the rotation matrix
        # a = yaw (rotation about Z axis)
        a = atan2(self.m[1, 0], self.m[0, 0])

        # b = pitch (rotation about Y axis)
        # sqrt(self.m[2,1]**2 + self.m[2,2]**2) computes the projection of the rotation onto the XZ-plane
        b = atan2(-self.m[2, 0], sqrt(self.m[2, 1] ** 2 + self.m[2, 2] ** 2))

        # g = roll (rotation about X axis)
        g = atan2(self.m[2, 1], self.m[2, 2])

        # Combine the three Euler angles into a single array [yaw, pitch, roll]
        angles = np.array([a, b, g])

        # Convert to degrees if requested, otherwise leave in radians

        return rad2deg(angles) if degrees else angles

    # # Returns ZYX Euler angles (alias for as_zyx), optionally in degrees
    # as_ABC = lambda self, degrees=False: self.as_zyx(degrees)

    # # Returns roll-pitch-yaw (RPY) Euler angles by flipping the order of ZYX angles
    # # Useful when interfacing with systems that expect RPY instead of ZYX
    # as_rpy = lambda self, degrees=False: np.flip(self.as_zyx(degrees))

    # # Converts the rotation matrix to a quaternion using numpy-quaternion
    # # Returns a np.quaternion object representing the same rotation
    # as_quat = lambda self: np.quaternion.from_rotation_matrix(self.m)

    def as_abc(self, degrees=False):
        """
        Returns the Euler angles in ABC order (ZYX), optionally in degrees.

        This is an alias for `as_zyx`.

        :param degrees: If True, angles are returned in degrees; otherwise in radians
        :type degrees: bool
        :return: Euler angles [A, B, C] (same as ZYX order)
        :rtype: np.ndarray
        """
        return self.as_zyx(degrees=degrees)

    # Legacy alias
    as_ABC = as_abc  # noqa: N815

    def as_rpy(self, degrees=False):
        """
        Returns the Euler angles in roll-pitch-yaw (RPY) order [X, Y, Z].

        Internally flips the ZYX angles to XYZ order.
        Useful when interfacing with systems that expect RPY instead of ZYX.

        :param degrees: If True, angles are returned in degrees; otherwise in radians
        :type degrees: bool
        :return: Euler angles [roll, pitch, yaw]
        :rtype: np.ndarray
        """
        return np.flip(self.as_zyx(degrees=degrees))

    def as_quat(self):
        """
        Convert the rotation matrix to a quaternion (np.quaternion) using
        a pure numpy implementation.

        Returns w, x, y, z in np.quaternion format.
        """
        r = self.m
        tr = np.trace(r)

        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2  # s = 4*w
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s

            # Find the largest diagonal element
        elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2  # s = 4*x
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2  # s = 4*y
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2  # s = 4*z
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s

        return np.quaternion(w, x, y, z)

    def as_geometry_orientation(self):
        """
        Converts rotation to ROS geometry_msgs Quaternion.

        Useful for publishing rotations in ROS topics or working with ROS messages.

        :return: geometry_msgs.msg.Quaternion with x, y, z, w components
        :rtype: Quaternion
        :raises ModuleNotFoundError: If geometry_msgs is not available.
        """

        # Check if the ROS geometry messages are available (ROS1 or ROS2)
        if not use_geomsg:
            # If unavailable, then raise an error
            raise ModuleNotFoundError("geometry_msgs module not available")

        # Convert the internal rotation matrix to a quaternion object
        q = self.as_quat()

        # Construct and return a ROS-compatible Quaternion message
        # The ROS Quaternion fields are ordered as x, y, z, w
        return Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)

    def as_axisangle(self):
        """
        Returns axis-angle representation of rotation.

        The axis-angle representation defines a rotation as an axis vector
        and an angle of rotation about that axis.

        Special cases:
        - Identity rotation: angle=0, axis arbitrary ([1,0,0] used)
        - 180-degree rotation: handled separately to avoid division by zero

        :return: Tuple containing:
                - axis vector as a 3-element np.ndarray
                - rotation angle in radians as a float
        :rtype: (np.ndarray, float)
        """

        tr = np.trace(self.m)
        if self.is_identity():  # Identity case
            # if the rotation is the identity
            return np.array([1, 0, 0]), 0
        elif abs(tr + 1) < TOLERANCE:  # 180 degree case
            # Loop through diagonal elements to find a valid axis component
            for i in range(3):
                if abs(self.m[i, i] + 1) > TOLERANCE:
                    w = np.zeros(3)
                    w[i] = self.m[i, i] + 1
                    w /= np.linalg.norm(w)
                    return w, pi
        else:  # General case
            # Compute axis-angle from rotation matrix
            theta = np.arccos((tr - 1) / 2)
            w = skew_to_vector((self.m - self.m.T) * 0.5 / np.sin(theta))
            return w, theta

    @property
    def x_axis(self):
        """
        Returns the x-axis of the rotation matrix.

        The x-axis corresponds to the first column of the 3x3 rotation matrix,
        representing the direction of the rotated frame's x-axis in world coordinates.

        :return: 3-element vector representing the x-axis
        :rtype: np.ndarray
        """
        return self.m[:, 0]

    @property
    def y_axis(self):
        """
        Returns the y-axis of the rotation matrix.

        The y-axis corresponds to the second column of the 3x3 rotation matrix,
        representing the direction of the rotated frame's y-axis in world coordinates.

        :return: 3-element vector representing the y-axis
        :rtype: np.ndarray
        """
        return self.m[:, 1]

    @property
    def z_axis(self):
        """
        Returns the z-axis of the rotation matrix.

        The z-axis corresponds to the third column of the 3x3 rotation matrix,
        representing the direction of the rotated frame's z-axis in world coordinates.

        :return: 3-element vector representing the z-axis
        :rtype: np.ndarray
        """
        return self.m[:, 2]

    @staticmethod
    def is_valid(mat, verbose=False, tol=1e-6):
        """
        Checks if the given matrix is a valid 3x3 rotation matrix.

        A valid rotation matrix is a 3x3 orthogonal matrix with a determinant of 1.
        This method verifies the following:
          - The input is a numpy ndarray of shape (3, 3)
          - The matrix is orthogonal (R.T @ R == I within tolerance)
          - The determinant of the matrix is 1 (within tolerance)

        :param mat: Matrix to check for validity as a rotation matrix.
        :type mat: np.ndarray
        :param verbose: If True, prints detailed error messages or success confirmation.
        :type verbose: bool, optional
        :param tol: Tolerance for orthogonality and determinant checks.
        :type tol: float, optional
        :return: True if the matrix is a valid rotation matrix, False otherwise.
        :rtype: bool
        """
        try:
            # Type mismatch indicates incorrect usage of the API
            if not isinstance(mat, np.ndarray):
                raise TypeError(f"Rotation matrix must be of type np.ndarray, got {type(mat)}")

            if mat.shape != (3, 3):
                raise ValueError(f"Rotation matrix must be 3x3, got {mat.shape}")

            if not np.allclose(mat.T @ mat, np.eye(3), atol=tol):
                raise ValueError("Matrix is not orthogonal (R.T @ R != I)")

            det_val = np.linalg.det(mat)
            if not np.isclose(det_val, 1.0, atol=tol):
                raise ValueError(f"Determinant must be 1, got {det_val}")

        except (ValueError, TypeError) as e:
            if verbose:
                logger.error("❌ %s", e)
            return False

        if verbose:
            logger.info("✔️  Matrix is a valid rotation matrix.")
        return True

    @staticmethod
    def from_quat(q):
        """
        Create a ``Rotation`` object from a quaternion.

        The input quaternion is expected in the format ``(x, y, z, w)``, where
        ``w`` is the real component. Internally this is converted into an
        ``np.quaternion`` object with ordering ``(w, x, y, z)`` before being passed
        to the ``Rotation`` constructor.

        :param q: Quaternion as a tuple or list in the order ``(x, y, z, w)``.
        :type q: tuple[float, float, float, float]
        :return: A ``Rotation`` instance representing the same rotation.
        :rtype: Rotation
        """
        q_np = quaternion.quaternion(q[3], q[0], q[1], q[2])
        return Rotation(q_np)

    @staticmethod
    def from_axisangle(axis, angle_rad):
        """
        Construct a ``Rotation`` object from an axis-angle representation using
        Rodrigues' rotation formula.

        The axis must be a 3D vector and does not need to be normalized; it will be
        normalized internally. The angle is given in radians. This method computes
        the corresponding 3x3 rotation matrix:

        .. math::
            R = I \\cos\\theta + (1 - \\cos\\theta) \\, a a^T + [a]_\\times \\sin\\theta

        where ``a`` is the unit rotation axis and ``[a]_x`` is the cross-product
        (skew-symmetric) matrix of ``a``.

        :param axis: 3D rotation axis. Does not need to be unit length.
        :type axis: array-like of float with shape (3,)
        :param angle_rad: Rotation angle in radians.
        :type angle_rad: float
        :return: A ``Rotation`` instance representing the given rotation.
        :rtype: Rotation
        """
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        x, y, z = axis
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        c_factor = 1 - c

        r_mat = np.array(
            [
                [c + x * x * c_factor, x * y * c_factor - z * s, x * z * c_factor + y * s],
                [y * x * c_factor + z * s, c + y * y * c_factor, y * z * c_factor - x * s],
                [z * x * c_factor - y * s, z * y * c_factor + x * s, c + z * z * c_factor],
            ]
        )

        return Rotation(r_mat)
