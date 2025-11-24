"""
rotation.py

Defines a Rotation class representing a 3x3 rotation matrix
with constructors from quaternions, Euler angles, and utility
methods for axis-angle, ZYX Euler angles, and ROS geometry types.
"""


import logging

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

    def is_identity(self):
        """
        Checks whether the rotation matrix represents the identity rotation.

        This means the rotation matrix is effectively the 3x3 identity matrix,
        with no rotation applied.

        :return: True if rotation is identity, False otherwise.
        :rtype: bool
        """
        return is_identity(self.m)

    @staticmethod
    def rotate_x(theta, degrees=False):
        """
        Produces a rotation matrix for rotation of 'theta' angle around x axis.

        :param theta: Rotation angle
        :type theta: float
        :return: Rotation matrix around x axis
        :rtype: Rotation
        """
        theta = deg2rad(theta) if degrees else theta
        return Rotation(
            np.array(
                [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
            )
        )

    @staticmethod
    def rotate_y(theta, degrees=False):
        """
        Produces a rotation matrix for rotation of 'theta' angle around y axis.

        :param theta: Rotation angle
        :type theta: float
        :return: Rotation matrix around y axis
        :rtype: Rotation
        """
        theta = deg2rad(theta) if degrees else theta
        return Rotation(
            np.array(
                [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
            )
        )

    @staticmethod
    def rotate_z(theta, degrees=False):
        """
        Produces a rotation matrix for rotation of 'theta' angle around z axis.

        :param theta: Rotation angle
        :type theta: float
        :return: Rotation matrix around z axis
        :rtype: Rotation
        """
        theta = deg2rad(theta) if degrees else theta
        return Rotation(
            np.array(
                [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
            )
        )

    @staticmethod
    def from_zyx(euler, extrinsic=False, degrees=False):
        """
        Creates a Rotation from ZYX (default intrinsic) Euler angles.
        More on intrinsic and extrinsic 3D rotations:
        https://dominicplein.medium.com/extrinsic-intrinsic-rotation-do-i-multiply-from-right-or-left-357c38c1abfd

        The input angles are applied in Z, Y, X order using intrinsic or extrinsic rotation to generate
        the corresponding 3x3 rotation matrix.

        :param euler: Euler angles as [z, y, x]
        :type euler: array-like of 3 floats
        :param extrinsic: If True, extrinsic rotation is applied (with respect to the fixed frame)
        :type extrinsic: bool
        :param degrees: If True, input is in degrees. Defaults to False (radians)
        :type degrees: bool
        :return: Rotation object representing the specified rotation.
        :rtype: Rotation
        """

        # Convert input Euler angles to radians if they are in degrees, else just convert to NumPy array
        angles = deg2rad(euler) if degrees else np.array(euler)

        # Unpack angles into separate components: alpha = Z , beta = Y , gamma = X
        alpha, beta, gamma = angles

        rx = Rotation.rotate_x(gamma)
        ry = Rotation.rotate_y(beta)
        rz = Rotation.rotate_z(alpha)

        # Construct the 3x3 rotation matrix
        # if intrinsic: 1) rotate for alpha around Z 2) rotate for beta around Y' 3) rotate for gamma around X"
        # if extrinsic: 1) rotate for alpha around Z 2) rotate for beta around Y 3) rotate for gamma around X
        return rz * ry * rx if not extrinsic else rx * ry * rz

    @staticmethod
    def from_abc(abc, degrees=False):
        """
        Lowercase alias for creating a Rotation from ABC angles (ZY'X" order).
        This method assumes intrinsic rotation, as KUKA notation.
        """
        return Rotation.from_zyx(abc, degrees=degrees)

    # Backwards-compatible aliases (legacy mixed-case names)
    from_ABC = from_abc  # noqa: N815

    @staticmethod
    def from_rpy(rpy, extrinsic=True, degrees=False):
        """
        Creates a Rotation object from roll(around X)-pitch(around Y)-yaw(around Z) angles.
        As in aviation roll-pitch-yaw are usually considered among fixed axes, this method assumes extrinsic rotation as default.

        :param rpy: Roll-Pitch-Yaw angles [X, Y, Z]
        :type rpy: list, tuple, or np.ndarray
        :param extrinsic: If True, extrinsic rotation is applied (with respect to the fixed frame)
        :type extrinsic: bool
        :param degrees: If True, angles are in degrees; otherwise radians
        :type degrees: bool
        :return: Rotation object representing the rotation
        :rtype: Rotation
        """
        rpy = np.asarray(rpy)
        return Rotation.from_zyx(np.flip(rpy), extrinsic=not (extrinsic), degrees=degrees)

    def as_zyx(self, extrinsic=False, degrees=False):
        """
        Converts the rotation matrix to ZY'X" Euler angles, using intrinsic rotation as default.

        Handles the singularity cases when the rotation is identity or near 180 degrees.

        :param degrees: If True, returns angles in degrees; otherwise in radians.
        :type degrees: bool, optional (default=False)
        :param extrinsic: If True, extrinsic rotation is assumed (with respect to the fixed frame)
        :type extrinsic: bool
        :return: Euler angles as a 3-element array [z, y', x"].
        :rtype: np.ndarray
        """

        if self.is_identity():
            # If the rotation is the identity matrix (no rotation), return zero angles
            return np.zeros(3)

        if extrinsic:
            return np.flip(self.as_xyz(extrinsic=False, degrees=degrees))

        # Compute ZY'X" Euler angles from the rotation matrix
        # alpha = rotation about Z axis
        alpha = np.arctan2(self.m[1, 0], self.m[0, 0])

        # beta = rotation about Y' axis
        beta = np.arctan2(-self.m[2, 0], np.sqrt(self.m[2, 1] ** 2 + self.m[2, 2] ** 2))

        # gamma = rotation about X" axis
        gamma = np.arctan2(self.m[2, 1], self.m[2, 2])

        # Combine the three Euler angles into a single array [yaw, pitch, roll]
        angles = (
            np.flip(np.array([alpha, beta, gamma])) if extrinsic else np.array([alpha, beta, gamma])
        )

        # Convert to degrees if requested, otherwise leave in radians
        return rad2deg(angles) if degrees else angles

    def as_xyz(self, extrinsic=False, degrees=False):
        """
        Converts the rotation matrix to XY'Z" Euler angles, using intrinsic rotation as default.

        Handles the singularity cases when the rotation is identity or near 180 degrees.

        :param degrees: If True, returns angles in degrees; otherwise in radians.
        :type degrees: bool, optional (default=False)
        :param extrinsic: If True, extrinsic rotation is assumed (with respect to the fixed frame)
        :type extrinsic: bool
        :return: Euler angles as a 3-element array [x, y', z"].
        :rtype: np.ndarray
        """
        if self.is_identity():
            # If the rotation is the identity matrix (no rotation), return zero angles
            return np.zeros(3)

        if extrinsic:
            return np.flip(self.as_zyx(extrinsic=False, degrees=degrees))

        # Compute XY'Z" Euler angles from the rotation matrix
        # alpha = rotation about Z" axis
        alpha = np.arctan2(-self.m[0, 1], self.m[0, 0])

        # beta = rotation about Y' axis
        beta = np.arctan2(self.m[0, 2], np.sqrt(self.m[1, 2] ** 2 + self.m[2, 2] ** 2))

        # gamma = rotation about X axis
        gamma = np.arctan2(self.m[1, 2], self.m[2, 2])

        # Combine the three Euler angles into a single array [X, Y', Z"]
        angles = np.array([gamma, beta, alpha])

        # Convert to degrees if requested, otherwise leave in radians
        return rad2deg(angles) if degrees else angles

    def as_abc(self, degrees=False):
        """
        Returns the Euler angles in ABC order (ZY'X"), optionally in degrees.
        Assumed intrinsic rotation similar to KUKA notation.

        This is an alias for `as_zyx`.

        :param degrees: If True, angles are returned in degrees; otherwise in radians
        :type degrees: bool
        :return: Euler angles [A, B, C] (same as ZY'X" order)
        :rtype: np.ndarray
        """
        return self.as_zyx(degrees=degrees)

    # Legacy alias
    as_ABC = as_abc  # noqa: N815

    def as_rpy(self, extrinsic=True, degrees=False):
        """
        Returns the Euler angles in roll-pitch-yaw (RPY) order [X, Y, Z] assuming extrinsic rotation as default.

        :param extrinsic: If True, extrinsic rotation is assumed (with respect to the fixed frame)
        :type extrinsic: bool
        :param degrees: If True, angles are returned in degrees; otherwise in radians
        :type degrees: bool
        :return: Euler angles [roll, pitch, yaw]
        :rtype: np.ndarray
        """
        return self.as_zyx(extrinsic=extrinsic, degrees=degrees)

    def as_quat(self):
        """
        Converts the rotation matrix to a quaternion.

        Uses `numpy-quaternion` to generate a quaternion representing the same rotation.

        :return: Quaternion representing the rotation
        :rtype: quaternion.quaternion
        """
        return quaternion.from_rotation_matrix(self.m)

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

    def as_axisangle(self, degrees=False):
        """
        Returns axis-angle representation of rotation.

        The axis-angle representation defines a rotation as an axis vector
        and an angle of rotation about that axis.

        Special cases:
        - Identity rotation: angle=0, axis arbitrary ([1,0,0] used)
        - 180-degree rotation: handled separately to avoid division by zero

        :param degrees: If True, angle is returned in degrees; otherwise in radians
        :type degrees: bool
        :return: Tuple containing:
                - axis vector as a 3-element np.ndarray
                - rotation angle in radians as a float
        :rtype: (np.ndarray, float)
        """
        tr = np.trace(self.m)

        if self.is_identity():
            # if the rotation is the identity
            return np.array([1, 0, 0]), 0

        elif abs(tr + 1) < TOLERANCE:  # tr == -1, 180 degree case
            # Loop through diagonal elements to find a valid axis component
            i = np.argmax(np.array([self.m[i, i] for i in range(3)]))
            if abs(self.m[i, i] + 1) > TOLERANCE:
                w = np.array([self.m[j, i] for j in range(3)])
                w[i] += 1
                w /= np.sqrt(2 * (1 + self.m[i, i]))
                return w, (np.pi if not degrees else 180.0)
        else:  # General case
            # Compute axis-angle from rotation matrix
            theta = np.arccos((tr - 1) / 2)
            w = skew_to_vector((self.m - self.m.T) / (2 * np.sin(theta)))
            return w, (theta if not degrees else rad2deg(theta))

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
    def T(self):  # noqa: N802
        """
        Returns the transpose of the rotation matrix.

        :return: Transposed rotation matrix
        :rtype: se3kit.rotation.Rotation
        """
        return Rotation(self.m.T)

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
                logger.error("Not a valid rotation: %s", e)
            return False

        if verbose:
            logger.info("Matrix is a valid rotation matrix.")
        return True
