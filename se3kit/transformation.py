import logging

import numpy as np

from se3kit.hpoint import HPoint
from se3kit.ros_compat import Pose, use_geomsg
from se3kit.rotation import Rotation
from se3kit.translation import Translation
from se3kit.utils import is_near

# Constants to avoid magic-numbers in argument checks
_TRANSLATION_ROTATION_ARG_COUNT = 2

logger = logging.getLogger(__name__)


class Transformation:
    """Represents a 4x4 homogeneous transformation matrix with rotation and translation."""

    def __init__(self, *args):
        """
        Initializes the Transformation from various sources.

        Can initialize from:
        - A 4x4 numpy matrix
        - A ROS geometry_msgs.msg.Pose message (ROS1 or ROS2)
        - A se3kit.translation.Translation object
        - se3kit.translation.Translation + se3kit.rotation.Rotation

        :param args: variable length arguments
        :raises AssertionError: if matrix is not 4x4
        :raises TypeError: if argument types are invalid
        """
        self._matrix = np.eye(4)

        if len(args) == 1:
            init = args[0]
            if isinstance(init, np.ndarray):
                # Direct 4x4 numpy array treated as a full transformation matrix
                if not Transformation.is_valid(init):
                    raise ValueError("Transformation matrix is invalid.")
                self._matrix = init

            elif use_geomsg and isinstance(init, Pose):
                # Single argument is a ROS Pose message and rotation and translation are extracted
                self.rotation = Rotation(init.orientation)
                self.translation = Translation(init.position)

            elif isinstance(init, Translation):
                # Single argument is a Translation object
                # Only the translation is set; rotation defaults to identity
                self.translation = init
        elif (
            len(args) == _TRANSLATION_ROTATION_ARG_COUNT
            and isinstance(args[0], Translation)
            and isinstance(args[1], Rotation)
        ):
            # Two arguments: first is Translation, second is Rotation
            # Directly set translation and rotation components
            self.translation = args[0]
            self.rotation = args[1]
        elif len(args) > 0:
            # Any other combination of arguments is invalid
            # Raise a TypeError to indicate incorrect usage
            raise TypeError(f"Invalid arguments for Transformation: {args}")

    def __mul__(self, other):
        """
        Multiplies this transformation with another Transformation (matrix composition).

        :param other: Transformation to multiply with
        :type other: se3kit.transformation.Transformation
        :return: Resulting Transformation
        :rtype: se3kit.transformation.Transformation
        :raises TypeError: if other is not a Transformation
        """
        if isinstance(other, Transformation):
            return Transformation(self._matrix @ other._matrix)
        raise TypeError(f"Invalid multiplication type {type(other)}")

    @property
    def rotation(self):
        """
        Returns the rotation component of the transformation as a Rotation object.

        The rotation is extracted from the upper-left 3x3 submatrix of the 4x4
        homogeneous transformation matrix.

        :return: se3kit.rotation.Rotation object representing the rotation part
        :rtype: se3kit.rotation.Rotation
        """
        return Rotation(self._matrix[0:3, 0:3])

    @rotation.setter
    def rotation(self, val):
        """
        Sets rotation component from a Rotation object or 3x3 ndarray.

        :param val: Rotation object or 3x3 rotation matrix
        :type val: se3kit.rotation.Rotation | np.ndarray
        """
        # Let Rotation class handle all type checking and conversion
        self._matrix[0:3, 0:3] = Rotation(val).m

    @property
    def translation(self):
        """
        Returns the translation component of the transformation as a Translation object.

        The translation is extracted from the top-right 3x1 part of the 4x4
        homogeneous transformation matrix.

        :return: se3kit.translation.Translation object representing the translation part
        :rtype: se3kit.translation.Translation
        """
        return Translation(self._matrix[0:3, 3])

    @translation.setter
    def translation(self, val):
        """
        Sets translation component from a Translation object or 3-element ndarray.

        :param val: Translation or 3-element array
        :type val: se3kit.translation.Translation | np.ndarray
        :raises TypeError: if input type is invalid
        """
        self._matrix[0:3, 3] = Translation(val).m

    # ---------------- Matrix / Inverse ----------------
    @property
    def m(self):
        """
        Returns the full 4x4 homogeneous transformation matrix.

        :return: 4x4 transformation matrix
        :rtype: numpy.ndarray
        """
        return self._matrix

    @property
    def inv(self):
        """
        Returns the inverse of this transformation.

        The inverse is computed by inverting the 4x4 transformation matrix.

        :return: Inverse transformation
        :rtype: se3kit.transformation.Transformation
        """
        return Transformation(np.linalg.inv(self._matrix))

    # ---------------- Scaling ----------------
    def convert_m_to_mm(self):
        """
        Converts the translation component from meters to millimeters in-place.

        :return: None
        """
        self._matrix[0:3, 3] *= 1000.0

    def convert_mm_to_m(self):
        """
        Converts the translation component from millimeters to meters in-place.

        :return: None
        """
        self._matrix[0:3, 3] *= 0.001

    def scaled(self, factor):
        """
        Returns a copy of the transformation with translation scaled by a factor.

        Rotation remains unchanged.

        :param factor: Scaling factor
        :type factor: float
        :return: Scaled transformation
        :rtype: se3kit.transformation.Transformation
        """
        return Transformation(self.translation.scaled(factor), self.rotation)

    def scaled_m_to_mm(self):
        """
        Returns a copy of the transformation with translation converted from meters to millimeters.

        :return: Scaled transformation
        :rtype: se3kit.transformation.Transformation
        """
        return Transformation(self.translation.scaled_m_to_mm(), self.rotation)

    def scaled_mm_to_m(self):
        """Returns a copy with translation scaled from millimeters to meters."""
        return Transformation(self.translation.scaled_mm_to_m(), self.rotation)

    def transform_hpoint(self, p):
        """
        Transforms a homogeneous point by this Transformation.

        :param p: HPoint to transform
        :type p: se3kit.hpoint.HPoint
        :return: Transformed HPoint
        :rtype: se3kit.hpoint.HPoint
        :raises AssertionError: if p is not an HPoint
        """
        if not isinstance(p, HPoint):
            raise TypeError(f"transform_hpoint expects HPoint, got {type(p)}")
        return HPoint(self._matrix @ p.m)

    def as_geometry_pose(self):
        """
        Converts this Transformation to a ROS Pose message.

        Works for ROS1 or ROS2 depending on the environment.

        :return: ROS Pose message
        :rtype: geometry_msgs.msg.Pose
        :raises ModuleNotFoundError: if geometry_msgs module not available
        """
        if not use_geomsg:
            raise ModuleNotFoundError("geometry_msgs module not available")
        return Pose(
            position=self.translation.as_geometry_point(),
            orientation=self.rotation.as_geometry_orientation(),
        )

    @staticmethod
    def from_xyz_mm_abc(xyz_abc, degrees=False):
        """
        Creates a Transformation from a 6-element array: XYZ translation in meters
        and ABC Euler angles.

        :param xyzABC: Array-like [x, y, z, A, B, C]
        :type xyzABC: np.ndarray | list
        :return: Transformation object
        :rtype: se3kit.transformation.Transformation
        """
        return Transformation(
            Translation(xyz_abc[:3]), Rotation.from_abc(xyz_abc[3:6], degrees=degrees)
        )

    @staticmethod
    def compose(a, b):
        """
        Composes two Transformations (matrix multiplication).

        :param A: First Transformation
        :param B: Second Transformation
        :type A: se3kit.transformation.Transformation
        :type B: se3kit.transformation.Transformation
        :return: Resulting Transformation
        :rtype: se3kit.transformation.Transformation
        """
        return Transformation(a.matrix @ b.matrix)

    @staticmethod
    def are_close(transform_1, transform_2, rot_tol=0.0174533, trans_tol=0.001, degrees=False):
        """
        Returns a bool specifying whether two transformation matrices are close to each other by checking their
        rotational and translational parts.


        :param transform_1: First transformation matrix
        :type transform_1: se3kit.transformation.Transformation
        :param transform_2: Second transformation matrix
        :type transform_2: se3kit.transformation.Transformation
        :param rot_tol: Rotational tolerance. Default value corresponding to 1 deg
        :type rot_tol: float
        :param trans_tol: Translation tolerance. Default value corresponding to 1 mm
        :type trans_tol: float
        :param degrees: If True, rot_tol angle should be inputted in degrees; otherwise in radians
        :type degrees: bool
        :return: True if the transformation matrices are close, False otherwise
        :rtype: bool
        """
        return Rotation.are_close(
            transform_1.rotation, transform_2.rotation, tol=rot_tol, degrees=degrees
        ) and Translation.are_close(transform_1.translation, transform_2.translation, tol=trans_tol)

    @staticmethod
    def is_valid(mat, verbose=False):
        """
        Checks if the input is a valid 4x4 homogeneous transformation matrix.

        A valid transformation matrix must:
        - Be a numpy ndarray of shape (4, 4)
        - Have a valid rotation part (upper-left 3x3 submatrix)
        - Have a valid translation part (first three elements of the last column)
        - Have the last row equal to [0, 0, 0, 1] (homogeneous row)

        :param mat: Matrix to validate
        :type mat: np.ndarray
        :param verbose: If True, prints detailed validation messages
        :type verbose: bool
        :return: True if valid transformation matrix, False otherwise
        :rtype: bool
        """
        try:
            if not isinstance(mat, np.ndarray):
                raise TypeError(
                    f"Transformation matrix must be of type np.ndarray, got {type(mat)}"
                )

            if not mat.shape == (4, 4):
                raise ValueError(f"Transformation matrix must be 4x4, got {mat.shape}.")

            rot = mat[:3, :3]
            if not Rotation.is_valid(rot):
                raise ValueError("Transformation matrix has invalid rotation part.")
            vec = mat[:3, 3]
            if not Translation.is_valid(vec):
                raise ValueError("Transformation matrix has invalid translation part.")

            homog_vec = mat[3, :]
            expected = np.array([0, 0, 0, 1])

            if not all(is_near(a, b, tol=1e-9) for a, b in zip(homog_vec, expected)):
                raise ValueError(
                    f"Transformation matrix is not affine. Last row must be [0, 0, 0, 1], got {homog_vec}"
                )

        except (ValueError, TypeError) as e:
            if verbose:
                logger.error("Not a valid transformation. %s", e)
            return False

        if verbose:
            logger.info("Matrix is a valid transformation matrix.")
        return True
