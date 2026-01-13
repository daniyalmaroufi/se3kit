"""
translation.py

Represents a 3D translation vector and provides utilities for arithmetic,
scaling, unit conversion, and ROS message conversion.

Compatible with ROS1 and ROS2 using ros_compat.py.
"""

import logging

import numpy as np

from se3kit.hpoint import HPoint
from se3kit.ros_compat import Point, Vector3, use_geomsg

# module logger
logger = logging.getLogger(__name__)

# Constants
_CARTESIAN_SIZE = 3


class Translation:
    """Represents a 3D translation vector."""

    def __init__(self, init_xyz=None, unit="m"):
        """
        Initializes translation from various sources.

        :param init_xyz: Can be one of:
            - None (defaults to zero vector)
            - numpy array or list-like with 3 elements
            - HPoint instance
            - ROS Point or Vector3 message
            - Another Translation instance
        :type init_xyz: np.ndarray | list | HPoint | Translation | Point | Vector3 | None
        :raises ValueError: If a numpy array or list is provided that does not have exactly 3 elements
        """
        if init_xyz is None:
            # Default zero vector
            self.m = np.zeros(3)
        elif use_geomsg and isinstance(init_xyz, (Point, Vector3)):
            # ROS Point/Vector3 message
            self.m = np.array([init_xyz.x, init_xyz.y, init_xyz.z])
        elif isinstance(init_xyz, HPoint):
            # Homogeneous point
            self.m = init_xyz.xyz
        elif isinstance(init_xyz, Translation):
            # Copy constructor
            self.m = np.copy(init_xyz.m)
        else:
            # Array or list-like input
            if not Translation.is_valid(init_xyz):
                raise ValueError("Translation vector is invalid.")
            self.m = np.squeeze(np.array(init_xyz, dtype=float))

        self.unit = unit

    def __add__(self, other):
        """
        Adds two Translation vectors element-wise.

        :param other: Another Translation instance
        :type other: se3kit.translation.Translation
        :return: New Translation representing the sum
        :rtype: se3kit.translation.Translation
        :raises TypeError: If `other` is not a Translation
        """
        if isinstance(other, Translation):
            return Translation(self.m + other.m)
        raise TypeError(f"Cannot add Translation with {type(other)}")

    def __sub__(self, other):
        """
        Subtracts another Translation vector element-wise.

        :param other: Another Translation instance
        :type other: se3kit.translation.Translation
        :return: New Translation representing the difference
        :rtype: se3kit.translation.Translation
        :raises TypeError: If `other` is not a Translation
        """
        if isinstance(other, Translation):
            return Translation(self.m - other.m)
        raise TypeError(f"Cannot subtract Translation with {type(other)}")

    def __mul__(self, other):
        """
        Scales the Translation by a scalar.

        :param other: Scalar factor
        :type other: int | float
        :return: New scaled Translation
        :rtype: se3kit.translation.Translation
        :raises TypeError: If `other` is not a numeric scalar
        """
        if isinstance(other, int | float):
            return Translation(self.m * other)
        raise TypeError(f"Cannot multiply Translation with {type(other)}")

    def __truediv__(self, other):
        """
        Divides the Translation by a scalar.

        :param other: Scalar divisor
        :type other: int | float
        :return: New scaled Translation
        :rtype: se3kit.translation.Translation
        :raises TypeError: If `other` is not a numeric scalar
        """
        if isinstance(other, int | float):
            return Translation(self.m / other)
        raise TypeError(f"Cannot divide Translation with {type(other)}")

    @property
    def x(self):
        """
        Returns x-component.

        :return: x-component
        :rtype: float
        """
        return self.m[0]

    @x.setter
    def x(self, val):
        """
        Sets the x-component of the translation.

        :param val: New x value
        :type val: float
        """
        self.m[0] = val

    @property
    def y(self):
        """
        Returns the y-component of the translation.

        :return: y-component
        :rtype: float
        """
        return self.m[1]

    @y.setter
    def y(self, val):
        """
        Sets the y-component of the translation.

        :param val: New y value
        :type val: float
        """
        self.m[1] = val

    @property
    def z(self):
        """
        Returns the z-component of the translation.

        :return: z-component
        :rtype: float
        """
        return self.m[2]

    @z.setter
    def z(self, val):
        """
        Sets the z-component of the translation.

        :param val: New z value
        :type val: float
        """
        self.m[2] = val

    @property
    def xyz(self):
        """
        Returns the full translation vector as a numpy array.

        :return: 3-element vector [x, y, z]
        :rtype: numpy.ndarray
        """
        return self.m

    def norm(self):
        """
        Computes the Euclidean norm (magnitude) of the translation vector.

        :return: Euclidean norm
        :rtype: float
        """
        return np.linalg.norm(self.m)

    def scale(self, factor, inplace=True):
        """
        Scales the translation.

        :param factor: Scaling factor
        :type factor: float
        :param inplace: Whether to modify the current object in-place or return a new one, defaults to True
        :type inplace: bool
        :return: Scaled Translation if not inplace, else None
        :rtype: se3kit.translation.Translation | None
        """
        if inplace:
            self.m *= factor
            return None
        return Translation(self.m * factor)

    def convert_m_to_mm(self, inplace=True):
        """
        Converts the translation from meters to millimeters.

        :param inplace: Whether to modify the current object in-place or return a new one, defaults to True
        :type inplace: bool
        :return: Scaled Translation if not inplace, else None
        :rtype: se3kit.translation.Translation | None
        """
        return self.scale(1000.0, inplace)

    def convert_mm_to_m(self, inplace=True):
        """
        Converts the translation from millimeters to meters.

        :param inplace: Whether to modify the current object in-place or return a new one, defaults to True
        :type inplace: bool
        :return: Scaled Translation if not inplace, else None
        :rtype: se3kit.translation.Translation | None
        """
        return self.scale(0.001, inplace)

    def as_geometry_point(self):
        """
        Converts the translation to a ROS geometry_msgs Point message.

        Works for ROS1 or ROS2 depending on the environment.

        :return: ROS geometry_msgs.msg.Point message
        :rtype: geometry_msgs.msg.Point
        :raises ModuleNotFoundError: If geometry_msgs is not available
        """
        if not use_geomsg:
            raise ModuleNotFoundError("geometry_msgs module not available")
        return Point(x=self.x, y=self.y, z=self.z)

    @staticmethod
    def are_close(trans_1, trans_2, tol=0.001):
        """
        Returns a bool specifying whether two translation vectors are close to each other within a given tolerance.
        The comparison is based on the Euclidean distance between the two translation vectors.
        :param trans_1: First translation
        :type trans_1: se3kit.translation.Translation
        :param trans_2: Second translation
        :type trans_2: se3kit.translation.Translation
        :param tol: Tolerance. Default value corresponding to 1 mm
        :type tol: float
        :return: True if the translation vectors are close, False otherwise
        """
        return (trans_1 - trans_2).norm() < tol

    @staticmethod
    def is_valid(vec, verbose=False):
        """
        Checks if the input is a valid translation vector.

        A valid translation vector is an array-like object of length 3.

        :param vec: The vector to validate.
        :type vec: np.ndarray | list | tuple
        :param verbose: If True, prints validation messages.
        :type verbose: bool
        :return: True if valid, False otherwise.
        :rtype: bool
        """
        try:
            # Attempt to convert to a NumPy array to handle lists/tuples
            vec_np = np.array(vec)

            if vec_np.ndim != 1:
                raise ValueError(
                    f"Translation vector must be 1-dimensional, got {vec_np.ndim} dimensions"
                )

            if vec_np.size != _CARTESIAN_SIZE:
                raise ValueError(
                    f"Translation vector must be of length {_CARTESIAN_SIZE}, got {vec_np.size}"
                )

        except (ValueError, TypeError) as e:
            if verbose:
                logger.error("Not a valid translation. %s", e)
            return False

        if verbose:
            logger.info("Vector is a valid translation vector.")
        return True
