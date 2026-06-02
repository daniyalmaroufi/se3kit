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
        if isinstance(other, (int, float)):
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
        if isinstance(other, (int, float)):
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

    def as_geometry_vector3(self):
        """
        Converts the translation to a ROS geometry_msgs Vector3 message.

        Works for ROS1 or ROS2 depending on the environment.

        :return: ROS geometry_msgs.msg.Vector3 message
        :rtype: geometry_msgs.msg.Vector3
        :raises ModuleNotFoundError: If geometry_msgs is not available
        """
        if not use_geomsg:
            raise ModuleNotFoundError("geometry_msgs module not available")
        return Vector3(x=self.x, y=self.y, z=self.z)

    def __repr__(self):
        """
        Official string representation of the Translation object.

        :return: Reconstructable string representation.
        :rtype: str
        """
        return f"Translation([{float(self.m[0])!r}, {float(self.m[1])!r}, {float(self.m[2])!r}])"

    def __str__(self):
        """
        Human-friendly string representation.

        :return: Formatted string with xyz values.
        :rtype: str
        """
        return f"Translation(xyz=[{self.m[0]:.6f}, {self.m[1]:.6f}, {self.m[2]:.6f}])"

    def to_dict(self):
        """
        Serializes the translation to a plain Python dictionary.

        :return: Dictionary with keys 'type' and 'xyz'.
        :rtype: dict
        """
        return {
            "type": "Translation",
            "xyz": [float(self.m[0]), float(self.m[1]), float(self.m[2])],
        }

    @staticmethod
    def from_dict(d):
        """
        Constructs a Translation from a dictionary.

        :param d: Dictionary with 'xyz' key.
        :type d: dict
        :return: Translation object.
        :rtype: Translation
        :raises ValueError: If 'xyz' key is missing.
        """
        if "xyz" in d:
            return Translation(d["xyz"])
        raise ValueError("Dict must contain 'xyz' key")

    def to_json(self, indent=2):
        """
        Serializes the translation to a JSON string.

        :param indent: JSON indentation level, defaults to 2.
        :type indent: int
        :return: JSON string.
        :rtype: str
        """
        import json
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(json_str):
        """
        Constructs a Translation from a JSON string.

        :param json_str: JSON string containing translation data.
        :type json_str: str
        :return: Translation object.
        :rtype: Translation
        """
        import json
        return Translation.from_dict(json.loads(json_str))

    def to_csv(self, path, header=True):
        """
        Writes the translation to a CSV file.

        :param path: File path to write to.
        :type path: str or pathlib.Path
        :param header: Whether to write a header row, defaults to True.
        :type header: bool
        """
        import csv
        from pathlib import Path

        path = Path(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(["x", "y", "z"])
            writer.writerow([float(self.m[0]), float(self.m[1]), float(self.m[2])])

    @staticmethod
    def from_csv(path):
        """
        Reads a Translation from the first data row of a CSV file.

        :param path: File path to read from.
        :type path: str or pathlib.Path
        :return: Translation object.
        :rtype: Translation
        """
        import csv
        from pathlib import Path

        path = Path(path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Skip header if first row is non-numeric
        data_row = rows[0]
        try:
            float(data_row[0])
        except ValueError:
            data_row = rows[1]

        return Translation([float(v) for v in data_row])

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
