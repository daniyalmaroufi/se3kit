import numpy as np

# Constants to avoid magic numbers
_CARTESIAN_SIZE = 3
_HOMOGENEOUS_SIZE = 4


class HPoint:
    """Represents a homogeneous point in 3D space (4x1 vector)."""

    def __init__(self, *args):
        """
        Initializes HPoint from:

        - Three separate coordinates x, y, z
        - A 3-element numpy array
        - A 4-element numpy array (homogeneous coordinates)

        :param args: variable-length arguments
        :raises ValueError: if array size is not 3 or 4, or invalid number of arguments
        :raises TypeError: if input type is invalid
        """
        if len(args) == _CARTESIAN_SIZE:
            # Three separate coordinates provided (x, y, z)
            self.m = np.reshape([args[0], args[1], args[2], 1.0], (4, 1))
        elif len(args) == 1:
            # A single argument provided — could be a NumPy array or similar
            arr = args[0]
            if isinstance(arr, np.ndarray):
                if arr.size == _CARTESIAN_SIZE:
                    # If it's a 3-element vector [x, y, z]
                    # → Convert to homogeneous coordinates by appending 1
                    self.m = np.reshape([arr[0], arr[1], arr[2], 1.0], (4, 1))
                elif arr.size == _HOMOGENEOUS_SIZE:
                    # If it's already a 4-element homogeneous vector [x, y, z, w]
                    # → Just reshape it into a 4x1 column vector
                    self.m = np.reshape(arr, (4, 1))
                else:
                    # Invalid array size — must be either 3 (Cartesian) or 4 (homogeneous)
                    raise ValueError(f"Cannot initialize HPoint from array of size {arr.size}")
            else:
                # Invalid input type — must be a NumPy array
                raise TypeError(f"Cannot initialize HPoint from type {type(arr)}")
        else:
            # Invalid number of arguments — must be either 3 (x, y, z) or 1 (array)
            raise ValueError(f"Cannot initialize HPoint from {len(args)} arguments")

    @property
    def x(self):
        """
        Get the x-coordinate of the homogeneous point.

        :return: The x-coordinate value.
        :rtype: float
        """
        return self.m[0, 0]

    @x.setter
    def x(self, val):
        """
        Set the x-coordinate of the homogeneous point.

        :param val: New x-coordinate value.
        :type val: float
        """
        self.m[0, 0] = val

    @property
    def y(self):
        """
        Get the y-coordinate of the homogeneous point.

        :return: The y-coordinate value.
        :rtype: float
        """
        return self.m[1, 0]

    @y.setter
    def y(self, val):
        """
        Set the y-coordinate of the homogeneous point.

        :param val: New y-coordinate value.
        :type val: float
        """
        self.m[1, 0] = val

    @property
    def z(self):
        """
        Get the z-coordinate of the homogeneous point.

        :return: The z-coordinate value.
        :rtype: float
        """
        return self.m[2, 0]

    @z.setter
    def z(self, val):
        """
        Set the z-coordinate of the homogeneous point.

        :param val: New z-coordinate value.
        :type val: float
        """
        self.m[2, 0] = val

    @property
    def xyz(self):
        """
        Get the 3D Cartesian coordinates of the point as a NumPy array.

        :return: A 1D NumPy array containing [x, y, z].
        :rtype: numpy.ndarray
        """
        return self.m[0:3, 0]

    # ---------------- Convenience methods ----------------
    def as_array(self):
        """
        Get the full 4x1 homogeneous vector representation of the point.

        This includes the x, y, z, and homogeneous coordinate (typically 1).

        :return: A 4x1 NumPy array representing [x, y, z, 1]^T.
        :rtype: numpy.ndarray
        """
        return self.m.copy()

    def __repr__(self):
        """
        Official string representation of the HPoint object.

        :return: A string showing the class name and coordinate values.
        :rtype: str
        """
        return f"HPoint(x={self.x}, y={self.y}, z={self.z})"

    def __str__(self):
        """
        User-friendly string representation of the homogeneous point.

        :return: A formatted string "[x, y, z, 1]".
        :rtype: str
        """
        return f"[{self.x}, {self.y}, {self.z}, 1]"
