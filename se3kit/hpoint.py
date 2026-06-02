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
        - A 3-element numpy array or list
        - A 4-element numpy array or list (homogeneous coordinates)

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
            if isinstance(arr, (list, np.ndarray)):
                if isinstance(arr, list):
                    arr = np.array(arr)
                if arr.size == _CARTESIAN_SIZE:
                    # If it's a 3-element vector [x, y, z]
                    # → Convert to homogeneous coordinates by appending 1
                    self.m = np.reshape([arr[0], arr[1], arr[2], 1.0], (4, 1))
                elif arr.size == _HOMOGENEOUS_SIZE:
                    # If it's already a 4-element homogeneous vector [x, y, z, w]
                    # → Just reshape it into a 4x1 column vector
                    if arr.flat[-1] != 1:
                        raise ValueError("The last element of the homogeneous vector must be 1.0")

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

    def to_dict(self):
        """
        Serializes the homogeneous point to a plain Python dictionary.

        :return: Dictionary with keys 'type' and 'xyz'.
        :rtype: dict
        """
        return {
            "type": "HPoint",
            "xyz": [float(self.x), float(self.y), float(self.z)],
        }

    @staticmethod
    def from_dict(d):
        """
        Constructs an HPoint from a dictionary.

        :param d: Dictionary with 'xyz' key (3-element list).
        :type d: dict
        :return: HPoint object.
        :rtype: HPoint
        :raises ValueError: If 'xyz' key is missing.
        """
        if "xyz" in d:
            return HPoint(d["xyz"])
        raise ValueError("Dict must contain 'xyz' key")

    def to_json(self, indent=2):
        """
        Serializes the homogeneous point to a JSON string.

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
        Constructs an HPoint from a JSON string.

        :param json_str: JSON string containing point data.
        :type json_str: str
        :return: HPoint object.
        :rtype: HPoint
        """
        import json
        return HPoint.from_dict(json.loads(json_str))

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
