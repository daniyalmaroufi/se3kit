import logging

import numpy as np

from se3kit.hpoint import HPoint
from se3kit.ros_compat import ROS_VERSION, Pose, PoseStamped, Transform, use_geomsg
from se3kit.rotation import Rotation
from se3kit.translation import Translation
from se3kit.utils import is_near

# Constants to avoid magic-numbers in argument checks
_TRANSLATION_ROTATION_ARG_COUNT = 2
_ROS_VERSION_2 = 2

logger = logging.getLogger(__name__)


class Transformation:
    """Represents a 4x4 homogeneous transformation matrix with rotation and translation."""

    def __init__(self, *args):
        """
        Initializes the Transformation from various sources.

        Can initialize from:
        - A 4x4 numpy matrix
        - A ROS geometry_msgs.msg.Pose message (ROS1 or ROS2)
        - A ROS geometry_msgs.msg.PoseStamped message (TF2, ROS1 or ROS2)
        - A ROS geometry_msgs.msg.Transform message (TF2, ROS1 or ROS2)
        - A se3kit.translation.Translation object
        - se3kit.translation.Translation + se3kit.rotation.Rotation (in any order)

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

            elif use_geomsg and self._init_from_ros_message(init):
                pass

            elif isinstance(init, Translation):
                # Single argument is a Translation object
                # Only the translation is set; rotation defaults to identity
                self.translation = init

            elif isinstance(init, Rotation):
                # Single argument is a Rotation object
                # Only the rotation is set; translation defaults to zero
                self.rotation = init

            else:
                raise TypeError(f"Invalid argument type for Transformation: {type(init)}")

        elif len(args) == _TRANSLATION_ROTATION_ARG_COUNT:
            arg0, arg1 = args
            if isinstance(arg0, Translation) and isinstance(arg1, Rotation):
                # Translation, Rotation
                self.translation = arg0
                self.rotation = arg1
            elif isinstance(arg0, Rotation) and isinstance(arg1, Translation):
                # Rotation, Translation (Swapped order support)
                self.translation = arg1
                self.rotation = arg0
            else:
                raise TypeError(f"Invalid arguments for Transformation: {args}")

        elif len(args) > _TRANSLATION_ROTATION_ARG_COUNT:
            raise TypeError(f"Too many arguments for Transformation: {args}")

    def _init_from_ros_message(self, msg):
        """Helper method to initialize from ROS message types."""
        if isinstance(msg, PoseStamped):
            self.rotation = Rotation(msg.pose.orientation)
            self.translation = Translation(msg.pose.position)
        elif isinstance(msg, Transform):
            self.rotation = Rotation(msg.rotation)
            self.translation = Translation(msg.translation)
        elif isinstance(msg, Pose):
            self.rotation = Rotation(msg.orientation)
            self.translation = Translation(msg.position)
        else:
            return False
        return True

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
        elif isinstance(other, HPoint):
            return HPoint(self._matrix @ other.m)
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

    def inv(self):
        """
        Returns the inverse of this transformation.

        The inverse is computed by inverting the 4x4 transformation matrix.

        :return: Inverse transformation
        :rtype: se3kit.transformation.Transformation
        """
        return Transformation(np.linalg.inv(self._matrix))

    def scale(self, factor, inplace=True):
        """
        Scales the translation component by a factor.

        :param factor: Scaling factor
        :type factor: float
        :param inplace: Whether to modify in-place or return a copy, defaults to True
        :return: None if inplace, else a new Transformation
        """
        if inplace:
            self._matrix[0:3, 3] *= factor
            return None
        return Transformation(self.translation.scale(factor, inplace=False), self.rotation)

    def convert_m_to_mm(self, inplace=True):
        """
        Converts the translation component from meters to millimeters.

        :param inplace: Whether to modify in-place or return a copy, defaults to True
        :return: None if inplace, else a new Transformation
        """
        return self.scale(1000, inplace)

    def convert_mm_to_m(self, inplace=True):
        """
        Converts the translation component from millimeters to meters.

        :param inplace: Whether to modify in-place or return a copy, defaults to True
        :return: None if inplace, else a new Transformation
        """
        return self.scale(0.001, inplace)

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

    def as_pose_stamped(self, frame_id="base_link", stamp=None):
        """
        Converts this Transformation to a ROS PoseStamped message (TF2).

        Works for ROS1 or ROS2 depending on the environment.

        :param frame_id: Frame ID for the message header, defaults to "base_link"
        :type frame_id: str
        :param stamp: ROS timestamp for the message header, defaults to current time (ROS only)
        :return: ROS PoseStamped message
        :rtype: geometry_msgs.msg.PoseStamped
        :raises ModuleNotFoundError: if geometry_msgs module not available
        """
        if not use_geomsg or PoseStamped is None:
            raise ModuleNotFoundError("geometry_msgs.msg.PoseStamped not available")

        if stamp is None:
            try:
                if ROS_VERSION == _ROS_VERSION_2:
                    import rclpy

                    stamp = rclpy.clock.Clock().now()
                elif ROS_VERSION == 1:
                    import rospy

                    stamp = rospy.Time.now()
            except Exception as e:
                logger.debug("Failed to get ROS timestamp: %s", e)

        # Create header with frame_id and timestamp
        from geometry_msgs.msg import Header  # type: ignore

        header = Header(frame_id=frame_id, stamp=stamp)

        return PoseStamped(
            header=header,
            pose=self.as_geometry_pose(),
        )

    def as_transform(self):
        """
        Converts this Transformation to a ROS Transform message (TF2).

        Works for ROS1 or ROS2 depending on the environment.

        :return: ROS Transform message
        :rtype: geometry_msgs.msg.Transform
        :raises ModuleNotFoundError: if geometry_msgs module not available
        """
        if not use_geomsg or Transform is None:
            raise ModuleNotFoundError("geometry_msgs.msg.Transform not available")

        return Transform(
            translation=self.translation.as_geometry_vector3(),
            rotation=self.rotation.as_geometry_orientation(),
        )

    def __repr__(self):
        """
        Official string representation of the Transformation object.

        Returns a string that can reconstruct the object via eval() when numpy is imported.

        :return: Reconstructable string representation.
        :rtype: str
        """
        matrix_str = np.array2string(self._matrix, separator=', ')
        return f"Transformation(np.array({matrix_str}))"

    def __str__(self):
        """
        Human-friendly string representation showing translation and RPY angles.

        :return: Formatted multi-line string.
        :rtype: str
        """
        t = self.translation
        rpy = self.rotation.as_rpy(degrees=True)
        return (
            f"Transformation(\n"
            f"  xyz=[{t.x:.6f}, {t.y:.6f}, {t.z:.6f}],\n"
            f"  rpy_deg=[{rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f}]\n"
            f")"
        )

    def to_dict(self):
        """
        Serializes the transformation to a plain Python dictionary.

        Includes the full 4x4 matrix and decomposed translation/rotation sub-dicts.

        :return: Dictionary with keys 'type', 'translation', 'rotation', 'matrix'.
        :rtype: dict
        """
        return {
            "type": "Transformation",
            "translation": self.translation.to_dict(),
            "rotation": self.rotation.to_dict(),
            "matrix": self._matrix.tolist(),
        }

    @staticmethod
    def from_dict(d):
        """
        Constructs a Transformation from a dictionary.

        Checks for keys in order of fidelity: 'matrix' (lossless 4x4) > decomposed
        'translation' + 'rotation'.

        :param d: Dictionary with transformation data.
        :type d: dict
        :return: Transformation object.
        :rtype: Transformation
        :raises ValueError: If no recognized keys are found.
        """
        if "matrix" in d:
            return Transformation(np.array(d["matrix"]))
        if "translation" in d and "rotation" in d:
            return Transformation(
                Translation.from_dict(d["translation"]),
                Rotation.from_dict(d["rotation"]),
            )
        raise ValueError("Dict must contain 'matrix' or both 'translation' and 'rotation'")

    def to_json(self, indent=2):
        """
        Serializes the transformation to a JSON string.

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
        Constructs a Transformation from a JSON string.

        :param json_str: JSON string containing transformation data.
        :type json_str: str
        :return: Transformation object.
        :rtype: Transformation
        """
        import json
        return Transformation.from_dict(json.loads(json_str))

    def to_csv(self, path, header=True, rotation_format="quaternion"):
        """
        Writes the transformation to a CSV file.

        :param path: File path to write to.
        :type path: str or pathlib.Path
        :param header: Whether to write a header row, defaults to True.
        :type header: bool
        :param rotation_format: Rotation output format. One of 'quaternion', 'rpy_rad',
            'rpy_deg', or 'matrix'. Defaults to 'quaternion'.
        :type rotation_format: str
        """
        import csv
        from pathlib import Path

        path = Path(path)
        rot_headers, rot_values = self.rotation._csv_data(rotation_format)
        headers = ["x", "y", "z"] + rot_headers
        values = [float(self.translation.x), float(self.translation.y), float(self.translation.z)] + rot_values

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(headers)
            writer.writerow(values)

    @staticmethod
    def from_csv(path, rotation_format="quaternion"):
        """
        Reads a Transformation from the first data row of a CSV file.

        :param path: File path to read from.
        :type path: str or pathlib.Path
        :param rotation_format: Rotation input format matching the CSV columns.
            Defaults to 'quaternion'.
        :type rotation_format: str
        :return: Transformation object.
        :rtype: Transformation
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

        values = [float(v) for v in data_row]
        trans = Translation(values[:3])
        rot_values = values[3:]

        if rotation_format == "quaternion":
            rot = Rotation.from_quat(rot_values)
        elif rotation_format == "rpy_rad":
            rot = Rotation.from_rpy(rot_values)
        elif rotation_format == "rpy_deg":
            rot = Rotation.from_rpy(rot_values, degrees=True)
        elif rotation_format == "matrix":
            rot = Rotation(np.array(rot_values).reshape(3, 3))
        else:
            raise ValueError(f"Unknown rotation format: {rotation_format}")

        return Transformation(trans, rot)

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
        return Transformation(a.m @ b.m)

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
