"""
Unit tests for SE3Kit modules.

This module defines the Tests class for verifying functionality
of Robot, Rotation, and Transformation classes under both ROS1 and ROS2
environments. It can be executed directly to run tests.
"""

import unittest

import numpy as np

from se3kit import rotation, transformation, translation

# Import ROS compatibility layer and core SE3Kit modules
from se3kit.ros_compat import ROS_VERSION


class Tests(unittest.TestCase):
    """Collection of unit tests for verifying SE3Kit robot kinematics and transformations."""

    def test_ros_version(self):
        """
        Validate ROS environment detection.

        This test ensures that the ROS compatibility layer correctly identifies
        the active ROS version. It confirms that `ROS_VERSION` is one of the
        supported values: 0 (no ROS), 1 (ROS1), or 2 (ROS2).

        Raises:
            AssertionError: If the detected ROS version is not one of the expected values.
        """
        self.assertIn(ROS_VERSION, [0, 1, 2], "Invalid ROS version detected.")

    def test_rotation_matrix_validity(self):
        # Valid rotation matrix
        mat = np.asarray(
            [
                [0.8389628, 0.4465075, -0.3110828],
                [0.1087932, 0.4224873, 0.8998158],
                [0.5332030, -0.7887557, 0.3058742],
            ]
        )
        self.assertTrue(
            rotation.Rotation.is_valid(mat, verbose=False),
            "Expected mat to be a valid rotation matrix",
        )

        # Invalid rotation matrix (determinant not ~1)
        mat_bad = np.asarray(
            [
                [1.8389628, 0.4465075, -0.3110828],
                [0.1087932, 0.4224873, 0.8998158],
                [0.5332030, -0.7887557, 0.3058742],
            ]
        )
        self.assertFalse(
            rotation.Rotation.is_valid(mat_bad, verbose=False),
            "Expected mat_bad to be invalid as a rotation matrix",
        )

    def test_translation_vector_validity(self):
        vec = np.asarray([1, 2, 3])
        self.assertTrue(
            translation.Translation.is_valid(vec, verbose=False),
            "Expected vec to be a valid translation vector",
        )

        vec_bad = np.asarray([[1], [2], [3.0], [3]])
        self.assertFalse(
            translation.Translation.is_valid(vec_bad, verbose=False),
            "Expected vec_bad to be invalid (size != 3)",
        )

    def test_transformation_validity(self):
        # 3x3 input -> invalid transformation (expects 4x4)
        mat3 = np.asarray(
            [
                [0.8389628, 0.4465075, -0.3110828],
                [0.1087932, 0.4224873, 0.8998158],
                [0.5332030, -0.7887557, 0.3058742],
            ]
        )
        self.assertFalse(
            transformation.Transformation.is_valid(mat3, verbose=False),
            "3x3 matrix should not be a valid transformation",
        )

        # 3x4 input -> invalid
        mat3x4 = np.asarray(
            [
                [0.8389628, 0.4465075, -0.3110828, 1],
                [0.1087932, 0.4224873, 0.8998158, 2.0],
                [0.5332030, -0.7887557, 0.3058742, -3],
            ]
        )
        self.assertFalse(
            transformation.Transformation.is_valid(mat3x4, verbose=False),
            "3x4 matrix should not be a valid transformation",
        )

        # Proper 4x4 homogeneous transformation
        mat4 = np.asarray(
            [
                [0.8389628, 0.4465075, -0.3110828, 1],
                [0.1087932, 0.4224873, 0.8998158, 2.0],
                [0.5332030, -0.7887557, 0.3058742, -3],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(
            transformation.Transformation.is_valid(mat4, verbose=False),
            "4x4 matrix should be a valid transformation",
        )

    def test_rotation_from_zyx(self):
        angles = [0, 0, 0]
        self.assertTrue(
            rotation.Rotation.from_rpy(angles).is_identity(),
            "Rotation from zero RPY should be identity",
        )


# --- Script execution entry point ---
if __name__ == "__main__":
    # Let unittest discover and run all tests
    unittest.main()
