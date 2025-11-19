"""
Unit tests for SE3Kit modules.

This module defines the Tests class for verifying functionality
of Robot, Rotation, and Transformation classes under both ROS1 and ROS2
environments. It can be executed directly to run tests.
"""

import unittest
import numpy as np
import math

# Import ROS compatibility layer and core SE3Kit modules
from se3kit.ros_compat import ROS_VERSION
from se3kit.robot import Robot  # assuming robot.py exists in same folder
import se3kit.rotation as Rotation
import se3kit.translation as Translation
import se3kit.transformation as Transformation


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
        Mat = np.asarray(
            [[0.8389628,  0.4465075, -0.3110828],
             [0.1087932,  0.4224873,  0.8998158],
             [0.5332030, -0.7887557,  0.3058742 ]])
        self.assertTrue(Rotation.Rotation.is_valid(Mat, verbose=False), "Expected Mat to be a valid rotation matrix")

        # Invalid rotation matrix (determinant not ~1)
        Mat_bad = np.asarray(
            [[1.8389628,  0.4465075, -0.3110828],
             [0.1087932,  0.4224873,  0.8998158],
             [0.5332030, -0.7887557,  0.3058742 ]])
        self.assertFalse(Rotation.Rotation.is_valid(Mat_bad, verbose=False), "Expected Mat_bad to be invalid as a rotation matrix")


    def test_translation_vector_validity(self):
        vec = np.asarray([1, 2, 3])
        self.assertTrue(Translation.Translation.is_valid(vec, verbose=False), "Expected vec to be a valid translation vector")

        vec_bad = np.asarray([[1], [2], [3.0], [3]])
        self.assertFalse(Translation.Translation.is_valid(vec_bad, verbose=False), "Expected vec_bad to be invalid (size != 3)")

    def test_transformation_validity(self):
        # 3x3 input -> invalid transformation (expects 4x4)
        Mat3 = np.asarray(
            [[0.8389628,  0.4465075, -0.3110828],
             [0.1087932,  0.4224873,  0.8998158],
             [0.5332030, -0.7887557,  0.3058742 ]])
        self.assertFalse(Transformation.Transformation.is_valid(Mat3, verbose=False), "3x3 matrix should not be a valid transformation")

        # 3x4 input -> invalid
        Mat3x4 = np.asarray(
            [[0.8389628,  0.4465075, -0.3110828, 1],
             [0.1087932,  0.4224873,  0.8998158, 2.0],
             [0.5332030, -0.7887557,  0.3058742 , -3]])
        self.assertFalse(Transformation.Transformation.is_valid(Mat3x4, verbose=False), "3x4 matrix should not be a valid transformation")

        # Proper 4x4 homogeneous transformation
        Mat4 = np.asarray(
            [[0.8389628,  0.4465075, -0.3110828, 1],
             [0.1087932,  0.4224873,  0.8998158, 2.0],
             [0.5332030, -0.7887557,  0.3058742 , -3],
             [0, 0, 0, 1]])
        self.assertTrue(Transformation.Transformation.is_valid(Mat4, verbose=False), "4x4 matrix should be a valid transformation")

    def test_rotation_from_zyx(self):
        angles = [0, 0, 0]
        self.assertTrue(Rotation.Rotation.from_rpy(angles).is_identity(), "Rotation from zero RPY should be identity")


# --- Script execution entry point ---
if __name__ == '__main__':
    # Let unittest discover and run all tests
    unittest.main()
