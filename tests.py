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

    def setUp(self):
        """
        Set up the testing environment before each unit test.

        This method initializes the test fixture by creating a robot instance 
        and defining a set of test joint angles (in both degrees and radians)
        for use in all kinematic tests.

        This method is automatically invoked before each test_* method 
        by the unittest framework.

        Attributes
        ----------
        robot : Robot
            Instance of the IIWA robot model used for testing.
        ja_deg : list of float
            Joint angles in degrees for the test configuration.
        ja_rad : numpy.ndarray
            Joint angles converted to radians for FK calculations.
        """
        self.robot = Robot.create_iiwa()
        self.ja_deg = [-0.01, -35.10, 47.58, 24.17, 0.00, 0.00, 0.00]
        self.ja_rad = np.deg2rad(self.ja_deg)

    def test_forward_kinematics(self):
        """
        Verify the correctness of the robot's forward kinematics computation.

        This test checks whether the `FK_space` method produces the expected 
        end-effector rotation (in ZYX Euler angles) and translation vectors 
        for a known IIWA robot joint configuration.

        The computed results are compared against pre-validated reference values 
        using NumPy’s `allclose` function with a tolerance of 1e-8.

        Raises:
            AssertionError:If the computed rotation or translation values differ from expected results
            beyond the specified numerical tolerance.
        """
        t = self.robot.FK_space(self.ja_rad)
        expected_rotation = [68.28584782, -43.53993415, -35.84364870]
        expected_translation = [-636.32792290, -158.87809407, 1012.70702237]

        self.assertTrue(
            np.allclose(t.rotation.as_zyx(degrees=True), expected_rotation, atol=1e-8),
            "Rotation (ZYX Euler) output mismatch in FK_space"
        )
        self.assertTrue(
            np.allclose(t.translation.m.flatten(), expected_translation, atol=1e-8),
            "Translation vector output mismatch in FK_space"
        )

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
        Mat = np.asarray(
            [[0.8389628,  0.4465075, -0.3110828],
             [0.1087932,  0.4224873,  0.8998158],
             [0.5332030, -0.7887557,  0.3058742 ]])
        print(Mat)
        Rotation.Rotation.is_valid(Mat, verbose= True)
        # temp_test = Rotation.Rotation(Mat)

        Mat = np.asarray(
            [[1.8389628,  0.4465075, -0.3110828],
             [0.1087932,  0.4224873,  0.8998158],
             [0.5332030, -0.7887557,  0.3058742 ]])
        print('\n', Mat)
        Rotation.Rotation.is_valid(Mat, verbose= True)
        # temp_test = Rotation.Rotation(Mat)


    def test_translation_vector_validity(self):
        vec = np.asarray([1, 2, 3])
        print(vec)
        Translation.Translation.is_valid(vec, verbose= True)

        vec = np.asarray([[1], [2], [3.0], [3]])
        print('\n', vec)
        Translation.Translation.is_valid(vec, verbose= True)
        # temp_test = Translation.Translation(vec)

    def test_transformation_validity(self):
        Mat = np.asarray(
            [[0.8389628,  0.4465075, -0.3110828],
             [0.1087932,  0.4224873,  0.8998158],
             [0.5332030, -0.7887557,  0.3058742 ]])
        
        print(Mat)
        Transformation.Transformation.is_valid(Mat, verbose= True)
        # temp_test = Transformation.Transformation(Mat)

        Mat = np.asarray(
            [[0.8389628,  0.4465075, -0.3110828, 1],
             [0.1087932,  0.4224873,  0.8998158, 2.0],
             [0.5332030, -0.7887557,  0.3058742 , -3]])
        print('\n', Mat)
        Transformation.Transformation.is_valid(Mat, verbose= True)
        # temp_test = Transformation.Transformation(Mat)

        Mat = np.asarray(
            [[0.8389628,  0.4465075, -0.3110828, 1],
             [0.1087932,  0.4224873,  0.8998158, 2.0],
             [0.5332030, -0.7887557,  0.3058742 , -3],
             [0, 0, 0, 1]])
        print('\n', Mat)
        Transformation.Transformation.is_valid(Mat, verbose= True)
        # temp_test = Transformation.Transformation(Mat)

    def test_rotation_from_zyx(self):
        angles = [0 ,0 ,0]
        print(Rotation.Rotation.from_rpy(angles).is_identity())


# --- Script execution entry point ---
if __name__ == '__main__':
    Tests = Tests()
    Tests.test_rotation_from_zyx()
