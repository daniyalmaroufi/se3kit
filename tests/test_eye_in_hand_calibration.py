"""
Unit tests for EyeInHandCalibration class.

Uses test pose data from four files:
- robot_rotations.txt
- camera_rotations.txt
- robot_translations.txt
- camera_translations.txt
to build robot and camera transformations and verifies that calibration runs and returns a valid result.
"""

import unittest

import numpy as np
import quaternion

from se3kit.calibration.eye_in_hand_calibration import EyeInHandCalibration
from se3kit.rotation import Rotation
from se3kit.transformation import Transformation
from se3kit.translation import Translation


class TestEyeInHandCalibration(unittest.TestCase):
    """Tests for the EyeInHandCalibration class."""

    @staticmethod
    def _load_pose_data():
        """Loads test poses for eye in hand calibration and converts entries into transformation lists."""

        # Robot and camera quaternions
        q_robot = np.loadtxt("tests/eye_in_hand_test_poses/robot_rotations.txt")
        q_cam = np.loadtxt("tests/eye_in_hand_test_poses/camera_rotations.txt")

        # Robot and camera translations
        t_robot = np.loadtxt("tests/eye_in_hand_test_poses/robot_translations.txt")
        t_cam = np.loadtxt("tests/eye_in_hand_test_poses/camera_translations.txt")

        robot_tf = []
        cam_tf = []

        num_poses = q_robot.shape[0]

        for i in range(num_poses):
            # Each quaternion in file stored as (w, x, y, z).
            # Rotation initializer expects numpy-quaternion (w, x, y, z).
            q_r = quaternion.from_float_array(
                [q_robot[i, 0], q_robot[i, 1], q_robot[i, 2], q_robot[i, 3]]
            )
            q_c = quaternion.from_float_array([q_cam[i, 0], q_cam[i, 1], q_cam[i, 2], q_cam[i, 3]])

            r_r = Rotation(q_r)
            r_c = Rotation(q_c)

            t_r = Translation(t_robot[i])
            t_c = Translation(t_cam[i])

            robot_tf.append(Transformation(t_r, r_r))
            cam_tf.append(Transformation(t_c, r_c))

        return robot_tf, cam_tf

    def test_constructor_type_check(self):
        """Constructor must reject mismatched list lengths and wrong types."""

        robot_tf, cam_tf = self._load_pose_data()

        with self.assertRaises(ValueError):
            EyeInHandCalibration(robot_tf[:-1], cam_tf)

        with self.assertRaises(TypeError):
            EyeInHandCalibration("invalid", cam_tf)

        with self.assertRaises(TypeError):
            EyeInHandCalibration(robot_tf, "invalid")

    def test_calibration_runs_and_returns_valid_transform(self):
        """Calibration should produce a valid transformation."""

        robot_tf, cam_tf = self._load_pose_data()
        calib = EyeInHandCalibration(robot_tf, cam_tf)
        result = calib.run_calibration()

        # Validate type
        self.assertIsInstance(
            result, Transformation, "Calibration result should be a Transformation."
        )

        # Validate format
        self.assertTrue(
            Transformation.is_valid(result.m, verbose=False),
            "Camera-in-EE transformation should be a valid 4x4 matrix.",
        )


if __name__ == "__main__":
    unittest.main()
