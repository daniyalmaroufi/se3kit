"""
Unit tests for P2PRegistration class.

Uses test pose data from two files:
- pcd_1.txt
- pcd_2.txt
to build point clouds and verifies that registration runs and returns a valid result.
"""

import unittest

import numpy as np

from se3kit.calibration.p2p_registration import P2PRegistration
from se3kit.transformation import Transformation


class Test(unittest.TestCase):
    """Tests for the P2PRegistration class."""

    @staticmethod
    def _load_point_cloud_data():
        """Loads test point clouds and converts entries into numpy arrays."""

        # Point clouds
        pcd_1 = np.loadtxt("tests/p2p_test_data/pcd_1_data.txt")
        pcd_2 = np.loadtxt("tests/p2p_test_data/pcd_2_data.txt")

        # True transformation
        t_true = np.loadtxt("tests/p2p_test_data/ground_truth_transformation.txt")

        return pcd_1, pcd_2, t_true

    def test_constructor_type_check(self):
        """Constructor must reject mismatched list lengths and wrong types."""

        pcd_1, pcd_2, t_true = self._load_point_cloud_data()

        with self.assertRaises(ValueError):
            P2PRegistration(pcd_1=pcd_1, pcd_2=None)

        with self.assertRaises(ValueError):
            P2PRegistration(pcd_1=pcd_1, pcd_2=pcd_2[:-1, :])

        with self.assertRaises(TypeError):
            P2PRegistration(pcd_1=pcd_1, pcd_2=pcd_2.tolist())

    def test_calibration_runs_and_returns_valid_transform(self):
        """Calibration should produce a valid transformation."""

        pcd_1, pcd_2, t_true = self._load_point_cloud_data()
        p2p_registrator = P2PRegistration(pcd_1=pcd_1, pcd_2=pcd_2)
        result = p2p_registrator.run_registration()

        # Validate type
        self.assertIsInstance(
            result, Transformation, "Registration result should be a Transformation."
        )

        # Validate format
        self.assertTrue(
            Transformation.is_valid(result.m, verbose=False),
            "Resulted transformation should be a valid 4x4 matrix.",
        )

        self.assertTrue(
            Transformation.are_close(result, Transformation(t_true)),
            "Resulted transformation should be close to the true transformation.",
        )


if __name__ == "__main__":
    unittest.main()
