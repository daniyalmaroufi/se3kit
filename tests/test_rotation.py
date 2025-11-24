"""
Unit tests for Rotation class.

Tests rotation matrix validation, creation from Euler angles,
and conversion methods.
"""

import unittest

import numpy as np

from se3kit import rotation


class TestRotation(unittest.TestCase):
    """Tests for the Rotation class."""

    def test_rotation_matrix_validity(self):
        """Test validation of rotation matrices."""
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

    def test_rotation_matrix_valid(self):
        """Test validation of a valid rotation matrix."""
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

    def test_rotation_matrix_invalid(self):
        """Test validation of an invalid rotation matrix (determinant not ~1)."""
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

    def test_rotation_from_zyx(self):
        """Test creation of rotation from ZYX Euler angles."""
        angles = [0, 0, 0]
        self.assertTrue(
            rotation.Rotation.from_rpy(angles).is_identity(),
            "Rotation from zero RPY should be identity",
        )


if __name__ == "__main__":
    unittest.main()
