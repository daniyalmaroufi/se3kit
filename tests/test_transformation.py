"""
Unit tests for Transformation class.

Tests 4x4 homogeneous transformation matrix validation,
composition, and conversion methods.
"""

import unittest

import numpy as np

from se3kit import hpoint, transformation, utils


class TestTransformation(unittest.TestCase):
    """Tests for the Transformation class."""

    def test_invalid_transformation_3x3(self):
        """3x3 input -> invalid transformation (expects 4x4)."""
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

    def test_invalid_transformation_3x4(self):
        """3x4 input -> invalid transformation (expects 4x4)."""
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

    def test_valid_transformation_4x4(self):
        """Proper 4x4 homogeneous transformation -> valid."""
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

    def test_transformation_multiplication(self):
        """Test multiplication of transformations."""
        t1 = transformation.Transformation(
            transformation.Translation([1, 2, 3]),
            transformation.Rotation.from_rpy([0, 0, np.pi / 2]),
        )
        t2 = transformation.Transformation(
            transformation.Translation([0.5, 0, 0]),
            transformation.Rotation.from_rpy([0, 0, np.pi / 2]),
        )
        t_combined = t1 * t2
        self.assertTrue(np.all(utils.is_near(t_combined.translation.xyz, [1, 2.5, 3])))
        self.assertTrue(np.all(utils.is_near(t_combined.rotation.as_rpy(), [0, 0, np.pi])))

    def test_transformation_multiplication_with_hpoint(self):
        """Test multiplication of transformations with HPoint."""
        t = transformation.Transformation(
            transformation.Translation([1, 2, 3]),
            transformation.Rotation.from_rpy([0, 0, np.pi / 2]),
        )
        p = hpoint.HPoint([1, 2, 3])
        p_transformed = t * p
        self.assertTrue(np.all(utils.is_near(p_transformed.xyz, [-1, 3, 6])))


if __name__ == "__main__":
    unittest.main()
