from math import pi
import unittest

import numpy as np

# Import from your library
from se3kit.utils import (
    deg2rad,
    is_identity,
    is_near,
    rad2deg,
    skew_to_vector,
    vector_to_skew,
)


class TestUtils(unittest.TestCase):
    # -----------------------------
    # deg2rad / rad2deg
    # -----------------------------
    def test_deg2rad_scalar(self):
        self.assertTrue(is_near(deg2rad(180), pi))
        self.assertTrue(is_near(deg2rad(90), pi / 2))

    def test_deg2rad_array(self):
        arr = np.array([0, 90, 180])
        expected = np.array([0, pi / 2, pi])
        self.assertTrue(all(is_near(a, b) for a, b in zip(deg2rad(arr), expected)))

    def test_rad2deg_scalar(self):
        self.assertTrue(is_near(rad2deg(pi), 180))
        self.assertTrue(is_near(rad2deg(pi / 2), 90))

    def test_rad2deg_array(self):
        arr = np.array([0, pi / 2, pi])
        expected = np.array([0, 90, 180])
        self.assertTrue(all(is_near(a, b) for a, b in zip(rad2deg(arr), expected)))


    def test_deg2rad_rad2deg_roundtrip(self):
        x = np.array([-180, -90, 0, 45, 123])
        
        self.assertTrue(all(is_near(a, b, tol=1e-12) for a, b in zip(rad2deg(deg2rad(x)), x)))



    # -----------------------------
    # is_near
    # -----------------------------
    def test_is_near_true(self):
        self.assertTrue(is_near(1.0, 1.0 + 1e-15))

    def test_is_near_false(self):
        self.assertFalse(is_near(1.0, 1.0 + 1e-10))

    def test_is_near_custom_tol(self):
        self.assertTrue(is_near(1.0, 1.05, tol=0.1))
        self.assertFalse(is_near(1.0, 1.2, tol=0.1))

    # -----------------------------
    # is_identity
    # -----------------------------
    def test_is_identity_exact(self):
        i = np.eye(3)
        self.assertTrue(is_identity(i))

    def test_is_identity_small_perturbations(self):
        """Matrix with tiny perturbations should still be considered identity."""
        i = np.eye(4)

        # Add tiny noise to multiple entries (still below tolerance)
        i[0, 1] = 1e-15
        i[1, 2] = 2e-15
        i[2, 0] = -3e-15
        i[3, 1] = 5e-16

        self.assertTrue(
            is_identity(i), msg=f"Matrix with small perturbations should still be identity: {i}"
        )

    def test_is_identity_false(self):
        m = np.eye(3)
        m[0, 1] = 1e-5  # Too large
        self.assertFalse(is_identity(m))

    # -----------------------------
    # vector_to_skew (3D)
    # -----------------------------
    def test_vector_to_skew_3d(self):
        v = np.array([1, 2, 3])
        sk = vector_to_skew(v)

        expected = np.array(
            [
                [0, -3, 2],
                [3, 0, -1],
                [-2, 1, 0],
            ]
        )

        self.assertTrue(all(is_near(a, b) for a, b in zip(sk.flat, expected.flat)))

    # -----------------------------
    # vector_to_skew (6D screw)
    # -----------------------------
    def test_vector_to_skew_6d(self):
        v = np.array([1, 2, 3, 4, 5, 6])
        sk = vector_to_skew(v)

        expected = np.array(
            [
                [0, -3, 2, 4],
                [3, 0, -1, 5],
                [-2, 1, 0, 6],
                [0, 0, 0, 0],
            ]
        )

        self.assertTrue(all(is_near(a, b) for a, b in zip(sk.flat, expected.flat)))

    def test_vector_to_skew_invalid_size(self):
        v = np.array([1, 2])  # Invalid size
        with self.assertRaises(ValueError):
            vector_to_skew(v)

    # -----------------------------
    # skew_to_vector
    # -----------------------------
    def test_skew_to_vector_valid(self):
        sk = np.array(
            [
                [0, -3, 2],
                [3, 0, -1],
                [-2, 1, 0],
            ]
        )
        expected = np.array([1, 2, 3])
        self.assertTrue(all(is_near(a, b) for a, b in zip(skew_to_vector(sk), expected)))

    def test_skew_to_vector_invalid_symmetric(self):
        """A symmetric matrix should be rejected as invalid skew-symmetric."""
        symmetric = np.array(
            [
                [0, 1, 2],
                [1, 0, 3],
                [2, 3, 0],
            ]
        )

        with self.assertRaises(ValueError):
            skew_to_vector(symmetric)

    def test_skew_to_vector_invalid_shape(self):
        with self.assertRaises(ValueError):
            skew_to_vector(np.zeros((4, 4)))

    # -----------------------------
    # round-trip: v -> skew -> v
    # -----------------------------
    def test_round_trip_3d(self):
        v = np.array([4.2, -1.1, 9.5])
        self.assertTrue(all(is_near(a, b) for a, b in zip(skew_to_vector(vector_to_skew(v)), v)))


    # Also test list input works
    def test_round_trip_list_input(self):
        v = [3, 5, 7]
        self.assertTrue(all(is_near(a, b) for a, b in zip(skew_to_vector(vector_to_skew(v)), np.array(v))))



if __name__ == "__main__":
    unittest.main()
