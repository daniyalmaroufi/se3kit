"""Tests for __repr__ and __str__ methods on all core classes."""

import unittest

import numpy as np

from se3kit.hpoint import HPoint
from se3kit.rotation import Rotation
from se3kit.transformation import Transformation
from se3kit.translation import Translation


class TestTranslationReprStr(unittest.TestCase):
    """Test __repr__ and __str__ for Translation."""

    def test_repr_contains_class_name(self):
        t = Translation([1.0, 2.0, 3.0])
        self.assertIn("Translation", repr(t))

    def test_repr_contains_values(self):
        t = Translation([1.5, -2.5, 3.0])
        r = repr(t)
        self.assertIn("1.5", r)
        self.assertIn("-2.5", r)
        self.assertIn("3.0", r)

    def test_repr_roundtrip(self):
        """eval(repr(t)) should reconstruct an equivalent Translation."""
        t = Translation([0.5, -0.25, 1.0])
        t2 = eval(repr(t))  # noqa: S307
        self.assertTrue(Translation.are_close(t, t2, tol=1e-12))

    def test_str_contains_class_name(self):
        t = Translation([1.0, 2.0, 3.0])
        self.assertIn("Translation", str(t))

    def test_str_contains_xyz(self):
        t = Translation([1.0, 2.0, 3.0])
        self.assertIn("xyz", str(t))

    def test_str_fixed_precision(self):
        """__str__ should show 6 decimal places."""
        t = Translation([1.0, 0.0, 0.0])
        self.assertIn("1.000000", str(t))

    def test_zero_translation(self):
        t = Translation()
        s = str(t)
        self.assertIn("0.000000", s)


class TestRotationReprStr(unittest.TestCase):
    """Test __repr__ and __str__ for Rotation."""

    def test_repr_contains_class_name(self):
        r = Rotation()
        self.assertIn("Rotation", repr(r))

    def test_repr_contains_np_array(self):
        r = Rotation()
        self.assertIn("np.array", repr(r))

    def test_str_contains_rpy(self):
        r = Rotation()
        self.assertIn("rpy_deg", str(r))

    def test_str_identity_shows_zeros(self):
        r = Rotation()
        s = str(r)
        self.assertIn("0.000000", s)

    def test_str_90deg_rotation(self):
        """A 90-degree Z rotation should show ~90 in the yaw component."""
        r = Rotation.from_rpy([0, 0, np.pi / 2])
        s = str(r)
        self.assertIn("90.0", s)


class TestTransformationReprStr(unittest.TestCase):
    """Test __repr__ and __str__ for Transformation."""

    def test_repr_contains_class_name(self):
        T = Transformation()
        self.assertIn("Transformation", repr(T))

    def test_repr_contains_np_array(self):
        T = Transformation()
        self.assertIn("np.array", repr(T))

    def test_str_contains_xyz_and_rpy(self):
        T = Transformation(Translation([1.0, 2.0, 3.0]), Rotation())
        s = str(T)
        self.assertIn("xyz", s)
        self.assertIn("rpy_deg", s)

    def test_str_multiline(self):
        """__str__ should produce multi-line output."""
        T = Transformation(Translation([1.0, 2.0, 3.0]), Rotation())
        s = str(T)
        self.assertGreater(s.count("\n"), 0)

    def test_str_identity(self):
        T = Transformation()
        s = str(T)
        self.assertIn("0.000000", s)


class TestHPointReprStr(unittest.TestCase):
    """Test __repr__ and __str__ for HPoint (pre-existing but verify consistency)."""

    def test_repr_contains_class_name(self):
        p = HPoint(1.0, 2.0, 3.0)
        self.assertIn("HPoint", repr(p))

    def test_str_contains_coordinates(self):
        p = HPoint(1.0, 2.0, 3.0)
        s = str(p)
        self.assertIn("1.0", s)
        self.assertIn("2.0", s)
        self.assertIn("3.0", s)


if __name__ == "__main__":
    unittest.main()
