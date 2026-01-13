"""
Unit tests for Translation class.

Tests translation vector validation, arithmetic operations,
and unit conversion methods.
"""

import unittest

import numpy as np

from se3kit import translation


class TestTranslation(unittest.TestCase):
    """Tests for the Translation class."""

    def test_translation_vector_valid(self):
        """Test that a proper translation vector is recognized as valid."""
        vec = np.asarray([1, 2, 3])
        self.assertTrue(
            translation.Translation.is_valid(vec, verbose=False),
            "Expected vec to be a valid translation vector",
        )

    def test_translation_vector_invalid_size(self):
        """Test that an incorrectly sized translation vector is recognized as invalid."""
        vec_bad = np.asarray([[1], [2], [3.0], [3]])
        self.assertFalse(
            translation.Translation.is_valid(vec_bad, verbose=False),
            "Expected vec_bad to be invalid (size != 3)",
        )

    def test_translation_scaling(self):
        """Test scaling of translation vectors."""
        # 1. Test out-of-place scaling
        t1 = translation.Translation([1, 2, 3])
        t2 = t1.scale(2.0, inplace=False)

        self.assertTrue(
            np.all(np.isclose(t1.m, [1, 2, 3])), "Original should not change for inplace=False"
        )
        self.assertTrue(np.all(np.isclose(t2.m, [2, 4, 6])), "Result should be scaled")
        self.assertIsNot(t1, t2, "Should return a new object")

        # 2. Test in-place scaling
        t3 = translation.Translation([1, 2, 3])
        res = t3.scale(3.0, inplace=True)

        self.assertIsNone(res, "inplace=True should return None")
        self.assertTrue(np.all(np.isclose(t3.m, [3, 6, 9])), "Original should be scaled")


if __name__ == "__main__":
    unittest.main()
