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


if __name__ == "__main__":
    unittest.main()
