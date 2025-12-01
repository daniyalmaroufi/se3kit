import unittest

import numpy as np

from se3kit.hpoint import HPoint


class TestHPoint(unittest.TestCase):
    # -------------------------------
    # Constructor tests
    # -------------------------------
    def test_constructor_xyz_coords(self):
        p = HPoint(1, 2, 3)
        self.assertTrue(np.array_equal(p.as_array(), np.array([[1], [2], [3], [1]])))

    def test_constructor_xyz_coords_decimal(self):
        p = HPoint(1.5, 2.25, 3.75)
        self.assertTrue(np.array_equal(p.as_array(), np.array([[1.5], [2.25], [3.75], [1]])))

    def test_constructor_3_array(self):
        arr = np.array([4, 5, 6])
        p = HPoint(arr)
        self.assertTrue(np.array_equal(p.as_array(), np.array([[4], [5], [6], [1]])))

    def test_constructor_3_array_decimal(self):
        arr = np.array([4.1, 5.2, 6.3])
        p = HPoint(arr)
        self.assertTrue(np.array_equal(p.as_array(), np.array([[4.1], [5.2], [6.3], [1]])))

    def test_constructor_4_array(self):
        arr = np.array([7, 8, 9, 2])
        p = HPoint(arr)
        self.assertTrue(np.array_equal(p.as_array(), np.array([[7], [8], [9], [2]])))

    def test_constructor_4_array_decimal(self):
        arr = np.array([7.7, 8.8, 9.9, 1.5])
        p = HPoint(arr)
        self.assertTrue(np.array_equal(p.as_array(), np.array([[7.7], [8.8], [9.9], [1.5]])))

    def test_constructor_invalid_array_size(self):
        arr = np.array([1, 2])
        with self.assertRaises(ValueError):
            HPoint(arr)

    def test_constructor_invalid_type(self):
        with self.assertRaises(TypeError):
            HPoint("not an array")

    def test_constructor_invalid_arg_count(self):
        with self.assertRaises(ValueError):
            HPoint(1, 2)  # too few args
        with self.assertRaises(ValueError):
            HPoint(1, 2, 3, 4)  # too many args

    # -------------------------------
    # Property tests
    # -------------------------------
    def test_x_property(self):
        p = HPoint(1, 2, 3)
        self.assertEqual(p.x, 1)
        p.x = 10
        self.assertEqual(p.x, 10)

    def test_x_property_decimal(self):
        p = HPoint(1.1, 2, 3)
        self.assertEqual(p.x, 1.1)
        p.x = 10.5
        self.assertEqual(p.x, 10.5)

    def test_y_property(self):
        p = HPoint(1, 2, 3)
        self.assertEqual(p.y, 2)
        p.y = 20
        self.assertEqual(p.y, 20)

    def test_y_property_decimal(self):
        p = HPoint(1, 2.2, 3)
        self.assertEqual(p.y, 2.2)
        p.y = 20.7
        self.assertEqual(p.y, 20.7)

    def test_z_property(self):
        p = HPoint(1, 2, 3)
        self.assertEqual(p.z, 3)
        p.z = 30
        self.assertEqual(p.z, 30)

    def test_z_property_decimal(self):
        p = HPoint(1, 2, 3.3)
        self.assertEqual(p.z, 3.3)
        p.z = 30.9
        self.assertEqual(p.z, 30.9)

    def test_xyz_property(self):
        p = HPoint(3, 4, 5)
        self.assertTrue(np.array_equal(p.xyz, np.array([3, 4, 5])))

    def test_xyz_property_decimal(self):
        p = HPoint(3.1, 4.2, 5.3)
        self.assertTrue(np.array_equal(p.xyz, np.array([3.1, 4.2, 5.3])))

    # -------------------------------
    # Method tests
    # -------------------------------
    def test_as_array(self):
        p = HPoint(9, 8, 7)
        arr = p.as_array()
        expected = np.array([[9], [8], [7], [1]])
        self.assertTrue(np.array_equal(arr, expected))
        self.assertNotEqual(id(arr), id(p.m))  # ensure it returns a copy

    def test_as_array_decimal(self):
        p = HPoint(9.9, 8.8, 7.7)
        arr = p.as_array()
        expected = np.array([[9.9], [8.8], [7.7], [1]])
        self.assertTrue(np.array_equal(arr, expected))
        self.assertNotEqual(id(arr), id(p.m))


if __name__ == "__main__":
    unittest.main()
