import math
import unittest

from se3kit.degrees import Degrees
from se3kit.utils import deg2rad, is_near


class TestDegrees(unittest.TestCase):
    """Unit tests for the Degrees class."""

    def test_initialization(self):
        """Test that initialization correctly stores the degree value."""
        d = Degrees(45)
        self.assertEqual(d.deg, 45)
        self.assertTrue(is_near(d.rad, deg2rad(45)))

    def test_deg_property_setter(self):
        """Test setting the degree value via .deg property."""
        d = Degrees(0)
        d.deg = 90
        self.assertEqual(d.deg, 90)
        self.assertTrue(is_near(d.rad, deg2rad(90)))

    def test_rad_property_getter(self):
        """Test getting the radian value from .rad property."""
        d = Degrees(180)
        self.assertTrue(is_near(d.rad, math.pi))

    def test_rad_property_setter(self):
        """Test setting the angle using radians."""
        d = Degrees(0)
        d.rad = math.pi / 2
        self.assertTrue(is_near(d.deg, 90))
        self.assertTrue(is_near(d.rad, math.pi / 2))

    def test_str_method(self):
        """Test the string representation of the Degrees object."""
        d = Degrees(30)
        self.assertEqual(str(d), "30")

    def test_multiple_conversions(self):
        """Test multiple conversions back and forth between deg and rad."""
        d = Degrees(0)
        for deg in [0, 45, 90, 180, 360]:
            d.deg = deg
            self.assertTrue(is_near(d.rad, deg2rad(deg)))

            # Now set via radians and check degrees
            d.rad = deg2rad(deg)
            self.assertTrue(is_near(d.deg, deg))

    def test_negative_angles(self):
        """Test that negative angles are handled correctly."""
        d = Degrees(-90)
        self.assertTrue(is_near(d.rad, deg2rad(-90)))

        d.rad = deg2rad(-45)
        self.assertTrue(is_near(d.deg, -45))


if __name__ == "__main__":
    unittest.main()
