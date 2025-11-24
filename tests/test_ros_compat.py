"""
Unit tests for ROS compatibility layer.

Tests ROS version detection and environment setup.
"""

import unittest

from se3kit.ros_compat import ROS_VERSION


class TestRosCompat(unittest.TestCase):
    """Tests for ROS compatibility and version detection."""

    def test_ros_version(self):
        """
        Validate ROS environment detection.

        This test ensures that the ROS compatibility layer correctly identifies
        the active ROS version. It confirms that `ROS_VERSION` is one of the
        supported values: 0 (no ROS), 1 (ROS1), or 2 (ROS2).

        Raises:
            AssertionError: If the detected ROS version is not one of the expected values.
        """
        self.assertIn(ROS_VERSION, [0, 1, 2], "Invalid ROS version detected.")


if __name__ == "__main__":
    unittest.main()
