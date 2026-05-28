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

    def test_flexible_initialization(self):
        """Test flexible initialization (swapped arguments and kwargs)."""
        t = transformation.Translation([1, 2, 3])
        r = transformation.Rotation.from_rpy([0, 0, np.pi / 2])

        # Test standard order
        t_1 = transformation.Transformation(t, r)

        # Test swapped order
        t_2 = transformation.Transformation(r, t)

        # Test single rotation
        t_3 = transformation.Transformation(r)

        self.assertTrue(transformation.Transformation.are_close(t_1, t_2))
        self.assertTrue(np.all(utils.is_near(t_3.translation.xyz, [0, 0, 0])))
        self.assertTrue(transformation.Rotation.are_close(t_3.rotation, r))

    def test_transformation_scaling(self):
        """Test scaling of the translation part of a transformation."""
        t = transformation.Translation([1, 2, 3])
        r = transformation.Rotation.from_rpy([0, 0, np.pi / 2])
        tf1 = transformation.Transformation(t, r)

        # 1. Test out-of-place scaling
        # This was buggy before user's fix
        tf2 = tf1.scale(2.0, inplace=False)

        self.assertTrue(
            np.all(utils.is_near(tf1.translation.xyz, [1, 2, 3])),
            "Original should keep translation",
        )
        self.assertTrue(
            np.all(utils.is_near(tf2.translation.xyz, [2, 4, 6])),
            "Result should have scaled translation",
        )
        self.assertTrue(
            transformation.Rotation.are_close(tf2.rotation, r), "Rotation should be unchanged"
        )
        self.assertIsNot(tf1, tf2, "Should return new object")

        # 2. Test in-place scaling
        res = tf1.scale(3.0, inplace=True)
        self.assertIsNone(res, "inplace=True should return None")
        self.assertTrue(
            np.all(utils.is_near(tf1.translation.xyz, [3, 6, 9])), "Original should be scaled"
        )
        self.assertTrue(
            transformation.Rotation.are_close(tf1.rotation, r), "Rotation should be unchanged"
        )

    def test_pose_stamped_initialization(self):
        """Test Transformation initialization from PoseStamped message (TF2)."""
        from se3kit.ros_compat import ROS_VERSION, use_geomsg

        if not use_geomsg or ROS_VERSION == 0:
            self.skipTest("ROS not available")

        try:
            from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
        except ImportError:
            self.skipTest("geometry_msgs not available")

        # Create a PoseStamped message
        pose = Pose(
            position=Point(x=1.0, y=2.0, z=3.0),
            orientation=Quaternion(x=0, y=0, z=0.7071068, w=0.7071068),  # 90 deg around Z
        )
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header.frame_id = "base_link"

        # Initialize Transformation from PoseStamped
        t = transformation.Transformation(pose_stamped)

        # Verify translation
        self.assertTrue(np.all(utils.is_near(t.translation.xyz, [1.0, 2.0, 3.0])))

        # Verify rotation (should be 90 deg around Z)
        rpy = t.rotation.as_rpy()
        self.assertTrue(np.all(utils.is_near(rpy, [0, 0, np.pi / 2], tol=1e-5)))

    def test_transform_initialization(self):
        """Test Transformation initialization from Transform message (TF2)."""
        from se3kit.ros_compat import ROS_VERSION, use_geomsg

        if not use_geomsg or ROS_VERSION == 0:
            self.skipTest("ROS not available")

        try:
            from geometry_msgs.msg import Transform, Vector3, Quaternion
        except ImportError:
            self.skipTest("geometry_msgs not available")

        # Create a Transform message
        transform = Transform(
            translation=Vector3(x=1.0, y=2.0, z=3.0),
            rotation=Quaternion(x=0, y=0, z=0.7071068, w=0.7071068),  # 90 deg around Z
        )

        # Initialize Transformation from Transform
        t = transformation.Transformation(transform)

        # Verify translation
        self.assertTrue(np.all(utils.is_near(t.translation.xyz, [1.0, 2.0, 3.0])))

        # Verify rotation (should be 90 deg around Z)
        rpy = t.rotation.as_rpy()
        self.assertTrue(np.all(utils.is_near(rpy, [0, 0, np.pi / 2], tol=1e-5)))

    def test_as_pose_stamped(self):
        """Test conversion of Transformation to PoseStamped message."""
        from se3kit.ros_compat import ROS_VERSION, use_geomsg

        if not use_geomsg or ROS_VERSION == 0:
            self.skipTest("ROS not available")

        try:
            from geometry_msgs.msg import PoseStamped
        except ImportError:
            self.skipTest("geometry_msgs not available")

        # Create a Transformation
        t = transformation.Transformation(
            transformation.Translation([1.0, 2.0, 3.0]),
            transformation.Rotation.from_rpy([0, 0, np.pi / 2]),
        )

        # Convert to PoseStamped
        pose_stamped = t.as_pose_stamped(frame_id="base_link")

        # Verify result type
        self.assertIsInstance(pose_stamped, PoseStamped)

        # Verify header
        self.assertEqual(pose_stamped.header.frame_id, "base_link")

        # Verify translation
        self.assertTrue(
            np.all(utils.is_near([pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z], [1.0, 2.0, 3.0]))
        )

        # Verify rotation
        q = pose_stamped.pose.orientation
        rotation = transformation.Rotation([q.x, q.y, q.z, q.w])
        rpy = rotation.as_rpy()
        self.assertTrue(np.all(utils.is_near(rpy, [0, 0, np.pi / 2], tol=1e-5)))

    def test_as_transform(self):
        """Test conversion of Transformation to Transform message."""
        from se3kit.ros_compat import ROS_VERSION, use_geomsg

        if not use_geomsg or ROS_VERSION == 0:
            self.skipTest("ROS not available")

        try:
            from geometry_msgs.msg import Transform
        except ImportError:
            self.skipTest("geometry_msgs not available")

        # Create a Transformation
        t = transformation.Transformation(
            transformation.Translation([1.0, 2.0, 3.0]),
            transformation.Rotation.from_rpy([0, 0, np.pi / 2]),
        )

        # Convert to Transform
        tf_msg = t.as_transform()

        # Verify result type
        self.assertIsInstance(tf_msg, Transform)

        # Verify translation
        self.assertTrue(
            np.all(utils.is_near([tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z], [1.0, 2.0, 3.0]))
        )

        # Verify rotation
        q = tf_msg.rotation
        rotation = transformation.Rotation([q.x, q.y, q.z, q.w])
        rpy = rotation.as_rpy()
        self.assertTrue(np.all(utils.is_near(rpy, [0, 0, np.pi / 2], tol=1e-5)))

    def test_tf2_roundtrip_pose_stamped(self):
        """Test round-trip conversion: Transformation → PoseStamped → Transformation."""
        from se3kit.ros_compat import ROS_VERSION, use_geomsg

        if not use_geomsg or ROS_VERSION == 0:
            self.skipTest("ROS not available")

        # Create original Transformation
        t_original = transformation.Transformation(
            transformation.Translation([1.5, 2.5, 3.5]),
            transformation.Rotation.from_rpy([0.1, 0.2, 0.3]),
        )

        # Convert to PoseStamped and back
        pose_stamped = t_original.as_pose_stamped()
        t_recovered = transformation.Transformation(pose_stamped)

        # Verify they match
        self.assertTrue(transformation.Transformation.are_close(t_original, t_recovered, rot_tol=1e-5, trans_tol=1e-5))

    def test_tf2_roundtrip_transform(self):
        """Test round-trip conversion: Transformation → Transform → Transformation."""
        from se3kit.ros_compat import ROS_VERSION, use_geomsg

        if not use_geomsg or ROS_VERSION == 0:
            self.skipTest("ROS not available")

        # Create original Transformation
        t_original = transformation.Transformation(
            transformation.Translation([1.5, 2.5, 3.5]),
            transformation.Rotation.from_rpy([0.1, 0.2, 0.3]),
        )

        # Convert to Transform and back
        tf_msg = t_original.as_transform()
        t_recovered = transformation.Transformation(tf_msg)

        # Verify they match
        self.assertTrue(transformation.Transformation.are_close(t_original, t_recovered, rot_tol=1e-5, trans_tol=1e-5))

        # Verify they match
        self.assertTrue(transformation.Transformation.are_close(t_original, t_recovered, rot_tol=1e-5, trans_tol=1e-5))


if __name__ == "__main__":
    unittest.main()
