"""
Unit tests for SE3Kit modules.

This module defines the Tests class for verifying functionality
of Robot, Rotation, and Transformation classes under both ROS1 and ROS2
environments. It can be executed directly to run tests.
"""

import unittest

import numpy as np

from se3kit import rotation, transformation, translation

# Import ROS compatibility layer and core SE3Kit modules
from se3kit.ros_compat import ROS_VERSION
from se3kit.robot import Robot
from se3kit.utils import deg2rad

TOLERANCE = 1e-8


class Tests(unittest.TestCase):
    """Collection of unit tests for verifying SE3Kit robot kinematics and transformations."""

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

    def test_rotation_matrix_validity(self):
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

        # Invalid rotation matrix (determinant not ~1)
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

    def test_translation_vector_validity(self):
        vec = np.asarray([1, 2, 3])
        self.assertTrue(
            translation.Translation.is_valid(vec, verbose=False),
            "Expected vec to be a valid translation vector",
        )

        vec_bad = np.asarray([[1], [2], [3.0], [3]])
        self.assertFalse(
            translation.Translation.is_valid(vec_bad, verbose=False),
            "Expected vec_bad to be invalid (size != 3)",
        )

    def test_transformation_validity(self):
        # 3x3 input -> invalid transformation (expects 4x4)
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

        # 3x4 input -> invalid
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

        # Proper 4x4 homogeneous transformation
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

    def test_rotation_from_zyx(self):
        angles = [0, 0, 0]
        self.assertTrue(
            rotation.Rotation.from_rpy(angles).is_identity(),
            "Rotation from zero RPY should be identity",
        )


    def test_FK_space_iiwa(self):
        r = Robot.create_iiwa()

        ja_deg = [-0.01, -35.10, 47.58, 24.17, 0.00, 0.00, 0.00]
        ja_rad = np.deg2rad(ja_deg)

        t = r.fk_space(ja_rad)

        self.assertTrue(
            np.allclose(t.rotation.as_zyx(degrees=True),
                        [68.28584782, -43.53993415, -35.84364870], atol=TOLERANCE)
        )
        self.assertTrue(
            np.allclose(t.rotation.as_rpy(degrees=True),
                        [-35.844, -43.540, 68.286], atol=TOLERANCE)
        )
        self.assertTrue(
            np.allclose(t.translation.m,
                        [-636.32792290, -158.87809407, 1012.70702237], atol=TOLERANCE)
        )

    # def test_rotation_from_ABC_legacy(self):
    #     """Compare Rotation.from_ABC_degrees to legacy quaternion formula."""

    #     def legacy_quat(adeg, bdeg, cdeg):
    #         Ar, Br, Cr = (deg2rad(d) for d in (adeg, bdeg, cdeg))
    #         x = np.cos(Ar/2)*np.cos(Br/2)*np.sin(Cr/2) - np.sin(Ar/2)*np.sin(Br/2)*np.cos(Cr/2)
    #         y = np.cos(Ar/2)*np.sin(Br/2)*np.cos(Cr/2) + np.sin(Ar/2)*np.cos(Br/2)*np.sin(Cr/2)
    #         z = np.sin(Ar/2)*np.cos(Br/2)*np.cos(Cr/2) - np.cos(Ar/2)*np.sin(Br/2)*np.sin(Cr/2)
    #         w = np.cos(Ar/2)*np.cos(Br/2)*np.cos(Cr/2) + np.sin(Ar/2)*np.sin(Br/2)*np.sin(Cr/2)
    #         return (x, y, z, w)

    #     for eg in [(20, 30, -40), (-15, 22, 10), (0, 190, -600)]:
    #         q = rotation.Rotation.from_ABC_degrees(list(eg)).as_quat()
    #         qxyzw = (q.x, q.y, q.z, q.w)
    #         q2 = legacy_quat(*eg)
    #         self.assertTrue(np.allclose(qxyzw, q2, atol=1e-12))

    def quat_equal(self, q1, q2, tol=1e-12):
        """
        Compare two quaternions, accounting for sign ambiguity.
        q1, q2: sequences of 4 floats (x, y, z, w)
        """
        q1 = np.array(q1)
        q2 = np.array(q2)
        return np.allclose(q1, q2, atol=tol) or np.allclose(q1, -q2, atol=tol)


    def test_rotation_from_ABC_legacy(self):
        """Compare Rotation.from_ABC_degrees to a manually computed quaternion."""

        def legacy_quat(adeg, bdeg, cdeg):
            Ar, Br, Cr = (deg2rad(d) for d in (adeg, bdeg, cdeg))
            x = np.cos(Ar/2)*np.cos(Br/2)*np.sin(Cr/2) - np.sin(Ar/2)*np.sin(Br/2)*np.cos(Cr/2)
            y = np.cos(Ar/2)*np.sin(Br/2)*np.cos(Cr/2) + np.sin(Ar/2)*np.cos(Br/2)*np.sin(Cr/2)
            z = np.sin(Ar/2)*np.cos(Br/2)*np.cos(Cr/2) - np.cos(Ar/2)*np.sin(Br/2)*np.sin(Cr/2)
            w = np.cos(Ar/2)*np.cos(Br/2)*np.cos(Cr/2) + np.sin(Ar/2)*np.sin(Br/2)*np.sin(Cr/2)
            return (x, y, z, w)

        test_angles = [
            (20, 30, -40),
            (-15, 22, 10),
            (0, 190, -600)
        ]

        for eg in test_angles:
            q_obj = rotation.Rotation.from_ABC_degrees(list(eg)).as_quat()
            # Convert np.quaternion to numeric tuple (x, y, z, w)
            qxyzw = (q_obj.x, q_obj.y, q_obj.z, q_obj.w)
            q2 = legacy_quat(*eg)
            self.assertTrue(
                self.quat_equal(qxyzw, q2),
                f"Quaternion mismatch for ABC={eg}, got {qxyzw}, expected {q2}"
        )


    def test_rotation_zyx_is_ABC(self):
        egs = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [np.pi/2, -np.pi/4, 0],
        ]

        for eg in egs:
            self.assertTrue(
                np.allclose(
                    rotation.Rotation.from_ABC(eg).as_ABC() - eg, 0, atol=1e-10
                )
            )
            self.assertTrue(
                np.allclose(
                    rotation.Rotation.from_ABC(eg).as_zyx() - eg, 0, atol=1e-10
                )
            )
            self.assertTrue(
                np.allclose(
                    rotation.Rotation.from_zyx(eg).as_zyx() - eg, 0, atol=1e-10
                )
            )
            self.assertTrue(
                np.allclose(
                    rotation.Rotation.from_zyx(eg).as_ABC() - eg, 0, atol=1e-10
                )
            )

            # rotation matrices equal
            self.assertTrue(
                np.allclose(
                    rotation.Rotation.from_zyx(eg).m,
                    rotation.Rotation.from_ABC(eg).m,
                    atol=1e-10,
                )
            )

    def test_rotation_rpy_is_zyx_reversed(self):
        egs = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [np.pi/2, -np.pi/4, 0],
        ]

        for eg in egs:
            Rzyx = rotation.Rotation.from_zyx(eg)
            Rrpy = rotation.Rotation.from_rpy(np.flip(eg))
            self.assertTrue(np.allclose(Rzyx.m, Rrpy.m, atol=1e-10))

        for eg in egs:
            rpy = rotation.Rotation.from_zyx(eg).as_rpy()
            self.assertTrue(np.allclose(np.flip(rpy), eg, atol=1e-10))


# --- Script execution entry point ---
if __name__ == "__main__":
    # Let unittest discover and run all tests
    unittest.main()
