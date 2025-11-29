"""
Unit tests for Rotation class.

Tests rotation matrix validation, creation from Euler angles,
and conversion methods.
"""

import unittest

import numpy as np

from se3kit import rotation
from se3kit.utils import deg2rad


class TestRotation(unittest.TestCase):
    """Tests for the Rotation class."""

    def test_rotation_matrix_validity(self):
        """Test validation of rotation matrices."""
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

    def test_rotation_matrix_valid(self):
        """Test validation of a valid rotation matrix."""
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

    def test_rotation_matrix_invalid(self):
        """Test validation of an invalid rotation matrix (determinant not ~1)."""
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

    def test_rotation_from_zyx(self):
        """Test creation of rotation from ZYX Euler angles."""
        angles = [0, 0, 0]
        self.assertTrue(
            rotation.Rotation.from_rpy(angles).is_identity(),
            "Rotation from zero RPY should be identity",
        )

    def quat_equal(self, q1, q2, tol=1e-7):
        """
        Compare two quaternions (np.quaternion or tuple) up to sign.
        Returns True if they are equal up to ±1 (q or -q).
        """
        # Convert np.quaternion to tuple (x, y, z, w)
        if hasattr(q1, "x"):
            q1 = (q1.x, q1.y, q1.z, q1.w)
        if hasattr(q2, "x"):
            q2 = (q2.x, q2.y, q2.z, q2.w)

        # Direct comparison
        direct = all(abs(a - b) < tol for a, b in zip(q1, q2, strict=True))
        # Compare with negated quaternion
        negated = all(abs(a + b) < tol for a, b in zip(q1, q2, strict=True))
        return direct or negated

    def test_rotation_from_abc_legacy(self):
        """Compare Rotation.from_ABC_degrees to a manually computed quaternion."""

        def legacy_quat(adeg, bdeg, cdeg):
            ar, br, cr = (deg2rad(d) for d in (adeg, bdeg, cdeg))
            x = np.cos(ar / 2) * np.cos(br / 2) * np.sin(cr / 2) - np.sin(ar / 2) * np.sin(
                br / 2
            ) * np.cos(cr / 2)
            y = np.cos(ar / 2) * np.sin(br / 2) * np.cos(cr / 2) + np.sin(ar / 2) * np.cos(
                br / 2
            ) * np.sin(cr / 2)
            z = np.sin(ar / 2) * np.cos(br / 2) * np.cos(cr / 2) - np.cos(ar / 2) * np.sin(
                br / 2
            ) * np.sin(cr / 2)
            w = np.cos(ar / 2) * np.cos(br / 2) * np.cos(cr / 2) + np.sin(ar / 2) * np.sin(
                br / 2
            ) * np.sin(cr / 2)
            return (x, y, z, w)

        test_angles = [(20, 30, -40), (-15, 22, 10), (0, 190, -600)]

        for eg in test_angles:
            q_obj = rotation.Rotation.from_ABC_degrees(list(eg)).as_quat()
            # Convert np.quaternion to numeric tuple (x, y, z, w)
            qxyzw = (q_obj.x, q_obj.y, q_obj.z, q_obj.w)
            q2 = legacy_quat(*eg)
            self.assertTrue(
                self.quat_equal(qxyzw, q2),
                f"Quaternion mismatch for ABC={eg}, got {qxyzw}, expected {q2}",
            )

    def test_rotation_zyx_is_abc(self):
        egs = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [np.pi / 2, -np.pi / 4, 0],
        ]

        for eg in egs:
            self.assertTrue(
                np.allclose(rotation.Rotation.from_ABC(eg).as_ABC() - eg, 0, atol=1e-10)
            )
            self.assertTrue(
                np.allclose(rotation.Rotation.from_ABC(eg).as_zyx() - eg, 0, atol=1e-10)
            )
            self.assertTrue(
                np.allclose(rotation.Rotation.from_zyx(eg).as_zyx() - eg, 0, atol=1e-10)
            )
            self.assertTrue(
                np.allclose(rotation.Rotation.from_zyx(eg).as_ABC() - eg, 0, atol=1e-10)
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
            [np.pi / 2, -np.pi / 4, 0],
        ]

        for eg in egs:
            rzyx = rotation.Rotation.from_zyx(eg)
            rrpy = rotation.Rotation.from_rpy(np.flip(eg))
            self.assertTrue(np.allclose(rzyx.m, rrpy.m, atol=1e-10))

        for eg in egs:
            rpy = rotation.Rotation.from_zyx(eg).as_rpy()
            self.assertTrue(np.allclose(np.flip(rpy), eg, atol=1e-10))

    def test_custom_rotation_matrix(self):
        """Test a specific custom rotation matrix from 3D Rotation Converter."""
        # Rotation matrix from your input
        r_mat = np.array(
            [
                [0.6190476, 0.7619048, -0.1904762],
                [-0.7619048, 0.5238096, -0.3809524],
                [-0.1904762, 0.3809524, 0.9047619],
            ]
        )

        # Create Rotation object
        r = rotation.Rotation(r_mat)

        # Check matrix validity
        self.assertTrue(
            rotation.Rotation.is_valid(r.m, verbose=False), "Rotation matrix should be valid"
        )

        # Round-trip via ZYX Euler angles
        zyx = r.as_zyx()
        r2 = rotation.Rotation.from_zyx(zyx)

        # The reconstructed matrix should be close to original
        self.assertTrue(
            np.allclose(r.m, r2.m, atol=1e-7),
            f"Round-trip ZYX Euler → matrix failed: {r.m} vs {r2.m}",
        )

        # Optionally: round-trip via ABC Euler (if available)
        abc = r.as_ABC()
        r3 = rotation.Rotation.from_ABC(abc)
        self.assertTrue(
            np.allclose(r.m, r3.m, atol=1e-7),
            f"Round-trip ABC Euler → matrix failed: {r.m} vs {r3.m}",
        )

    def test_custom_quaternion_roundtrip(self):
        """Test round-trip: quaternion → matrix → quaternion."""
        # Quaternion from your converter
        q = (0.2182179, 0.0, -0.4364358, 0.8728716)  # (x, y, z, w)

        # Create rotation from quaternion
        r = rotation.Rotation.from_quat(q)

        # Convert back to quaternion
        q2_obj = r.as_quat()
        q2 = (q2_obj.x, q2_obj.y, q2_obj.z, q2_obj.w)

        # Check if original and round-trip quaternion match (up to sign)
        self.assertTrue(self.quat_equal(q, q2), f"Quaternion roundtrip failed: {q} vs {q2}")

    def test_custom_axis_angle_roundtrip(self):
        """Test round-trip: axis-angle → matrix → axis-angle."""
        # Axis-angle from converter: axis normalized, angle in degrees
        axis = np.array([0.4472136, 0, -0.8944272])
        angle_deg = 58.4118645
        angle_rad = np.deg2rad(angle_deg)

        # Create rotation
        r = rotation.Rotation.from_axis_angle(axis, angle_rad)

        # Convert back to axis-angle
        axis2, angle2 = r.as_axisangle()

        # Axis can flip direction, adjust if needed
        if not np.allclose(axis, axis2, atol=1e-7):
            axis2 = -axis2
            angle2 = -angle2

        self.assertTrue(np.allclose(axis, axis2, atol=1e-7), f"Axis mismatch: {axis} vs {axis2}")
        self.assertAlmostEqual(
            angle_rad, angle2, delta=1e-7, msg=f"Angle mismatch: {angle_rad} vs {angle2}"
        )


if __name__ == "__main__":
    unittest.main()
