"""
Unit tests for Rotation class.

Tests rotation matrix validation, creation from Euler angles,
and conversion methods.
"""

import unittest

import numpy as np

from se3kit import rotation
from se3kit.utils import deg2rad, is_near


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
        Compare two quaternions while accounting for the fact that q and -q
        represent the same 3D rotation.

        Returns True if q1 ≈ q2 or q1 ≈ -q2 within the given tolerance.
        """

        # Normalize both inputs to tuples (x, y, z, w)
        if hasattr(q1, "x"):
            q1 = (q1.x, q1.y, q1.z, q1.w)
        if hasattr(q2, "x"):
            q2 = (q2.x, q2.y, q2.z, q2.w)

        # Ensure lengths match
        if len(q1) != len(q2):
            raise ValueError(f"Quaternion lengths mismatch: {len(q1)} vs {len(q2)}")

        # Direct comparison using is_near
        direct = all(is_near(a, b, tol) for a, b in zip(q1, q2))

        # Comparison with negated quaternion using is_near
        negated = all(is_near(a, -b, tol) for a, b in zip(q1, q2))

        return direct or negated

    @staticmethod
    def legacy_quat(adeg, bdeg, cdeg):
        """Compute the legacy quaternion from ABC degrees."""
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

    def test_rotation_from_abc_legacy_case1(self):
        """Test ABC=(20, 30, -40)."""
        eg = (20, 30, -40)
        q_obj = rotation.Rotation.from_abc(list(eg), degrees=True).as_quat()
        qxyzw = (q_obj.x, q_obj.y, q_obj.z, q_obj.w)
        q2 = self.legacy_quat(*eg)
        self.assertTrue(
            self.quat_equal(qxyzw, q2),
            f"Quaternion mismatch for ABC={eg}, got {qxyzw}, expected {q2}",
        )

    def test_rotation_from_abc_legacy_case2(self):
        """Test ABC=(-15, 22, 10)."""
        eg = (-15, 22, 10)
        q_obj = rotation.Rotation.from_abc(list(eg), degrees=True).as_quat()
        qxyzw = (q_obj.x, q_obj.y, q_obj.z, q_obj.w)
        q2 = self.legacy_quat(*eg)
        self.assertTrue(
            self.quat_equal(qxyzw, q2),
            f"Quaternion mismatch for ABC={eg}, got {qxyzw}, expected {q2}",
        )

    def test_rotation_from_abc_legacy_case3(self):
        """Test ABC=(0, 190, -600)."""
        eg = (0, 190, -600)
        q_obj = rotation.Rotation.from_abc(list(eg), degrees=True).as_quat()
        qxyzw = (q_obj.x, q_obj.y, q_obj.z, q_obj.w)
        q2 = self.legacy_quat(*eg)
        self.assertTrue(
            self.quat_equal(qxyzw, q2),
            f"Quaternion mismatch for ABC={eg}, got {qxyzw}, expected {q2}",
        )

    # -------------------------------
    # Rotation ABC / ZYX tests
    # -------------------------------
    def _check_abc_zyx_consistency(self, eg, tol=1e-10):
        """Check consistency between ABC and ZYX rotation representations using is_near."""

        def arrays_near(a, b, tol):
            if a.shape != b.shape:
                return False
            return all(is_near(x, y, tol) for x, y in zip(a.flat, b.flat))

        r_abc = rotation.Rotation.from_abc(eg)
        r_zyx = rotation.Rotation.from_zyx(eg)

        self.assertTrue(arrays_near(r_abc.as_abc() - eg, np.zeros_like(eg), tol))
        self.assertTrue(arrays_near(r_abc.as_zyx() - eg, np.zeros_like(eg), tol))
        self.assertTrue(arrays_near(r_zyx.as_zyx() - eg, np.zeros_like(eg), tol))
        self.assertTrue(arrays_near(r_zyx.as_abc() - eg, np.zeros_like(eg), tol))
        self.assertTrue(arrays_near(r_zyx.m, r_abc.m, tol))

    def test_rotation_abc_zyx_case1(self):
        self._check_abc_zyx_consistency([1, 0, 0])

    def test_rotation_abc_zyx_case2(self):
        self._check_abc_zyx_consistency([0, 1, 0])

    def test_rotation_abc_zyx_case3(self):
        self._check_abc_zyx_consistency([0, 0, 1])

    def test_rotation_abc_zyx_case4(self):
        self._check_abc_zyx_consistency([np.pi / 2, -np.pi / 4, 0])

    # -------------------------------
    # Rotation RPY / ZYX reversed tests
    # -------------------------------
    def _check_rpy_zyx_consistency(self, eg, tol=1e-10):
        """Check consistency between RPY and ZYX rotation representations using is_near."""

        def arrays_near(a, b, tol):
            a = np.asarray(a)
            b = np.asarray(b)
            if a.shape != b.shape:
                return False
            return all(is_near(x, y, tol) for x, y in zip(a.flat, b.flat))

        rzyx = rotation.Rotation.from_zyx(eg)
        rrpy = rotation.Rotation.from_rpy(np.flip(eg))

        # Compare rotation matrices
        self.assertTrue(arrays_near(rzyx.m, rrpy.m, tol))

        # Compare RPY angles (with flipping)
        rpy = rotation.Rotation.from_zyx(eg).as_rpy()
        self.assertTrue(arrays_near(np.flip(rpy), eg, tol))

    def test_rotation_rpy_zyx_case1(self):
        self._check_rpy_zyx_consistency([1, 0, 0])

    def test_rotation_rpy_zyx_case2(self):
        self._check_rpy_zyx_consistency([0, 1, 0])

    def test_rotation_rpy_zyx_case3(self):
        self._check_rpy_zyx_consistency([0, 0, 1])

    def test_rotation_rpy_zyx_case4(self):
        self._check_rpy_zyx_consistency([np.pi / 2, -np.pi / 4, 0])

    def test_custom_rotation_matrix(self):
        """
        Test a specific custom rotation matrix from 3D Rotation Converter.
        3D Rotation Converter: https://www.andre-gaschler.com/rotationconverter/
        """
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
            all(is_near(a, b, tol=1e-6) for a, b in zip(r.m.flat, r2.m.flat)),
            f"Round-trip ZYX Euler → matrix mismatch:\n"
            f"Expected (original):\n{r.m}\n"
            f"Got (reconstructed):\n{r2.m}",
        )

        # Round-trip via ABC Euler angles
        abc = r.as_abc()
        r3 = rotation.Rotation.from_abc(abc)
        self.assertTrue(
            all(is_near(a, b, tol=1e-6) for a, b in zip(r.m.flat, r3.m.flat)),
            f"Round-trip ABC Euler → matrix mismatch:\n"
            f"Expected (original):\n{r.m}\n"
            f"Got (reconstructed):\n{r3.m}",
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
        self.assertTrue(
            self.quat_equal(q, q2),
            f"Quaternion roundtrip mismatch:\n"
            f"Expected (original): {q}\n"
            f"Got (round-trip):     {q2}",
        )

    def test_custom_axis_angle_roundtrip(self):
        """Test round-trip: axis-angle → matrix → axis-angle."""
        # Axis-angle from converter: axis normalized, angle in degrees
        axis = np.array([0.4472136, 0, -0.8944272])
        angle_deg = 58.4118645
        angle_rad = deg2rad(angle_deg)

        # Create rotation
        r = rotation.Rotation.from_axisangle(axis, angle_rad)

        # Convert back to axis-angle
        axis2, angle2 = r.as_axisangle()

        # Axis can flip direction, adjust if needed
        if not all(is_near(a, b, tol=1e-7) for a, b in zip(axis, axis2)):
            axis2 = -axis2
            angle2 = -angle2

        # Check axis
        self.assertTrue(
            all(is_near(a, b, tol=1e-7) for a, b in zip(axis, axis2)),
            f"Axis mismatch: {axis} vs {axis2}",
        )

        # Check angle
        self.assertTrue(
            is_near(angle_rad, angle2, tol=1e-7), f"Angle mismatch: {angle_rad} vs {angle2}"
        )

    def test_invalid_non_orthogonal_matrix(self):
        """Rotation constructor should raise ValueError for non-orthogonal matrix."""
        r = np.eye(3)
        r[0, 1] = 0.1  # breaks orthogonality
        with self.assertRaises(ValueError):
            rotation.Rotation(r)

    def test_invalid_improper_rotation_det_minus_one(self):
        """Rotation constructor should raise ValueError for improper rotation (determinant=-1)."""
        r = np.eye(3)
        r[2, 2] = -1  # flips determinant to -1
        with self.assertRaises(ValueError):
            rotation.Rotation(r)

    def test_invalid_wrong_shape_matrix(self):
        """Rotation constructor should raise ValueError for wrong shape matrix."""
        r = np.eye(4)  # 4x4 matrix
        with self.assertRaises(ValueError):
            rotation.Rotation(r)

    def test_invalid_wrong_type_matrix(self):
        """Rotation constructor should raise TypeError for non-ndarray input."""
        r = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # list instead of np.ndarray
        with self.assertRaises(TypeError):
            rotation.Rotation(r)

    def test_invalid_almost_orthogonal_matrix(self):
        """Matrix nearly orthogonal but violates tolerance should raise ValueError."""
        r = np.eye(3)
        r[0, 0] = 1 + 1e-3  # violates orthogonality within default tol
        with self.assertRaises(ValueError):
            rotation.Rotation(r)


if __name__ == "__main__":
    unittest.main()
