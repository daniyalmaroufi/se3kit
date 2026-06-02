"""Tests for serialization methods (to_dict, from_dict, to_json, from_json, to_csv, from_csv)."""

import json
import os
import tempfile
import unittest

import numpy as np

from se3kit.hpoint import HPoint
from se3kit.rotation import Rotation
from se3kit.transformation import Transformation
from se3kit.translation import Translation


class TestTranslationSerialization(unittest.TestCase):
    """Test serialization for Translation."""

    def test_to_dict_keys(self):
        t = Translation([1.0, 2.0, 3.0])
        d = t.to_dict()
        self.assertIn("type", d)
        self.assertIn("xyz", d)
        self.assertEqual(d["type"], "Translation")

    def test_to_dict_native_types(self):
        """Verify dict values are native Python floats, not numpy."""
        t = Translation([1.0, 2.0, 3.0])
        d = t.to_dict()
        for v in d["xyz"]:
            self.assertIsInstance(v, float)

    def test_dict_roundtrip(self):
        t = Translation([0.5, -0.25, 1.0])
        t2 = Translation.from_dict(t.to_dict())
        self.assertTrue(Translation.are_close(t, t2, tol=1e-12))

    def test_from_dict_missing_key(self):
        with self.assertRaises(ValueError):
            Translation.from_dict({"bad": [1, 2, 3]})

    def test_json_roundtrip(self):
        t = Translation([0.5, -0.25, 1.0])
        t2 = Translation.from_json(t.to_json())
        self.assertTrue(Translation.are_close(t, t2, tol=1e-12))

    def test_json_is_valid(self):
        t = Translation([1.0, 2.0, 3.0])
        j = t.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["type"], "Translation")

    def test_csv_roundtrip(self):
        t = Translation([0.5, -0.25, 1.0])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            t.to_csv(path)
            t2 = Translation.from_csv(path)
            self.assertTrue(Translation.are_close(t, t2, tol=1e-12))
        finally:
            os.unlink(path)

    def test_csv_no_header(self):
        t = Translation([1.0, 2.0, 3.0])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            t.to_csv(path, header=False)
            t2 = Translation.from_csv(path)
            self.assertTrue(Translation.are_close(t, t2, tol=1e-12))
        finally:
            os.unlink(path)

    def test_csv_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Translation.from_csv("/nonexistent/path/file.csv")

    def test_zero_translation_roundtrip(self):
        t = Translation()
        t2 = Translation.from_dict(t.to_dict())
        self.assertTrue(Translation.are_close(t, t2, tol=1e-12))


class TestRotationSerialization(unittest.TestCase):
    """Test serialization for Rotation."""

    def test_to_dict_keys(self):
        r = Rotation.from_rpy([0.1, 0.2, 0.3])
        d = r.to_dict()
        self.assertIn("type", d)
        self.assertIn("quaternion_xyzw", d)
        self.assertIn("rpy_rad", d)
        self.assertIn("matrix", d)
        self.assertEqual(d["type"], "Rotation")

    def test_to_dict_native_types(self):
        """Verify all values are native Python types, not numpy."""
        r = Rotation.from_rpy([0.1, 0.2, 0.3])
        d = r.to_dict()
        for v in d["quaternion_xyzw"]:
            self.assertIsInstance(v, float)
        for v in d["rpy_rad"]:
            self.assertIsInstance(v, float)
        # Matrix rows
        for row in d["matrix"]:
            for v in row:
                self.assertIsInstance(v, float)

    def test_dict_roundtrip_via_matrix(self):
        r = Rotation.from_rpy([0.1, 0.2, 0.3])
        r2 = Rotation.from_dict(r.to_dict())
        self.assertTrue(Rotation.are_close(r, r2, tol=1e-10))

    def test_dict_roundtrip_via_quaternion(self):
        r = Rotation.from_rpy([0.1, 0.2, 0.3])
        d = r.to_dict()
        del d["matrix"]
        r2 = Rotation.from_dict(d)
        self.assertTrue(Rotation.are_close(r, r2, tol=1e-10))

    def test_dict_roundtrip_via_rpy(self):
        r = Rotation.from_rpy([0.1, 0.2, 0.3])
        d = r.to_dict()
        del d["matrix"]
        del d["quaternion_xyzw"]
        r2 = Rotation.from_dict(d)
        self.assertTrue(Rotation.are_close(r, r2, tol=1e-6))

    def test_from_dict_missing_keys(self):
        with self.assertRaises(ValueError):
            Rotation.from_dict({"type": "Rotation"})

    def test_json_roundtrip(self):
        r = Rotation.from_rpy([0.1, 0.2, 0.3])
        r2 = Rotation.from_json(r.to_json())
        self.assertTrue(Rotation.are_close(r, r2, tol=1e-10))

    def test_identity_roundtrip(self):
        r = Rotation()
        r2 = Rotation.from_dict(r.to_dict())
        self.assertTrue(r2.is_identity())

    def test_csv_roundtrip_quaternion(self):
        r = Rotation.from_rpy([0.1, 0.2, 0.3])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            r.to_csv(path, fmt="quaternion")
            r2 = Rotation.from_csv(path, fmt="quaternion")
            self.assertTrue(Rotation.are_close(r, r2, tol=1e-10))
        finally:
            os.unlink(path)

    def test_csv_roundtrip_rpy_rad(self):
        r = Rotation.from_rpy([0.1, 0.2, 0.3])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            r.to_csv(path, fmt="rpy_rad")
            r2 = Rotation.from_csv(path, fmt="rpy_rad")
            self.assertTrue(Rotation.are_close(r, r2, tol=1e-6))
        finally:
            os.unlink(path)

    def test_csv_roundtrip_rpy_deg(self):
        r = Rotation.from_rpy([0.1, 0.2, 0.3])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            r.to_csv(path, fmt="rpy_deg")
            r2 = Rotation.from_csv(path, fmt="rpy_deg")
            self.assertTrue(Rotation.are_close(r, r2, tol=1e-4))
        finally:
            os.unlink(path)

    def test_csv_roundtrip_matrix(self):
        r = Rotation.from_rpy([0.1, 0.2, 0.3])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            r.to_csv(path, fmt="matrix")
            r2 = Rotation.from_csv(path, fmt="matrix")
            self.assertTrue(Rotation.are_close(r, r2, tol=1e-10))
        finally:
            os.unlink(path)


class TestTransformationSerialization(unittest.TestCase):
    """Test serialization for Transformation."""

    def test_to_dict_keys(self):
        T = Transformation(Translation([1.0, 2.0, 3.0]), Rotation())
        d = T.to_dict()
        self.assertIn("type", d)
        self.assertIn("translation", d)
        self.assertIn("rotation", d)
        self.assertIn("matrix", d)
        self.assertEqual(d["type"], "Transformation")

    def test_dict_roundtrip_via_matrix(self):
        T = Transformation(Translation([1.0, 2.0, 3.0]), Rotation.from_rpy([0.1, 0.2, 0.3]))
        T2 = Transformation.from_dict(T.to_dict())
        self.assertTrue(Transformation.are_close(T, T2))

    def test_dict_roundtrip_via_components(self):
        T = Transformation(Translation([1.0, 2.0, 3.0]), Rotation.from_rpy([0.1, 0.2, 0.3]))
        d = T.to_dict()
        del d["matrix"]
        T2 = Transformation.from_dict(d)
        self.assertTrue(Transformation.are_close(T, T2))

    def test_from_dict_missing_keys(self):
        with self.assertRaises(ValueError):
            Transformation.from_dict({"type": "Transformation"})

    def test_json_roundtrip(self):
        T = Transformation(Translation([1.0, 2.0, 3.0]), Rotation.from_rpy([0.1, 0.2, 0.3]))
        T2 = Transformation.from_json(T.to_json())
        self.assertTrue(Transformation.are_close(T, T2))

    def test_json_is_valid(self):
        T = Transformation(Translation([1.0, 2.0, 3.0]), Rotation())
        j = T.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["type"], "Transformation")

    def test_csv_roundtrip_quaternion(self):
        T = Transformation(Translation([1.0, 2.0, 3.0]), Rotation.from_rpy([0.1, 0.2, 0.3]))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            T.to_csv(path, rotation_format="quaternion")
            T2 = Transformation.from_csv(path, rotation_format="quaternion")
            self.assertTrue(Transformation.are_close(T, T2))
        finally:
            os.unlink(path)

    def test_csv_roundtrip_rpy(self):
        T = Transformation(Translation([1.0, 2.0, 3.0]), Rotation.from_rpy([0.1, 0.2, 0.3]))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            T.to_csv(path, rotation_format="rpy_rad")
            T2 = Transformation.from_csv(path, rotation_format="rpy_rad")
            self.assertTrue(Transformation.are_close(T, T2, rot_tol=1e-4))
        finally:
            os.unlink(path)

    def test_identity_roundtrip(self):
        T = Transformation()
        T2 = Transformation.from_dict(T.to_dict())
        self.assertTrue(Transformation.are_close(T, T2))


class TestHPointSerialization(unittest.TestCase):
    """Test serialization for HPoint."""

    def test_to_dict_keys(self):
        p = HPoint(1.0, 2.0, 3.0)
        d = p.to_dict()
        self.assertIn("type", d)
        self.assertIn("xyz", d)
        self.assertEqual(d["type"], "HPoint")

    def test_to_dict_native_types(self):
        p = HPoint(1.0, 2.0, 3.0)
        d = p.to_dict()
        for v in d["xyz"]:
            self.assertIsInstance(v, float)

    def test_dict_roundtrip(self):
        p = HPoint(1.5, -2.5, 3.0)
        p2 = HPoint.from_dict(p.to_dict())
        np.testing.assert_allclose(p.xyz, p2.xyz, atol=1e-12)

    def test_from_dict_missing_key(self):
        with self.assertRaises(ValueError):
            HPoint.from_dict({"bad": [1, 2, 3]})

    def test_json_roundtrip(self):
        p = HPoint(1.5, -2.5, 3.0)
        p2 = HPoint.from_json(p.to_json())
        np.testing.assert_allclose(p.xyz, p2.xyz, atol=1e-12)

    def test_json_is_valid(self):
        p = HPoint(1.0, 2.0, 3.0)
        j = p.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["type"], "HPoint")


if __name__ == "__main__":
    unittest.main()
