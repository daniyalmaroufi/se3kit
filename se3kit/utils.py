from math import pi

import numpy as np

# Global Constants
NUMERICAL_TOLERANCE = 1e-14  # Default numerical tolerance

# Sizes used in vector helpers
_VECTOR3_SIZE = 3
_SCREW_SIZE = 6


def deg2rad(d):
    """
    Converts degrees to radians

    :param d: Angle value in degrees
    :return: The angle in radians
    :return_type: float | np.ndarray
    """
    if isinstance(d, (np.ndarray, list, tuple)):
        return np.deg2rad(np.array(d))
    return d / 180 * pi


def rad2deg(r):
    """
    Converts radians to degrees
    :param r: Angle value in radians
    :return: The angle in degrees
    :return_type: float | np.ndarray
    """
    if isinstance(r, (np.ndarray, list, tuple)):
        return np.rad2deg(np.array(r))
    return r / pi * 180


def is_near(a, b, tol=NUMERICAL_TOLERANCE):
    """
    Checks if two scalar values are approximately equal within a tolerance.

    :param a: First scalar value.
    :type a: float
    :param b: Second scalar value.
    :type b: float
    :param tol: Tolerance within which the values are considered equal.
    :type tol: float, optional (default=1e-14)
    :return: True if ``|a - b|`` < tol, False otherwise.
    :rtype: bool
    """
    return abs(a - b) < tol


def is_identity(a, tol=NUMERICAL_TOLERANCE):
    """
    Checks if a square matrix is approximately the identity matrix.

    :param a: Square matrix to check.
    :type a: np.ndarray
    :param tol: Tolerance for element-wise comparison to identity.
    :type tol: float, optional (default=1e-14)
    :return: True if a ≈ I, False otherwise.
    :rtype: bool
    """
    n = a.shape[0]
    identity = np.eye(n)

    # Flatten both matrices and compare element-wise using is_near
    return all(is_near(x, y, tol) for x, y in zip(a.flat, identity.flat))


def vector_to_skew(v):
    """
    Converts a vector into a skew-symmetric matrix for cross product or screw computations.

    :param v: Input vector of size 3 (3D vector) or 6 (screw vector).
    :type v: np.ndarray
    :return: Skew-symmetric matrix representing the vector.
    :rtype: numpy.ndarray

    :raises ValueError: If the input vector is not size 3 or 6.
    """
    v = np.squeeze(v)
    if v.size == _VECTOR3_SIZE:
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    elif v.size == _SCREW_SIZE:
        return np.array(
            [[0, -v[2], v[1], v[3]], [v[2], 0, -v[0], v[4]], [-v[1], v[0], 0, v[5]], [0, 0, 0, 0]]
        )
    else:
        raise ValueError(f"Not implemented for input size {v.size}")


def skew_to_vector(sk, tol=NUMERICAL_TOLERANCE):
    """
    Converts a 3x3 skew-symmetric matrix back into a 3D vector.

    :param sk: 3x3 skew-symmetric matrix.
    :type sk: np.ndarray
    :return: Corresponding 3D vector [x, y, z].
    :rtype: numpy.ndarray

    :raises ValueError: If input matrix is not 3x3 or not skew-symmetric.
    """
    sk = np.asarray(sk)

    if sk.shape != (3, 3):
        raise ValueError(f"Not implemented for shape {sk.shape}")

    # Check skew-symmetry: S + S.T should be near zero
    for i in range(3):
        for j in range(3):
            if not is_near(sk[i, j], -sk[j, i], tol):
                raise ValueError("Matrix is not skew-symmetric")

    return np.array([sk[2, 1], sk[0, 2], sk[1, 0]])
