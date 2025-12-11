"""
p2p_registration.py

Defines a class for correspondence-based point cloud point cloud registration.

"""

import logging

import numpy as np

from se3kit.ros_compat import get_ros_geometry_msgs
from se3kit.transformation import Transformation

# Retrieve the ROS geometry message types (Point, Quaternion, Pose, Vector3)
Point, Quaternion, Pose, Vector3 = get_ros_geometry_msgs()
use_geomsg = Quaternion is not None

# module logger
logger = logging.getLogger(__name__)

MIN_NUMBER_OF_POINTS = 3
PCD_ARRAY_DIMENSIONS = 2


class P2PRegistration:
    """
    Represents a correspondence-based point cloud point cloud registration.
    """

    def __init__(self, pcd_1, pcd_2):
        """
        Initializes registration from two numpy arrays with 2 or 3 columns.

        :param pcd_1: first point cloud (Nx3 or Nx2 numpy array)
        :type pcd_1: numpy.ndarray
        :param pcd_2: second point cloud (Nx3 or Nx2 numpy array)
        :type pcd_2: numpy.ndarray
        """

        if pcd_1 is None:
            # Case 1: No input provided
            raise TypeError("Cannot initialize registration without input data.")

        elif not isinstance(pcd_1, np.ndarray) or not isinstance(pcd_2, np.ndarray):
            # Case 2: Input is not a np.ndarray
            raise TypeError("Input is not a numpy array.")

        elif pcd_1.ndim != PCD_ARRAY_DIMENSIONS or pcd_2.ndim != PCD_ARRAY_DIMENSIONS:
            # Case 3: Input is not 2D
            raise TypeError("Input point clouds must be 2D numpy arrays.")

        elif pcd_1.shape != pcd_2.shape:
            # Case 4: Input shapes do not match
            raise ValueError("Input point clouds must have the same shape.")

        elif np.isfinite(pcd_1).all() and np.isfinite(pcd_2).all():
            # Case 5: Input does not contain any NaN values
            self.pcd_1 = pcd_1
            self.pcd_2 = pcd_2

        else:
            # Case 6: Input type is not supported
            raise ValueError("Input point clouds must not contain NaN values.")

    @staticmethod
    def estimate_rigid_transform(pcd_1, pcd_2):
        """
        Estimate the rigid transformation (rotation + translation)
        that aligns points A to points B using SVD.
        Assumes A and B are Nx3 and correspond one-to-one.
        """

        centroid_1 = np.mean(pcd_1, axis=0)
        centroid_2 = np.mean(pcd_2, axis=0)

        pcd_1_centered = pcd_1 - centroid_1
        pcd_2_centered = pcd_2 - centroid_2

        u, _, v_t = np.linalg.svd(pcd_1_centered.T @ pcd_2_centered)
        rotation = v_t.T @ u.T

        # Handle reflection (det(R) = -1)
        if np.linalg.det(rotation) < 0:
            v_t[2, :] *= -1
            rotation = v_t.T @ u.T

        translation = centroid_2 - rotation @ centroid_1

        mat = np.eye(4)
        mat[:3, :3] = rotation
        mat[:3, 3] = translation

        return Transformation(matrix=mat)

    def run_registration(self):
        """
        Runs the point-to-point registration and returns the estimated transformation.

        :return: Estimated transformation aligning pcd_1 to pcd_2
        :rtype: se3kit.transformation.Transformation
        """

        num_of_points = self.pcd_1.shape[0]

        if num_of_points < MIN_NUMBER_OF_POINTS:
            raise ValueError(
                f"Cannot run point-to-point registration with less than {MIN_NUMBER_OF_POINTS} points. Number of points provided: {num_of_points}"
            )

        transformation = P2PRegistration.estimate_rigid_transform(self.pcd_1, self.pcd_2)

        return transformation
