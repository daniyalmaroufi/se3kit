"""
pivot_calibration.py

Defines a "pivot calibration" class for pivot calibration.

"""


import logging

import numpy as np

from se3kit.ros_compat import get_ros_geometry_msgs
from se3kit.rotation import Rotation
from se3kit.transformation import Transformation

# Retrieve the ROS geometry message types (Point, Quaternion, Pose, Vector3)
Point, Quaternion, Pose, Vector3 = get_ros_geometry_msgs()
use_geomsg = Quaternion is not None

# module logger
logger = logging.getLogger(__name__)

TOLERANCE = 1e-14
MIN_NUMBER_OF_POSES = 2


class PivotCalibration:
    """
    Represents a pivot calibration method.
    """

    def __init__(self, init_value=None):
        """
        Initializes calibration from a list of ROS poses or list of se3kit.transformation.Transformation poses.

        :param init_value: list of ROS poses or se3kit.transformation.Transformation poses
        :type init_value: list
        """
        if init_value is None:
            # Case 1: No input provided
            raise TypeError("Cannot initialize calibration without input data.")

        elif len(init_value) == 0:
            # Case 2: Input is an empty list
            raise TypeError("Cannot initialize calibration from empty list.")

        elif not isinstance(init_value, list):
            # Case 3: Input is not a list
            raise TypeError("Input is not a list.")

        elif all(isinstance(pose_i, Transformation) for pose_i in init_value):
            # Case 4: Input is a list of se3kit.transformation.Transformations
            self.calib_poses = init_value

        elif all(isinstance(pose_i, Pose) for pose_i in init_value):
            # Case 5: Input is a list of ROS poses
            # Convert poses to type Transformation first
            self.calib_poses = [Transformation(pose_i) for pose_i in init_value]

        else:
            # Case 6: Input type is not supported
            raise TypeError(f"Cannot initialize calibration from {type(init_value)}")

    def run_pivot_calibration(self):
        """
        Runs the pivot calibration optimization and returns the tip location with respect to (wrt) the body frame,
        the divot location wrt the world frame, and the calibration residual.

        :return: tuple of tip wrt body frame (tuple of 3 floats), divot wrt world frame (tuple of 3 floats), calibration residual (ndarray of floats)
        :rtype: (tuple, tuple, np.ndarray)
        """

        num_of_poses = len(self.calib_poses)

        if num_of_poses < MIN_NUMBER_OF_POSES:
            raise ValueError(
                f"Cannot run pivot calibration with less than 2 poses. Number of poses provided: {num_of_poses}"
            )

        mat_a = np.empty((num_of_poses * 3, 6))
        mat_b = np.empty((num_of_poses * 3, 1))
        for ii, pose in enumerate(self.calib_poses):
            c = 3 * ii
            mat_b[c : c + 3, 0] = -pose.translation.m
            mat_a[c : c + 3, 0:3] = pose.rotation.m
            mat_a[c : c + 3, 3:6] = -np.eye(3)

        x = np.linalg.lstsq(mat_a, mat_b, rcond=None)[0]

        resid = np.matmul(mat_a, x) - mat_b
        resid_dist = np.sqrt(np.sum(np.square(resid.reshape((3, -1))), axis=0))
        tip = (x[0, 0], x[1, 0], x[2, 0])
        divot = (x[3, 0], x[4, 0], x[5, 0])
        self.calibration_result = (tip, divot, resid_dist)
        return (tip, divot, resid_dist)

    @staticmethod
    def find_similar_poses(poses):
        """
        Finds similar poses in 'poses' by comparing their rotation part.

        :param poses: a list of se3kit.transformation.Transformation objects
        :type poses: list
        :return: list of repeated poses indices
        :rtype: list
        """

        clean_poses = []
        idx_repeated_poses = []
        similarity_flag = False
        for ii, pose_i in enumerate(poses):
            for pose_c in clean_poses:
                if Rotation.are_close(pose_i.rotation, pose_c.rotation):
                    similarity_flag = True
                    break
            if similarity_flag:
                idx_repeated_poses.append(ii)
                similarity_flag = False
            else:
                clean_poses.append(pose_i)

        return idx_repeated_poses

    def remove_poses(self, idx_poses):
        """
        Removes poses with specific indices from the poses list.

        :param idx_poses: a list of indices to be removed
        :type idx_poses: list
        """
        self.calib_poses = [pose for i, pose in enumerate(self.calib_poses) if i not in idx_poses]

    def add_poses(self, poses):
        """
        Adds new poses to calib_poses.

        :param poses: a list of Transformation poses or ROS poses
        :type poses: list
        """
        if not isinstance(poses, list):
            raise TypeError("Input is not a list.")

        if len(poses) == 0:
            raise TypeError("Cannot add poses from empty list.")

        elif all(isinstance(pose_i, Transformation) for pose_i in poses):
            self.calib_poses.extend(poses)

        elif all(isinstance(pose_i, Pose) for pose_i in poses):
            self.calib_poses.extend([Transformation(pose_i) for pose_i in poses])
        else:
            raise TypeError(f"Cannot add poses from {type(poses)}")

    def reset_poses(self):
        """
        Resets the current list of calib_poses.
        """
        self.calib_poses = []
