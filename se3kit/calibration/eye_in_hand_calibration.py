"""
Eye-in-hand calibration using quaternion AX = XB formulation.
"""

import numpy as np
import quaternion

from se3kit.rotation import Rotation
from se3kit.transformation import Transformation
from se3kit.translation import Translation
from se3kit.utils import vector_to_skew

MIN_NUMBER_OF_POSES = 2


class EyeInHandCalibration:
    """
    Hand-eye calibration class: solves T_X in  A_i * T_X = T_X * B_i
    where A_i are robot poses and B_i are calibration board in camera poses.
    """

    def __init__(self, robot_transforms, camera_transforms):
        """
        :param robot_transforms: list of se3kit Transformation objects (robot EE poses)
        :param camera_transforms: list of se3kit Transformation objects (camera poses)
        """

        if not isinstance(robot_transforms, list) or not isinstance(camera_transforms, list):
            raise TypeError("Inputs must be lists.")

        elif len(robot_transforms) != len(camera_transforms):
            raise ValueError("Robot and camera lists must be same length.")

        elif all(isinstance(pose_i, Transformation) for pose_i in robot_transforms) and all(
            isinstance(pose_i, Transformation) for pose_i in camera_transforms
        ):
            self.robot_transforms = robot_transforms
            self.camera_transforms = camera_transforms

        else:
            raise TypeError(
                f"Cannot initialize calibration from {type(robot_transforms)} and {type(camera_transforms)}"
            )

    def run_calibration(self):
        """
        Solves rotation r_x (rotation component) using quaternion SVD
        then solves translation p_x with least-squares.
        :return: camera in EE transformation
        :rtype: se3kit.transformation.Transformation
        """

        def make_m_block(q_a, q_b):
            """
            Helper function for creating blocks used for rotation solving
            """
            s1, v1 = q_a[0], q_a[1:]  # quaternion in numpy-quaternion is of shape (w,x,y,z)
            s2, v2 = q_b[0], q_b[1:]

            left = np.hstack([(s1 - s2), (v1 - v2)])
            left = left.reshape(4, 1)

            right = np.block(
                [[-(v1 - v2).reshape(1, 3)], [np.eye(3) * (s1 - s2) + vector_to_skew(v1 + v2)]]
            )

            return np.hstack([left, right])

        if len(self.robot_transforms) < MIN_NUMBER_OF_POSES:
            raise ValueError(
                f"Need at least 2 poses for calibration got {len(self.robot_transforms)}."
            )

        m_list = []

        # 1. Form all pairwise motion pairs (i,j)
        for i in range(len(self.robot_transforms) - 1):
            for j in range(i + 1, len(self.robot_transforms)):
                # Robot motions
                mat_a = self.robot_transforms[i].inv * self.robot_transforms[j]
                # Camera motions
                mat_b = self.camera_transforms[i] * self.camera_transforms[j].inv

                # Convert to quaternions
                q_a = mat_a.rotation.as_quat()
                q_b = mat_b.rotation.as_quat()

                m_list.append(make_m_block(q_a, q_b))

        # Stacking
        m = np.vstack(m_list)

        # 2. SVD for rotation
        u, s, v_trans = np.linalg.svd(m)
        q_x = v_trans[-1, :]  # last singular vector
        q_x = quaternion.quaternion(q_x / np.linalg.norm(q_x))

        r_x = Rotation(q_x).m

        # 3. Translation solve (lhs * p_x = rhs)
        lhs = []
        rhs = []

        for i in range(len(self.robot_transforms) - 1):
            for j in range(i + 1, len(self.robot_transforms)):
                mat_a = self.robot_transforms[i].inv * self.robot_transforms[j]
                mat_b = self.camera_transforms[i] * self.camera_transforms[j].inv

                r_a = mat_a.rotation.m

                t_a = mat_a.translation.m
                t_b = mat_b.translation.m

                lhs.append(r_a - np.eye(3))
                rhs.append(r_x @ t_b - t_a)

        lhs = np.vstack(lhs)
        rhs = np.vstack(rhs)

        p_x = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        # calibration result
        cam_in_ee = Transformation(Translation(p_x.flatten()), Rotation(r_x))

        self.calib_result = cam_in_ee
        return cam_in_ee
