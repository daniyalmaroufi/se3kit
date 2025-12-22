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

    def __init__(self, robot_transforms=None, camera_transforms=None):
        """
        :param robot_transforms: list of se3kit Transformation objects (robot EE poses)
        :param camera_transforms: list of se3kit Transformation objects (camera poses)
        """

        self.robot_transforms = []
        self.camera_transforms = []

        if robot_transforms is not None and camera_transforms is not None:
            self.add_pairs(robot_transforms, camera_transforms)
        elif robot_transforms is not None or camera_transforms is not None:
            raise ValueError(
                "Both robot_transforms and camera_transforms must be provided together."
            )

    def add_pair(self, robot_transform, camera_transform):
        """
        Adds a single pair of robot and camera transforms.

        :param robot_transform: Transformation object (robot EE pose)
        :param camera_transform: Transformation object (camera pose)
        """
        if not isinstance(robot_transform, Transformation) or not isinstance(
            camera_transform, Transformation
        ):
            return False

        self.robot_transforms.append(robot_transform)
        self.camera_transforms.append(camera_transform)

        return True

    def add_pairs(self, robot_transforms, camera_transforms):
        """
        Adds pairs of robot and camera transforms.

        :param robot_transforms: single Transformation or list of Transformations (robot EE poses)
        :param camera_transforms: single Transformation or list of Transformations (camera poses)
        """
        if isinstance(robot_transforms, Transformation) and isinstance(
            camera_transforms, Transformation
        ):
            self.add_pair(robot_transforms, camera_transforms)
        elif isinstance(robot_transforms, list) and isinstance(camera_transforms, list):
            if len(robot_transforms) != len(camera_transforms):
                raise ValueError("Robot and camera lists must be same length.")

            if len(robot_transforms) == 0:
                raise ValueError("Cannot add empty transform lists.")

            for i, (rt, ct) in enumerate(zip(robot_transforms, camera_transforms)):
                success = self.add_pair(rt, ct)
                if not success:
                    raise TypeError(
                        f"Both inputs must be Transformation objects. Got {type(rt)} and {type(ct)} at index {i}."
                    )
        else:
            raise TypeError("Inputs must be Transformations or lists of Transformations.")

    def remove_pairs(self, indices):
        """
        Removes pairs at the specified indices.

        :param indices: list of indices to remove
        :type indices: list
        """
        if not isinstance(indices, list):
            raise TypeError("Indices must be a list.")

        for index in indices:
            if not isinstance(index, int) or not (0 <= index < self.num_pairs):
                raise ValueError(
                    f"Index {index} is out of range. Valid indices are 0 to {self.num_pairs - 1}."
                )

        self.robot_transforms = [
            pose for i, pose in enumerate(self.robot_transforms) if i not in indices
        ]
        self.camera_transforms = [
            pose for i, pose in enumerate(self.camera_transforms) if i not in indices
        ]

    def reset_pairs(self):
        """
        Resets the lists of robot and camera transforms.
        """
        self.robot_transforms = []
        self.camera_transforms = []

    @property
    def num_pairs(self):
        """
        Returns the number of transform pairs.

        :return: number of pairs
        :rtype: int
        """
        return len(self.robot_transforms)

    def run_calibration(self):
        """
        Solves rotation r_x (rotation component) using quaternion SVD
        then solves translation p_x with least-squares.
        :return: camera in EE transformation
        :rtype: se3kit.transformation.Transformation
        """

        def _make_m_block(q_a, q_b):
            """
            Helper function for creating blocks used for rotation solving
            """
            q_arr_a = quaternion.as_float_array(q_a)  # (w, x, y, z)
            q_arr_b = quaternion.as_float_array(q_b)

            s1, v1 = q_arr_a[0], q_arr_a[1:]
            s2, v2 = q_arr_b[0], q_arr_b[1:]

            left = np.hstack([(s1 - s2), (v1 - v2)])
            left = left.reshape(4, 1)

            right = np.block(
                [[-(v1 - v2).reshape(1, 3)], [np.eye(3) * (s1 - s2) + vector_to_skew(v1 + v2)]]
            )

            return np.hstack([left, right])

        if len(self.robot_transforms) < MIN_NUMBER_OF_POSES:
            raise ValueError(
                f"Need at least 2 poses for calibration, got {len(self.robot_transforms)}."
            )

        def _compute_motion_pairs(robot_transforms, camera_transforms):
            """Compute all pairwise robot and camera motions."""
            motion_pairs = []
            for i in range(len(robot_transforms) - 1):
                for j in range(i + 1, len(robot_transforms)):
                    # Robot motions
                    mat_a = robot_transforms[i].inv * robot_transforms[j]

                    # Camera motions
                    mat_b = camera_transforms[i] * camera_transforms[j].inv

                    motion_pairs.append((mat_a, mat_b))

            return motion_pairs

        motion_pairs = _compute_motion_pairs(self.robot_transforms, self.camera_transforms)

        # 1. Form all pairwise motion pairs (i,j)
        m_list = [
            _make_m_block(robot_motion.rotation.as_quat(), camera_motion.rotation.as_quat())
            for robot_motion, camera_motion in motion_pairs
        ]

        # Stacking
        m = np.vstack(m_list)

        # 2. SVD for rotation
        _, _, v_trans = np.linalg.svd(m)
        q_x = v_trans[-1, :]  # last singular vector
        q_x = quaternion.from_float_array(q_x / np.linalg.norm(q_x))

        r_x = Rotation(q_x).m

        # 3. Translation solve (lhs * p_x = rhs)
        lhs = [(robot_motion.rotation.m - np.eye(3)) for robot_motion, _ in motion_pairs]
        rhs = [
            (r_x @ (camera_motion.translation.m - robot_motion.translation.m).T).reshape(3, 1)
            for robot_motion, camera_motion in motion_pairs
        ]

        lhs = np.vstack(lhs)
        rhs = np.vstack(rhs)

        p_x = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        # calibration result
        cam_in_ee = Transformation(Translation(p_x.flatten()), Rotation(r_x))

        self.calib_result = cam_in_ee
        return cam_in_ee
