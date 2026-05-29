"""
ROS compatibility layer for SE3Kit modules.

This module provides lightweight imports and utilities for
handling ROS1 vs ROS2 message type differences.
"""

import importlib.util as _importlib_util

use_geomsg = False
Pose = Point = Quaternion = Vector3 = None
PoseStamped = Transform = None
ROS_VERSION = 0

# Probe ROS2 first, then ROS1 using importlib to avoid importing unused modules
if _importlib_util.find_spec("rclpy") is not None:
    try:
        from geometry_msgs.msg import (  # type: ignore
            Point,
            Pose,
            PoseStamped,
            Quaternion,
            Transform,
            Vector3,
        )

        ROS_VERSION = 2
        use_geomsg = True
    except Exception:
        ROS_VERSION = 0
elif _importlib_util.find_spec("rospy") is not None:
    try:
        from geometry_msgs.msg import (  # type: ignore
            Point,
            Pose,
            PoseStamped,
            Quaternion,
            Transform,
            Vector3,
        )

        ROS_VERSION = 1
        use_geomsg = True
    except Exception:
        ROS_VERSION = 0
else:
    Pose = Point = Quaternion = Vector3 = None
    PoseStamped = Transform = None
    ROS_VERSION = 0


def get_ros_geometry_msgs():
    """
    Returns the basic geometry message types (Point, Quaternion, Pose, Vector3)
    depending on which ROS environment is active.

    For TF2 message types, use get_ros_tf2_msgs() instead.

    :return: Tuple of message classes.
    :rtype: (Point, Quaternion, Pose, Vector3)
    """
    return Point, Quaternion, Pose, Vector3


def get_ros_tf2_msgs():
    """
    Returns the TF2-specific message types (PoseStamped, Transform)
    depending on which ROS environment is active.

    :return: Tuple of TF2 message classes.
    :rtype: (PoseStamped, Transform)
    """
    return PoseStamped, Transform
