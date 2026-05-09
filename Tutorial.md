# Mastering 3D Rigid-Body Transformations with SE3kit

Welcome to the comprehensive guide on **SE3kit**! 

`se3kit` is a lightweight Python library tailored for 3D rigid-body transformations, rotations, and spatial mathematics. If you are working in robotics, computer vision, augmented reality, or 3D simulation, you constantly deal with moving objects in 3D space. Mathematically, these movements belong to the **SE(3)** (Special Euclidean) group. 

This tutorial will guide you through the core concepts and show you how to apply them to real-world robotics problems, from tracking objects to calculating kinematics and calibrating sensors.

---

## Installation

`se3kit` is designed to be lean, relying only on standard scientific libraries (`numpy` and `numpy-quaternion`). You can install it quickly using pip:

```bash
pip install se3kit
```

---

## Building Blocks of 3D Geometry

Before building complex robotics applications, we need to understand the four mathematical pillars of `se3kit`.

### Translating Points in Space
A translation represents a pure linear displacement in 3D space along the X, Y, and Z axes.

```python
import se3kit as se3

# Represent moving 1 meter forward (X), 2 meters left (Y), and 0.5 meters up (Z)
t1 = se3.Translation([1.0, 2.0, 0.5])

# Access individual axes
print(f"X: {t1.x}, Y: {t1.y}, Z: {t1.z}")

# Get the magnitude of the vector (Euclidean distance from origin)
distance = t1.norm()
print(f"Distance from origin: {distance:.2f} meters")

# Scale translations (e.g., halving the movement)
t_half = t1.scale(0.5)
print(f"Scaled Translation: {t_half.xyz}")
```

### Defining Orientations
Representing 3D orientation is notoriously tricky. You can use Euler Angles (Roll, Pitch, Yaw), Quaternions, or Axis-Angles. `se3kit` abstracts this away by storing rotations as reliable 3x3 matrices under the hood while letting you input and output whatever format you prefer.

```python
import se3kit as se3
import numpy as np

# Rotate 90 degrees around the Z axis using Roll-Pitch-Yaw
rot_z = se3.Rotation.from_rpy([0.0, 0.0, np.pi/2])

# Avoid Gimbal Lock by using Quaternions (w, x, y, z)
quat = np.quaternion(0.707, 0, 0, 0.707)
rot_quat = se3.Rotation.from_quat(quat)

# Extract basis vectors (X, Y, Z axes of the rotated frame)
print("X-axis points towards:", rot_z.x_axis) # [0, 1, 0]
print("Z-axis points towards:", rot_z.z_axis) # [0, 0, 1]

# Calculate the exact angle between two different rotations
rot_identity = se3.Rotation.eye()
diff_angle = rot_identity.angle_difference(rot_z)
print(f"Angle difference: {se3.utils.rad2deg(diff_angle)} degrees")
```

### Representing Homogeneous Points
A standard 3D point `[x, y, z]` cannot be easily translated using matrix multiplication. To fix this, we add a 4th dimension `[x, y, z, 1]`, creating a *homogeneous point*.

```python
import se3kit as se3

# Initialize an HPoint
point = se3.HPoint(10.0, 5.0, -2.0)

# Access standard 3D coordinates
print(f"Cartesian: {point.xyz}")

# Access the 4D array used for math operations under the hood
print(f"Homogeneous: {point.as_array()}") # [10.0, 5.0, -2.0, 1.0]
```

### Assembling Full Poses
A `Transformation` is the ultimate container. It combines a `Translation` and a `Rotation` into a single 4x4 matrix. It defines a full "Pose" (position and orientation) of an object or coordinate frame.

```python
import se3kit as se3
import numpy as np

# Define a pose: Translated by [1, 1, 0] and rotated 45 degrees around Z
trans = se3.Translation([1.0, 1.0, 0.0])
rot = se3.Rotation.from_rpy([0, 0, np.pi/4])
T_pose = se3.Transformation(trans, rot)

# View the 4x4 SE(3) matrix representation
print(T_pose.m)

# Retrieve the inverse transformation. 
# If T_pose goes from Frame A to Frame B, T_inv goes from Frame B to Frame A.
T_inv = T_pose.inv()
print(f"Inverse Translation: {T_inv.translation.xyz}")
```

---

## Data Validation and Testing

When dealing with floating-point math in physics engines or spatial tests, absolute equality checks often fail. `se3kit` provides built-in methods for robust validation.

### Comparing Poses for Equality
Always use `.are_close()` when checking if two poses or translations are effectively identical.

```python
import se3kit as se3
import numpy as np

T1 = se3.Transformation(se3.Translation([1.0, 0.0, 0.0]), se3.Rotation.from_rpy([0, 0, np.pi]))

# T2 is slightly off due to floating point math approximations
T2 = se3.Transformation(se3.Translation([1.0000000001, 0.0, 0.0]), se3.Rotation.from_rpy([0, 0, 3.1415926535]))

print(f"Are T1 and T2 close? {T1.are_close(T2)}") # Output: True
```

### Verifying Mathematical Validity
You can ensure that a transformation matrix is mathematically valid (e.g., ensuring the rotation matrix is strictly orthogonal).

```python
import se3kit as se3

T1 = se3.Transformation(se3.Translation([1.0, 2.0, 3.0]), se3.Rotation())

# Check if the transformation matrix is a valid SE(3) matrix
print(f"Is T1 a valid SE(3) matrix? {T1.is_valid()}") # Output: True
```

---

## Advanced Rotation Mathematics

Beyond orienting physical objects, rotations are used to modify pure directional forces and velocities, and tie heavily into Lie Algebra.

### Rotating Directional Vectors
Sometimes you don't want to translate a point in space; you simply want to rotate a velocity vector or a force vector into a new coordinate frame.

```python
import se3kit as se3
import numpy as np

# A drone's velocity is [10, 0, 0] m/s in its local frame (flying straight forward)
local_velocity = np.array([10.0, 0.0, 0.0])

# The drone pitches up by 30 degrees (0.523 radians)
drone_rotation = se3.Rotation.from_rpy([0.0, 0.523, 0.0])

# Rotate the velocity vector into the global frame
global_velocity = drone_rotation.rotate_object(local_velocity)

print(f"Global velocity: X={global_velocity[0]:.2f}, Z(Up)={global_velocity[2]:.2f} m/s")
```

### Working with Skew-Symmetric Matrices
Advanced control algorithms and Jacobians require converting between a 3D vector and its 3x3 skew-symmetric matrix representation.

```python
import se3kit as se3
import numpy as np

# Define an angular velocity vector (omega)
omega = np.array([0.1, 0.5, -0.2])

# Convert to a 3x3 skew-symmetric matrix for Lie Algebra calculations
skew_mat = se3.utils.vector_to_skew(omega)
print("Skew Symmetric Matrix:\n", skew_mat)

# Convert the matrix back to a vector
extracted_omega = se3.utils.skew_to_vector(skew_mat)
print("Recovered vector:", extracted_omega)
```

---

## Robot Kinematics and Control

Controlling robotic arms requires assembling coordinate frames together sequentially to understand where the tool tip is located.

### Processing Outputs from Industrial Robot Controllers
Many industrial robots (like Universal Robots) output their poses in a specific format: `[X_mm, Y_mm, Z_mm, Rx_rad, Ry_rad, Rz_rad]`. You can ingest these formats instantly.

```python
import se3kit as se3

# Input: X=500mm, Y=200mm, Z=300mm, Rotations: Rx=0.1, Ry=0.2, Rz=0.3 radians
T_industrial = se3.Transformation.from_xyz_mm_abc(
    x_mm=500.0, y_mm=200.0, z_mm=300.0, 
    a=0.1, b=0.2, c=0.3
)

# Internally, the pose is automatically converted to standard SI meters
print(f"Translation in meters: {T_industrial.translation.xyz}") # Output: [0.5, 0.2, 0.3]
```

### Calculating Forward Kinematics for a Multi-Link Arm
For a 3-DOF planar robot arm, you can find the exact position of the end-effector (tool) by multiplying the transformation matrices of each link sequentially.

```python
import se3kit as se3
import numpy as np

def calculate_end_effector_pose(theta1, theta2, theta3):
    # Base to Joint 1
    T_base_j1 = se3.Transformation(se3.Translation([1.0, 0.0, 0.0]), se3.Rotation.from_rpy([0, 0, theta1]))
    
    # Joint 1 to Joint 2
    T_j1_j2 = se3.Transformation(se3.Translation([0.8, 0.0, 0.0]), se3.Rotation.from_rpy([0, 0, theta2]))
    
    # Joint 2 to End Effector
    T_j2_ee = se3.Transformation(se3.Translation([0.5, 0.0, 0.0]), se3.Rotation.from_rpy([0, 0, theta3]))
    
    # Compose the transformations in sequence (Base -> J1 -> J2 -> EE)
    return T_base_j1 * T_j1_j2 * T_j2_ee

# Input specific joint angles
q1, q2, q3 = np.deg2rad(45), np.deg2rad(-45), np.deg2rad(-45)
ee_pose = calculate_end_effector_pose(q1, q2, q3)

print("End Effector Position (X,Y,Z):", ee_pose.translation.xyz)
```

---

## Frame Transformations and System Integration

Sharing pose data between physics simulators, robot controllers, and ROS (Robot Operating System) often requires careful unit management and serialization.

### Tracking Objects Across Different Coordinate Frames
When a stationary camera tracks a moving ball, you often need to know where the ball is relative to the *World Origin*, not just relative to the camera lens.

```python
import se3kit as se3

# The camera is mounted 2 meters high, pitched down 45 degrees (0.785 rad)
T_world_camera = se3.Transformation(
    se3.Translation([0.0, 0.0, 2.0]), 
    se3.Rotation.from_rpy([0.0, 0.785, 0.0]) 
)

# The camera detects a ball 1 meter straight ahead of the lens
p_camera_ball = se3.HPoint(0.2, 0.0, 1.0)

# Transform the coordinates into the World Frame
p_world_ball = T_world_camera.transform_hpoint(p_camera_ball)

print(f"Ball relative to World Origin: {p_world_ball.xyz}")
```

### Converting Units Between Systems
It is very common to receive data in millimeters and need to export it in meters, or vice versa.

```python
import se3kit as se3

# A transformation received in millimeters
T_robot_mm = se3.Transformation(
    se3.Translation([500.0, -250.0, 1000.0]), 
    se3.Rotation.from_rpy([0, 0, 0])
)

# Safely convert to standard SI meters
T_sim_m = se3.Transformation.convert_mm_to_m(T_robot_mm)
print(f"Translation in meters: {T_sim_m.translation.xyz}")
```

### Exporting Poses for ROS Integration
You can serialize `se3kit` transformations natively into dictionary structures that perfectly match the `geometry_msgs/Pose` ROS message format.

```python
import se3kit as se3
import pprint

T_sim_m = se3.Transformation(
    se3.Translation([0.5, -0.25, 1.0]), 
    se3.Rotation.from_rpy([0, 0, 0])
)

# Export to a dictionary
pose_dict = T_sim_m.as_geometry_pose()

print("Serialized Pose ready for ROS geometry_msgs:")
pprint.pprint(pose_dict)
```

---

## Hardware Calibration

Setting up a robotics workstation requires finding the physical offsets between different pieces of hardware. `se3kit` provides specialized classes for discovering these hidden offsets.

### Eye-in-Hand Camera Calibration
To use a robot-mounted camera for visual servoing, you must find the exact static transformation between the robot's end-effector and the camera lens.

```python
import se3kit as se3
from se3kit.calibration import EyeInHandCalibration
import numpy as np

calibrator = EyeInHandCalibration()

# Record the robot's pose, and the camera's observation of a fixed checkerboard
T_base_ee_1 = se3.Transformation(se3.Translation([0.5, 0.0, 0.5]), se3.Rotation.from_rpy([0, np.pi, 0]))
T_camera_board_1 = se3.Transformation(se3.Translation([0.0, 0.0, 0.4]), se3.Rotation.from_rpy([0.1, 0.0, 0.0]))
calibrator.add_pair(T_base_ee_1, T_camera_board_1)

# Record at least 3 pairs with distinct rotations to allow the mathematics to converge
T_base_ee_2 = se3.Transformation(se3.Translation([0.6, 0.1, 0.5]), se3.Rotation.from_rpy([0, np.pi, 0.1]))
T_camera_board_2 = se3.Transformation(se3.Translation([-0.1, -0.1, 0.42]), se3.Rotation.from_rpy([0.1, -0.1, 0.0]))
calibrator.add_pair(T_base_ee_2, T_camera_board_2)

try:
    T_ee_camera = calibrator.run_calibration()
    print("Camera Offset Translation:", T_ee_camera.translation.xyz)
except Exception as e:
    print(f"Calibration required more variance: {e}")
```

### Tool Center Point (TCP) Pivot Calibration
Pivot calibration finds the physical offset of a tool's tip relative to a tracked marker on its body. By pivoting the tool around a fixed point in space while tracking the marker, the tip offset can be deduced.

```python
import se3kit as se3
from se3kit.calibration import PivotCalibration

calibrator = PivotCalibration()

# Record the marker's pose as the tool is pivoted around its tip
T_camera_marker1 = se3.Transformation(se3.Translation([0.5, 0.0, 1.0]), se3.Rotation.from_rpy([0.1, 0.2, 0.0]))
T_camera_marker2 = se3.Transformation(se3.Translation([0.52, 0.01, 1.0]), se3.Rotation.from_rpy([0.3, 0.1, 0.1]))
T_camera_marker3 = se3.Transformation(se3.Translation([0.48, -0.01, 1.0]), se3.Rotation.from_rpy([-0.1, 0.4, -0.1]))

# Ingest the recorded poses
calibrator.add_poses([T_camera_marker1, T_camera_marker2, T_camera_marker3])

# Run the optimization to find the tool tip translation relative to the marker
try:
    tip_translation, residual_error = calibrator.run_pivot_calibration()
    print(f"Tool Tip Offset: {tip_translation.xyz} (Error: {residual_error:.4f}m)")
except Exception as e:
    print(f"Pivot calibration required more data variance: {e}")
```
