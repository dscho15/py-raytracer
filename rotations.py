import numpy as np

def rpy(roll, pitch, yaw):

    # roll
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    # pitch
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    # yaw
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # extrinsic rotation, rotate around z-axis first, then y-axis, then x-axis
    R = R_z @ R_y @ R_x
    
    return R

def slerp(q0, q1, t):
    """Spherical linear interpolation.
    
    Args:
        q0 (np.ndarray): quaternion 0
        q1 (np.ndarray): quaternion 1
        t (float): interpolation parameter in the range [0, 1]
    
    Returns:
        np.ndarray: interpolated quaternion
    """
    x, y, z, w = q0
    x_, y_, z_, w_ = q1

    # dot product
    dot = x * x_ + y * y_ + z * z_ + w * w_

    # If the dot product is negative, slerp won't take
    # the shorter path. Note that v1 and -v1 are equivalent when
    # the negation is applied to all four components. Fix by 
    # reversing one quaternion.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Variables for interpolation
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the inputs are too close for comfort, linearly interpolate
        # and normalize the result.
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    # Since dot is in range [0, DOT_THRESHOLD], acos is safe
    theta_0 = np.arccos(dot)  # theta_0 = angle between input vectors
    theta = theta_0 * t  # theta = angle between v0 and result
    sin_theta = np.sin(theta)  # compute this value only once
    sin_theta_0 = np.sin(theta_0)  # compute this value only once

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0  # == sin(theta_0 - theta) / sin(theta_0)
    s1 = sin_theta / sin_theta_0
    return (s0 * q0) + (s1 * q1)


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = np.array([[1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                  [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                  [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]])
    return R

