"""Vectorized quaternion and rigid-transform helpers (XYZW, Hamilton product).

These mirror the Warp built-ins the MotionBricks runtime uses, so numpy results
match the reference kernels.
"""

import numpy as np

IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def qnorm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)


def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def qinv(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def qrot(q, v):
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def qrot_inv(q, v):
    return qrot(qinv(q), v)


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    half = angle[..., None] * 0.5
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=-1)


def yaw_q(angle):
    return axis_angle([0.0, 1.0, 0.0], angle)


def slerp(a, b, t):
    """Shortest-path slerp. t broadcasts against the leading dims."""
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot).clip(-1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6
    safe = np.where(small, 1.0, sin_theta)
    wa = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    wb = np.where(small, t, np.sin(t * theta) / safe)
    return qnorm(wa * a + wb * b)


def quat_from_matrix(m):
    """Rotation matrix [..., 3, 3] (columns are axes) to XYZW quaternion."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    trace = m00 + m11 + m22
    q = np.zeros(m.shape[:-2] + (4,))

    c0 = trace > 0
    s = np.sqrt(np.where(c0, trace + 1.0, 1.0)) * 2
    q0 = np.stack([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s], -1)

    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    s = np.sqrt(np.clip(1.0 + m00 - m11 - m22, 1e-12, None)) * 2
    q1 = np.stack([0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s], -1)

    c2 = (~c0) & (~c1) & (m11 > m22)
    s = np.sqrt(np.clip(1.0 + m11 - m00 - m22, 1e-12, None)) * 2
    q2 = np.stack([(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s], -1)

    s = np.sqrt(np.clip(1.0 + m22 - m00 - m11, 1e-12, None)) * 2
    q3 = np.stack([(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s], -1)

    q = np.where(c0[..., None], q0, np.where(c1[..., None], q1, np.where(c2[..., None], q2, q3)))
    return qnorm(q)


def rotation_from_6d(x_raw, y_raw):
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True).clip(1e-12)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True).clip(1e-12)
    y = np.cross(z, x)
    return quat_from_matrix(np.stack([x, y, z], axis=-1))


# Transforms are (p [...,3], q [...,4]) tuples.


def tmul(a, b):
    ap, aq = a
    bp, bq = b
    return ap + qrot(aq, bp), qmul(aq, bq)


def tinv(a):
    p, q = a
    qi = qinv(q)
    return -qrot(qi, p), qi


def tblend(a, b, t):
    t = np.asarray(t, dtype=np.float64)
    return a[0] + (b[0] - a[0]) * t[..., None], slerp(a[1], b[1], t)


def heading_from_hips(right, left):
    """Heading angle used by the model: forward = Y x (right - left), angle = atan2(fx, fz)."""
    ax = right[..., 0] - left[..., 0]
    az = right[..., 2] - left[..., 2]
    length = np.sqrt(ax * ax + az * az)
    ok = length > 1e-6
    ax = np.where(ok, ax / np.where(ok, length, 1), 0)
    az = np.where(ok, az / np.where(ok, length, 1), 0)
    # cross((0,1,0), (ax,0,az)) = (az, 0, -ax)
    return np.where(ok, np.arctan2(az, -ax), 0.0)
