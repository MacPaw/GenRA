# Copyright 2026 MacPaw Way Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License") with the
# "Commons Clause" License Condition v1.0; you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# and the Commons Clause condition in the LICENSE file distributed with this
# software.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class BezierKey:
    left_handle: np.ndarray
    key: np.ndarray
    right_handle: np.ndarray
    auto_handle_type: str = "FREE"


@dataclass
class LinearKeyframe:
    frame: float
    value: np.ndarray


def compute_auto_clamped_handles(
    fcurve: list[BezierKey], auto_smoothing: bool = False, extrapolation: str = "CONSTANT"
) -> None:
    n = len(fcurve)
    if n <= 1:
        return

    for i in range(1, n):
        if fcurve[i].key[0] <= fcurve[i - 1].key[0]:
            raise ValueError("Keyframe times must be strictly increasing.")

    threshold = 0.001

    for i, bezk in enumerate(fcurve):
        prev_ = fcurve[i - 1] if i > 0 else None
        next_ = fcurve[i + 1] if i < n - 1 else None

        bezk.left_handle[0] = min(bezk.left_handle[0], bezk.key[0] - threshold)
        bezk.right_handle[0] = max(bezk.right_handle[0], bezk.key[0] + threshold)

        p2 = bezk.key.copy()
        if prev_ is None:
            if next_ is None:
                continue  # Should not happen when n > 1, but handle for type safety
            p3 = next_.key
            p1 = np.array([2.0 * p2[0] - p3[0], 2.0 * p2[1] - p3[1]])
        else:
            p1 = prev_.key

        if next_ is None:
            p0 = p1  # noqa: F841
            p3 = np.array([2.0 * p2[0] - p1[0], 2.0 * p2[1] - p1[1]])
        else:
            p3 = next_.key

        dvec_a = p2 - p1
        dvec_b = p3 - p2

        len_a = dvec_a[0] if dvec_a[0] != 0 else 1.0
        len_b = dvec_b[0] if dvec_b[0] != 0 else 1.0

        tvec = dvec_b / len_b + dvec_a / len_a

        if auto_smoothing:
            length_factor = 6.0 / 2.5614
        else:
            length_factor = tvec[0]
        length_factor *= 2.5614

        if length_factor != 0.0:
            if not auto_smoothing:
                len_a = min(len_a, 5.0 * len_b)
                len_b = min(len_b, 5.0 * len_a)

            left_violate = False
            right_violate = False

            # Left handle
            scale = len_a / length_factor
            bezk.left_handle = p2 - tvec * scale

            # auto-clamp behavior
            if prev_ is not None and next_ is not None:
                ydiff1 = prev_.key[1] - bezk.key[1]
                ydiff2 = next_.key[1] - bezk.key[1]

                if (ydiff1 <= 0.0 and ydiff2 <= 0.0) or (ydiff1 >= 0.0 and ydiff2 >= 0.0):
                    bezk.left_handle[1] = bezk.key[1]
                    bezk.auto_handle_type = "LOCKED"
                else:
                    if ydiff1 <= 0.0:
                        if prev_.key[1] > bezk.left_handle[1]:
                            bezk.left_handle[1] = prev_.key[1]
                            left_violate = True
                    else:
                        if prev_.key[1] < bezk.left_handle[1]:
                            bezk.left_handle[1] = prev_.key[1]
                            left_violate = True

            # Right handle
            scale = len_b / length_factor
            bezk.right_handle = p2 + tvec * scale

            if prev_ is not None and next_ is not None:
                ydiff1 = prev_.key[1] - bezk.key[1]
                ydiff2 = next_.key[1] - bezk.key[1]
                if (ydiff1 <= 0.0 and ydiff2 <= 0.0) or (ydiff1 >= 0.0 and ydiff2 >= 0.0):
                    bezk.right_handle[1] = bezk.key[1]
                    bezk.auto_handle_type = "LOCKED"
                else:
                    if ydiff1 <= 0.0:
                        if next_.key[1] < bezk.right_handle[1]:
                            bezk.right_handle[1] = next_.key[1]
                            right_violate = True
                    else:
                        if next_.key[1] > bezk.right_handle[1]:
                            bezk.right_handle[1] = next_.key[1]
                            right_violate = True

            if left_violate or right_violate:
                h1_x = bezk.left_handle[0] - p2[0]
                h2_x = p2[0] - bezk.right_handle[0]
                if abs(h1_x) < 1e-12 or abs(h2_x) < 1e-12:
                    pass  # degeneracy
                else:
                    if left_violate:
                        bezk.right_handle[1] = p2[1] + ((p2[1] - bezk.left_handle[1]) / h1_x) * h2_x
                    else:
                        bezk.left_handle[1] = p2[1] + ((p2[1] - bezk.right_handle[1]) / h2_x) * h1_x

    bezk.left_handle[0] = min(bezk.left_handle[0], bezk.key[0] - threshold)
    bezk.right_handle[0] = max(bezk.right_handle[0], bezk.key[0] + threshold)

    if extrapolation == "CONSTANT":
        first = fcurve[0]
        last = fcurve[-1]

        first.auto_handle_type = "LOCKED"
        last.auto_handle_type = "LOCKED"

        first.left_handle[1] = first.key[1]
        first.right_handle[1] = first.key[1]
        last.left_handle[1] = last.key[1]
        last.right_handle[1] = last.key[1]


@lru_cache(maxsize=1000000)
def solve_cubic_for_t(x_target: float, q0: float, q1: float, q2: float, q3: float) -> float | None:
    c0 = q0 - x_target
    c1 = 3.0 * (q1 - q0)
    c2 = 3.0 * (q0 - 2.0 * q1 + q2)
    c3 = q3 - q0 + 3.0 * (q1 - q2)

    # solve c0 + c1 t + c2 t^2 + c3 t^3 = 0
    coeffs = [c3, c2, c1, c0]
    roots = np.roots(coeffs)

    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9:
            t = float(r.real)
            if -1e-9 <= t <= 1.0 + 1e-9:
                real_roots.append(min(max(t, 0.0), 1.0))

    if not real_roots:
        return None

    return min(real_roots, key=lambda t: abs(t - 0.5))


@lru_cache(maxsize=1000000)
def bezier_y_at_t(f1: float, f2: float, f3: float, f4: float, t: float) -> float:
    c0 = f1
    c1 = 3.0 * (f2 - f1)
    c2 = 3.0 * (f1 - 2.0 * f2 + f3)
    c3 = f4 - f1 + 3.0 * (f2 - f3)
    return c0 + t * c1 + (t**2) * c2 + (t**3) * c3


def evaluate_curve(fcurve: list[BezierKey], xs: np.ndarray) -> np.ndarray:
    ys = np.empty_like(xs, dtype=float)
    segments = [(a.key, a.right_handle, b.left_handle, b.key) for a, b in zip(fcurve[:-1], fcurve[1:], strict=False)]

    x_first, y_first = fcurve[0].key
    x_last, y_last = fcurve[-1].key
    j = 0
    for k, x in enumerate(xs):
        if x <= x_first:
            ys[k] = y_first
            continue
        if x >= x_last:
            ys[k] = y_last
            continue

        while j < len(segments) and x > segments[j][3][0]:
            j += 1
        if j >= len(segments):
            ys[k] = segments[-1][3][1]
            continue

        v1, v2, v3, v4 = segments[j]

        if abs(v4[0] - v1[0]) < 1e-6:
            ys[k] = v1[1]
            continue

        t = solve_cubic_for_t(x, v1[0], v2[0], v3[0], v4[0])
        if t is None:
            t = (x - v1[0]) / (v4[0] - v1[0])
            t = min(max(t, 0.0), 1.0)

        y = bezier_y_at_t(v1[1], v2[1], v3[1], v4[1], t)
        ys[k] = y

    return ys


def interpolate_axis(transformations, keyframes):  # maybe expose auto_smoothing parameter
    fcurve = []
    for kf in keyframes:
        axis_transformation = transformations[kf]

        vec1 = np.array([kf, axis_transformation], dtype=float)
        vec0 = np.array([kf - 0.333, axis_transformation], dtype=float)
        vec2 = np.array([kf + 0.333, axis_transformation], dtype=float)
        fcurve.append(BezierKey(left_handle=vec0, key=vec1, right_handle=vec2))

    compute_auto_clamped_handles(fcurve, auto_smoothing=True)
    return evaluate_curve(fcurve, np.arange(0, transformations.shape[0]))


def fcurve_interpolation(transformations, keyframes_by_joints):
    for joint_idx in range(transformations.shape[1]):
        for axis_idx in range(3):
            transformations[:, joint_idx, axis_idx] = interpolate_axis(
                transformations[:, joint_idx, axis_idx], keyframes_by_joints[joint_idx]
            )


def eval_linear(t, lin_keyframes):
    if t <= lin_keyframes[0].frame:
        return lin_keyframes[0].value
    if t >= lin_keyframes[-1].frame:
        return lin_keyframes[-1].value
    for i in range(len(lin_keyframes) - 1):
        k1, k2 = lin_keyframes[i], lin_keyframes[i + 1]
        if k1.frame <= t <= k2.frame:
            return k1.value + (k2.value - k1.value) * (t - k1.frame) / (k2.frame - k1.frame)
    return lin_keyframes[-1].value


def linear_interpolation(transformations, keyframes_by_joints):
    for joint_idx in range(transformations.shape[1]):
        lin_keyframes = [
            LinearKeyframe(frame=kf, value=transformations[kf, joint_idx]) for kf in keyframes_by_joints[joint_idx]
        ]
        for t in np.arange(0, transformations.shape[0]):
            transformations[t, joint_idx] = eval_linear(t, lin_keyframes)


def normalize(q):
    return q / np.linalg.norm(q)


def slerp(q1, q2, t):
    """Spherical linear interpolation (normalized quaternions)."""
    q1 = normalize(q1)
    q2 = normalize(q2)
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)

    if dot < 0.0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        return normalize((1 - t) * q1 + t * q2)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q1 + b * q2


def smoothstep(t):
    """Smoothstep easing (3t^2 - 2t^3)."""
    return 3 * t**2 - 2 * t**3


@dataclass
class QuaternionKeyframe:
    frame: float
    quat: np.ndarray


def eval_quaternion(f: float, keys: list[QuaternionKeyframe], smooth: bool = True) -> np.ndarray:
    """Evaluate quaternion at frame f using SLERP + smoothstep easing."""
    if f <= keys[0].frame:
        return keys[0].quat
    if f >= keys[-1].frame:
        return keys[-1].quat

    for i in range(len(keys) - 1):
        k1, k2 = keys[i], keys[i + 1]
        if k1.frame <= f <= k2.frame:
            t = (f - k1.frame) / (k2.frame - k1.frame)
            if smooth:
                t = smoothstep(t)
            result = slerp(k1.quat, k2.quat, t)
            return np.asarray(result)

    return keys[-1].quat


def slerp_interpolation(rotations, keyframes_by_joints, interpolation_type: str = "smooth_slerp"):
    if interpolation_type not in ["slerp", "smooth_slerp"]:
        raise ValueError(f"Invalid interpolation type or not implemented: {interpolation_type}")

    smooth = interpolation_type == "smooth_slerp"

    for joint_idx in range(len(keyframes_by_joints)):
        keyframes = keyframes_by_joints[joint_idx]
        quat_keyframes = []
        for kf in keyframes:
            quat = rotations[kf, joint_idx, :]
            quat_keyframes.append(QuaternionKeyframe(frame=kf, quat=quat))

        for t in np.arange(0, rotations.shape[0]):
            rotations[t, joint_idx, :] = eval_quaternion(t, quat_keyframes, smooth)


# def visualize_fcurve(x_min, x_max, fcurve):
#     xs = np.linspace(x_min, x_max, 800)
#     ys = evaluate_curve(fcurve, xs)

#     plt.figure(figsize=(9, 5))
#     plt.plot(xs, ys, label="Interpolated curve", zorder=1)
#     plt.scatter([b.key[0] for b in fcurve], [b.key[1] for b in fcurve], label="Keyframes", zorder=3, color="orange")

#     for b in fcurve:
#         plt.plot([b.left_handle[0], b.key[0]], [b.left_handle[1], b.key[1]], 'gray')
#         plt.plot([b.key[0], b.right_handle[0]], [b.key[1], b.right_handle[1]], 'gray')

#     plt.title("Auto-clamped Bezier Curve")
#     plt.xlabel("Frame / Time")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.show()
