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

import heapq
import logging
import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)


@dataclass
class BezierKey:
    vec: np.ndarray

    def copy(self):
        return BezierKey(self.vec.copy())


@dataclass
class Node:
    index: int
    point_index: int
    tan: np.ndarray
    handles: np.ndarray
    can_remove: bool
    prev: Optional["Node"] = None
    next: Optional["Node"] = None
    is_removed: bool = False

    version: int = 0
    cost_sq: float = np.inf
    new_prev_right_len: float = 0.0
    new_next_left_len: float = 0.0


def build_bezier_from_xy(xs: np.ndarray, ys: np.ndarray) -> list[BezierKey]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    n = len(xs)
    beziers: list[BezierKey] = []

    for i in range(n):
        key = np.array([xs[i], ys[i]])

        # left handle
        if i > 0:
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            hl = np.array([xs[i] - dx / 3.0, ys[i] - dy / 3.0])
        else:
            hl = key.copy()

        # right handle
        if i < n - 1:
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            hr = np.array([xs[i] + dx / 3.0, ys[i] + dy / 3.0])
        else:
            hr = key.copy()

        vec = np.stack([hl, key, hr], axis=0)
        beziers.append(BezierKey(vec))

    return beziers


@njit(cache=True)
def eval_cubic_bezier(p0, p1, p2, p3, t):
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3.0 * u * u * t
    b2 = 3.0 * u * t * t
    b3 = t * t * t
    return (b0 * p0) + (b1 * p1) + (b2 * p2) + (b3 * p3)


def sample_curve_from_beziers(beziers: list[BezierKey], resolu: int = 32) -> np.ndarray:
    pts = []
    n = len(beziers)
    for i in range(n - 1):
        b0 = beziers[i].vec[1]
        b1 = beziers[i].vec[2]
        b2 = beziers[i + 1].vec[0]
        b3 = beziers[i + 1].vec[1]
        for j in range(resolu):
            t = j / float(resolu)
            pts.append(eval_cubic_bezier(b0, b1, b2, b3, t))
    # final endpoint
    pts.append(beziers[-1].vec[1])
    return np.array(pts)


@njit(cache=True)
def chord_length_params(pts: np.ndarray) -> np.ndarray:
    n = pts.shape[0]
    cum = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        diff = pts[i] - pts[i - 1]
        seg_len = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
        cum[i] = cum[i - 1] + seg_len

    total_len = cum[-1]
    if total_len == 0.0:
        return cum  # all zeros

    for i in range(n):
        cum[i] /= total_len

    return cum


@njit(cache=True)
def fit_cubic_with_tangents(
    points: np.ndarray, tan_l: np.ndarray, tan_r: np.ndarray
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    P0 = points[0]
    P3 = points[-1]

    ts = chord_length_params(points)  # (M,)
    u = 1.0 - ts
    b0 = u * u * u
    b1 = 3.0 * u * u * ts
    b2 = 3.0 * u * ts * ts
    b3 = ts * ts * ts

    # Rewrite Bezier equation to a linear system in a,b:
    #
    # B(t) = P0*(b0+b1) + P3*(b2+b3)
    #        + a * (tan_l * b1) + b * (tan_r * b2)
    #
    # Let RHS_i = Q_i - [P0*(b0+b1) + P3*(b2+b3)]
    # We solve least squares in [a,b] over all i.

    base_start = (b0 + b1)[:, None] * P0[None, :]
    base_end = (b2 + b3)[:, None] * P3[None, :]
    rhs = points - (base_start + base_end)  # shape (M, 2)

    # Build normal equations for 2 unknowns.
    tl_tl = float(np.dot(tan_l, tan_l))
    tr_tr = float(np.dot(tan_r, tan_r))
    tl_tr = float(np.dot(tan_l, tan_r))

    A00 = float(np.sum((b1 * b1) * tl_tl))
    A01 = float(np.sum((b1 * b2) * tl_tr))
    A11 = float(np.sum((b2 * b2) * tr_tr))

    v0 = float(np.sum(b1 * (rhs @ tan_l)))
    v1 = float(np.sum(b2 * (rhs @ tan_r)))

    det = A00 * A11 - A01 * A01

    if abs(det) > 1e-7:
        a_len = float((A11 * v0 - A01 * v1) / det)
        b_len = float((-A01 * v0 + A00 * v1) / det)
    else:
        chord = np.linalg.norm(P3 - P0)
        a_len = float(chord / 3.0)
        b_len = float(chord / 3.0)

    P1 = P0 + a_len * tan_l
    P2 = P3 + b_len * tan_r

    curve_pts = (
        b0[:, None] * P0[None, :] + b1[:, None] * P1[None, :] + b2[:, None] * P2[None, :] + b3[:, None] * P3[None, :]
    )
    diffs = curve_pts - points
    err_sq_each = np.sum(diffs * diffs, axis=1)
    error_sq_max = float(np.max(err_sq_each))

    return a_len, b_len, error_sq_max, P1, P2


def build_nodes_from_beziers(
    beziers: list[BezierKey],
    resolu: int = 12,
) -> tuple[list[Node], np.ndarray]:
    # Dense curve samples.
    points = sample_curve_from_beziers(beziers, resolu=resolu)

    Nodes: list[Node] = []
    n = len(beziers)

    for i, b in enumerate(beziers):
        left_vec = b.vec[0] - b.vec[1]
        left_len = float(np.linalg.norm(left_vec))
        if left_len > 0:
            tan0_unit = left_vec / left_len
        else:
            tan0_unit = np.zeros(2)

        right_vec = b.vec[1] - b.vec[2]
        right_len = float(np.linalg.norm(right_vec))
        if right_len > 0:
            tan1_unit = right_vec / right_len
        else:
            tan1_unit = np.zeros(2)

        handles = np.array([left_len, -right_len], dtype=float)
        tan = np.stack([tan0_unit, tan1_unit], axis=0)

        k = Node(
            index=i,
            point_index=i * resolu,
            tan=tan,
            handles=handles,
            can_remove=(i not in (0, n - 1)),
        )
        Nodes.append(k)

    for i, k in enumerate(Nodes):
        if i > 0:
            k.prev = Nodes[i - 1]
        if i < n - 1:
            k.next = Nodes[i + 1]

    Nodes[0].can_remove = False
    Nodes[-1].can_remove = False

    return Nodes, points


def node_removal_cost(k: Node, points: np.ndarray, resolu: int = 12) -> tuple[float, tuple[float, float]]:
    if k.prev is None or k.next is None:
        raise ValueError("Node must have both prev and next neighbors")

    k_prev = k.prev
    k_next = k.next

    start_idx = k_prev.point_index
    end_idx = k_next.point_index
    if end_idx < start_idx:
        raise RuntimeError("Unexpected wrap-around in non-cyclic curve.")

    seg_points = points[start_idx : end_idx + 1]

    tan_l = k_prev.tan[1]
    tan_r = k_next.tan[0]

    a_len, b_len, error_sq_max, _, _ = fit_cubic_with_tangents(seg_points, tan_l, tan_r)

    return error_sq_max, (a_len, b_len)


def recompute_node_removal_cost(
    k: Node,
    points: np.ndarray,
    heap: list[tuple[float, int, int, Node]],
    error_sq_max: float,
    resolu: int,
):
    if (
        k.is_removed
        or (not k.can_remove)
        or (k.prev is None)
        or (k.next is None)
        or k.prev.is_removed
        or k.next.is_removed
    ):
        return

    start_idx = k.prev.point_index
    end_idx = k.next.point_index
    if end_idx < start_idx:
        return

    seg_points = points[start_idx : end_idx + 1]

    tan_l = k.prev.tan[1]
    tan_r = k.next.tan[0]

    a_len, b_len, cost_sq, _, _ = fit_cubic_with_tangents(seg_points, tan_l, tan_r)

    if cost_sq < error_sq_max:
        k.version += 1
        k.cost_sq = cost_sq
        k.new_prev_right_len = a_len
        k.new_next_left_len = b_len

        heapq.heappush(heap, (cost_sq, k.version, k.index, k))  # type: ignore[misc]
    else:
        k.version += 1
        k.cost_sq = math.inf


def sample_curve_heap(
    nodes: list[Node],
    points: np.ndarray,
    error_sq_max: float,
    error_target_len: int,
    resolu: int = 12,
) -> None:
    total_remaining = sum(1 for k in nodes if not k.is_removed)
    error_target_len = max(error_target_len, 2)

    heap: list[tuple[float, int, int, Node]] = []
    for k in nodes:
        recompute_node_removal_cost(k, points, heap, error_sq_max, resolu)

    while total_remaining > error_target_len and heap:
        cost_sq, ver, _, k = heapq.heappop(heap)

        if (
            k.is_removed
            or ver != k.version
            or k.prev is None
            or k.next is None
            or k.prev.is_removed
            or k.next.is_removed
            or cost_sq != k.cost_sq
            or cost_sq >= error_sq_max
        ):
            continue

        k_prev = k.prev
        k_next = k.next

        k_prev.handles[1] = k.new_prev_right_len
        k_next.handles[0] = k.new_next_left_len

        k_prev.next = k_next
        k_next.prev = k_prev

        k.prev = None
        k.next = None
        k.is_removed = True
        total_remaining -= 1

        if k_prev.can_remove and not k_prev.is_removed:
            recompute_node_removal_cost(k_prev, points, heap, error_sq_max, resolu)
        if k_next.can_remove and not k_next.is_removed:
            recompute_node_removal_cost(k_next, points, heap, error_sq_max, resolu)


def sample_beziers_heap(
    beziers: list[BezierKey], remove_ratio: float = 1.0, error_sq_max: float = float("inf"), resolu: int = 12
) -> list[BezierKey]:
    n = len(beziers)
    selected_len = n
    bezt_segment_len = n

    target_fcurve_verts = math.ceil(bezt_segment_len - selected_len * remove_ratio)

    nodes, points = build_nodes_from_beziers(beziers, resolu=resolu)

    sample_curve_heap(
        nodes,
        points,
        error_sq_max=error_sq_max,
        error_target_len=target_fcurve_verts,
        resolu=resolu,
    )

    new_beziers: list[BezierKey] = []
    for i, (b, k) in enumerate(zip(beziers, nodes, strict=False)):
        if k.is_removed:
            continue

        b_new = b.copy()

        if i > 0 and nodes[i - 1].is_removed:
            b_new.vec[0] = b_new.vec[1] + k.tan[0] * k.handles[0]

        if i < n - 1 and nodes[i + 1].is_removed:
            b_new.vec[2] = b_new.vec[1] + k.tan[1] * k.handles[1]

        new_beziers.append(b_new)

    return new_beziers


def cluster_keyframes(
    keyframes_by_tf_by_joint: Sequence[Sequence[Sequence[int]]],
    tolerance: int = 1,
) -> list[list[list[int]]]:
    def _cluster_one_tf(joint_keyframes_lists: Sequence[Sequence[int]]) -> list[list[int]]:
        joint_keyframes_lists = [sorted(set(kfs)) for kfs in joint_keyframes_lists]

        if tolerance == 0:
            return joint_keyframes_lists  # type: ignore[return-value]

        all_keys = sorted({k for joint_list in joint_keyframes_lists for k in joint_list})

        if not all_keys:
            return [[] for _ in joint_keyframes_lists]

        clusters: list[list[int]] = []
        current_cluster: list[int] = [all_keys[0]]

        for k in all_keys[1:]:
            if all(abs(k - c) <= tolerance for c in current_cluster):
                current_cluster.append(k)
            else:
                clusters.append(current_cluster)
                current_cluster = [k]
        clusters.append(current_cluster)

        cluster_map = {}
        for cluster in clusters:
            cluster.sort()
            n = len(cluster)
            representative = cluster[n // 2]
            for k in cluster:
                cluster_map[k] = representative

        reduced = []
        for joint_list in joint_keyframes_lists:
            mapped = [cluster_map[k] for k in joint_list]
            reduced.append(sorted(set(mapped)))

        return reduced

    result: list[list[list[int]]] = []
    for tf_keyframes_for_all_joints in keyframes_by_tf_by_joint:
        clustered_tf = _cluster_one_tf(tf_keyframes_for_all_joints)
        result.append(clustered_tf)

    return result


def sample_axis(values: Sequence[float], error_sq_max: float, tf_error: float = 1e-5) -> list[int]:
    xs = np.arange(len(values))
    ys = np.array(values, dtype=float)

    if np.allclose(ys, ys[0], atol=tf_error):
        return []

    max_ys = np.max(np.abs(ys))
    if max_ys < tf_error:
        return []

    normalized_ys = ys / max_ys

    beziers = build_bezier_from_xy(xs, normalized_ys)

    beziers_dec = sample_beziers_heap(
        beziers,
        remove_ratio=1.0,
        error_sq_max=error_sq_max,
        resolu=10,
    )

    keyframes = [b.vec[1][0].astype(int).item() for b in beziers_dec]

    if len(keyframes) >= 2 and np.isclose(values[keyframes[-1]], values[keyframes[-2]], atol=tf_error):
        keyframes.pop(-1)

    if len(keyframes) == 1:
        keyframes = []

    return keyframes


def post_process_keyframes(keyframes: list[list[list[int]]], rotations: np.ndarray) -> list[list[list[int]]]:
    """
    Check for deltas in angles on each Euler axis between consecutive keyframes for each joint.
    If the arc of motion between keyframes exceeds 180 degrees (peak-to-peak > 180),
    add intermediate keyframes recursively until all segments are within range.
    """
    # Create a deep copy of the structure to avoid mutating the input in place unexpectedly
    new_keyframes = deepcopy(keyframes)

    rotation_keyframes = new_keyframes[1]
    n_joints = len(rotation_keyframes)
    # n_frames = rotations.shape[0]

    for j in range(n_joints):
        kfs = rotation_keyframes[j]
        # Ensure sorted and unique
        kfs = sorted(set(kfs))

        i = 0
        while i < len(kfs) - 1:
            k1 = kfs[i]
            k2 = kfs[i + 1]

            # Cannot split if adjacent
            if k2 - k1 <= 1:
                i += 1
                continue

            should_split = False

            # Check all 3 axes
            for axis in range(3):
                # Get the sequence of angles for this joint and axis in the range [k1, k2] inclusive
                angles = rotations[k1 : k2 + 1, j, axis]

                # Unwrap to handle the -180/180 discontinuity (degrees)
                unwrapped = np.unwrap(angles, period=360)

                # Check range (peak-to-peak)
                _min = np.min(unwrapped)
                _max = np.max(unwrapped)
                if (_max - _min) >= 180.0:
                    should_split = True
                    break

            if should_split:
                mid = (k1 + k2) // 2
                if mid > k1 and mid < k2:
                    kfs.insert(i + 1, mid)
                    # Do not increment i, so we check the new segment [k1, mid] next
                else:
                    i += 1
            else:
                i += 1

        rotation_keyframes[j] = kfs

    return new_keyframes


def compute_keyframe_stats(keyframes_by_tf_by_joint: Sequence[Sequence[Sequence[int]]]) -> dict[str, Any]:
    n_tf = len(keyframes_by_tf_by_joint)
    n_joints = len(keyframes_by_tf_by_joint[0]) if n_tf > 0 else 0

    total_keyframes = 0
    for tf_idx in range(n_tf):
        for j in range(n_joints):
            total_keyframes += len(keyframes_by_tf_by_joint[tf_idx][j])

    total_old_style = 0
    for j in range(n_joints):
        union_frames_for_joint: set[int] = set()
        for tf_idx in range(n_tf):
            union_frames_for_joint.update(keyframes_by_tf_by_joint[tf_idx][j])
        total_old_style += len(union_frames_for_joint) * n_tf

    global_unique_frames: set[int] = set()
    for tf_idx in range(n_tf):
        for j in range(n_joints):
            global_unique_frames.update(keyframes_by_tf_by_joint[tf_idx][j])

    stats = {
        "total_keyframes": total_keyframes,
        "total_keyframes_old_style": total_old_style,
        "unique_frame_count_global": len(global_unique_frames),
    }

    return stats


def print_keyframe_stats(keyframes_by_tf_by_joint: Sequence[Sequence[Sequence[int]]]) -> None:
    stats = compute_keyframe_stats(keyframes_by_tf_by_joint)
    logger.info(f"Keyframe Stats: {stats}")
