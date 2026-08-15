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

import logging

import numpy as np

from ..config import get_system_config
from ..data_structs import ModelMetadata
from .sampling_utils import cluster_keyframes, post_process_keyframes, print_keyframe_stats, sample_axis

np.seterr(all="raise")  # to raise errors on numerical issues

logger = logging.getLogger(__name__)

_system_config = get_system_config()


def sample_keyframes(metadata: ModelMetadata, sq_sampling_error: float = 0.01) -> list[list[list[int]]]:
    def get_joint_depth(joint_name: str) -> int:
        return joint_name.count("/")

    translations = metadata.translations_bind_relative
    rotations = metadata.rotations_bind_relative
    scales = metadata.scales_bind_relative
    joints = metadata.joint_names
    n_joints = len(joints)

    if translations is None or rotations is None or scales is None:
        raise ValueError("Metadata has missing transformations")

    raw_keyframes_by_tf_by_joint: list[list[set[int]]] = [
        [set() for _ in range(n_joints)],  # translations
        [set() for _ in range(n_joints)],  # rotations
        [set() for _ in range(n_joints)],  # scales
    ]

    tf_errors = [
        # ignore changes relatively small to the overall scale of the translation
        float(np.max(np.abs(translations)) * 1e-5 if np.max(np.abs(translations)) > 1 else 1e-5),
        # as the scales can vary massively between different joints, treat all ranges of scales as equal
        1e-4,
        1e-5,
    ]

    def process_transformations(transformations: np.ndarray, tf_idx: int) -> None:
        num_frames, tf_n_joints, n_axes = transformations.shape
        if tf_n_joints != n_joints:
            raise ValueError("Mismatch in joint count across transforms")

        for j in range(n_joints):
            depth = get_joint_depth(joints[j])

            axis_error_sq_max = sq_sampling_error * (1.0 + 0.1 * depth)

            for axis_i in range(n_axes):
                axis_values = transformations[:, j, axis_i]
                kfs = sample_axis(axis_values.tolist(), axis_error_sq_max, tf_errors[tf_idx])
                raw_keyframes_by_tf_by_joint[tf_idx][j].update(kfs)

    logger.debug(f"Keyframe sampling square error: {sq_sampling_error}")

    process_transformations(translations, tf_idx=0)
    process_transformations(rotations, tf_idx=1)
    process_transformations(scales, tf_idx=2)

    prelim_keyframes: list[list[list[int]]] = []
    for tf_idx in range(3):
        tf_lists: list[list[int]] = []
        for j in range(n_joints):
            tf_lists.append(sorted(raw_keyframes_by_tf_by_joint[tf_idx][j]))
        prelim_keyframes.append(tf_lists)

    logger.debug(f"Tolerance for clustering keyframes: {_system_config.sampling_cluster_tolerance}")
    clustered_keyframes = cluster_keyframes(prelim_keyframes, tolerance=_system_config.sampling_cluster_tolerance)

    # plot_keyframes_timelines(prelim_keyframes, clustered_keyframes, joints)

    if not _system_config.decouple_transformations:
        unified_keyframes: list[set[int]] = [set() for _ in range(len(joints))]
        for i in range(len(joints)):
            unified_keyframes[i].update(clustered_keyframes[0][i])
            unified_keyframes[i].update(clustered_keyframes[1][i])
            unified_keyframes[i].update(clustered_keyframes[2][i])
        unified_keyframes_list: list[list[int]] = [sorted(kf) for kf in unified_keyframes]

        for i in range(3):
            for j in range(len(joints)):
                clustered_keyframes[i][j] = unified_keyframes_list[j]

    # Post-processing check for deltas in angles between keyframes (ensure no more than 180 degrees deltas)
    if _system_config.use_euler_angles:
        clustered_keyframes = post_process_keyframes(clustered_keyframes, metadata.get_transformations()[1])

    logger.info(f"Sampled keyframes: {clustered_keyframes}")
    print_keyframe_stats(clustered_keyframes)

    return clustered_keyframes
