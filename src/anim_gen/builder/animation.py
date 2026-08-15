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

import numpy as np

from ..data_structs import ModelMetadata


def get_animation_json(metadata: ModelMetadata, keyframe_idxs: list[list[list[int]]]) -> dict:
    """
    Returns the animation JSON for the given metadata and keyframe indices.

    Parameters:
        metadata (ModelMetadata): The metadata of the model.
        keyframe_idxs (list[list[list[int]]): The indices of the keyframes for each transformation for each joint
    Returns:
        dict: The animation JSON.
    """
    animation_json = {}
    tf_keys = ["p", "r", "s"]

    tfs = metadata.get_transformations()

    for joint_idx, joint_name in enumerate(metadata.joint_names):
        joint_keyframes: dict[float, dict[str, list[float]]] = {}
        for tf_idx, tf_key in enumerate(tf_keys):
            for kf_idx in keyframe_idxs[tf_idx][joint_idx]:
                timestamp = round((metadata.time_codes[kf_idx] - metadata.start_frame) / metadata.fps, 2)

                if timestamp not in joint_keyframes:
                    joint_keyframes[timestamp] = {}

                joint_keyframes[timestamp][tf_key] = np.round(tfs[tf_idx][kf_idx, joint_idx].astype(float), 2).tolist()

        joint_keyframes = dict(sorted(joint_keyframes.items()))

        if len(joint_keyframes) != 0:
            animation_json[joint_name] = joint_keyframes

    return animation_json
