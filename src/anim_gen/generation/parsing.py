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

import ast
import datetime
import json
import logging

import numpy as np

from ..config import get_system_config
from ..data_structs import GeneratedAnimation, ModelMetadata
from ..quat_utils import make_quat_from_euler

logger = logging.getLogger(__name__)

_system_config = get_system_config()


def parse_generated_animation(response_str: str, basefile_metadata: ModelMetadata) -> GeneratedAnimation:
    """
    Parse the generated animation from the response string.

    Parameters:
        response_str (str): The response string from the API containing the generated animation.
        basefile_metadata (ModelMetadata): Metadata about the base animation file.

    Returns:
        GeneratedAnimation: The parsed generated animation.
    """
    logger.info("Parsing generated animation...")
    logger.debug(f"Response string: {response_str}")
    response_str = response_str.replace("```json", "").replace("```", "")

    try:
        generated_animation = ast.literal_eval(response_str)
    except Exception as e:
        raise ValueError(f"Failed to parse generated animation: {e}") from e

    if "error" in generated_animation:
        raise ValueError(f"Failed to generate animation: {generated_animation['error']}")

    logger.debug(f"Generated animation: {json.dumps(generated_animation, indent=4)}")

    base_joint_names = basefile_metadata.joint_names
    generated_joint_names = list(generated_animation.keys())

    max_timestamp = -np.inf
    for joint_name in generated_joint_names:
        for timestamp_key in generated_animation[joint_name].keys():
            max_timestamp = max(max_timestamp, float(timestamp_key))

    fps = 60
    max_kf_idx = int(np.round(max_timestamp * fps))

    keyframes: list[list[list[int]]] = [[[] for _ in range(len(base_joint_names))] for _ in range(3)]

    translations = np.full((max_kf_idx + 1, len(base_joint_names), 3), np.nan, dtype=np.float64)

    if _system_config.use_euler_angles:
        rotations = np.full((max_kf_idx + 1, len(base_joint_names), 3), np.nan, dtype=np.float64)
    else:
        rotations = np.full((max_kf_idx + 1, len(base_joint_names), 4), np.nan, dtype=np.float64)

    scales = np.full((max_kf_idx + 1, len(base_joint_names), 3), np.nan, dtype=np.float64)

    # Unconditionally set the first frame to the base file values to handle the case
    # when no keyframes are generated for a joint

    # tfs = basefile_metadata.get_transformations()

    # translations[0, :, :] = tfs[0][0, :, :]
    # rotations[0, :, :] = tfs[1][0, :, :]
    # scales[0, :, :] = tfs[2][0, :, :]

    # For now, instead of using first frames, set the transformations to identity as long
    # as we use bind relative transformations. If not, for now, raise an error.
    # if not _system_config.bind_relative_transformations:
    #     raise ValueError(
    #         "Non-bind relative transformations are not supported yet. Rest transform handling should be implemented."
    #     )

    translations[0, :, :] = np.zeros((len(base_joint_names), 3))
    if _system_config.use_euler_angles:
        rotations[0, :, :] = np.zeros((len(base_joint_names), 3))
    else:
        rotations[0, :, :] = np.tile([1, 0, 0, 0], (len(base_joint_names), 1))
    scales[0, :, :] = np.ones((len(base_joint_names), 3))

    for joint_name, joint_keyframes in generated_animation.items():
        try:
            joint_idx = base_joint_names.index(joint_name)
        except ValueError as e:
            raise ValueError(f"Joint name '{joint_name}' not found in base file joint names") from e

        for timestamp, keyframe in joint_keyframes.items():
            kf = int(np.round(float(timestamp) * fps))
            if "p" in keyframe:
                translations[kf, joint_idx] = keyframe["p"]
                keyframes[0][joint_idx].append(kf)  # set the keyframe to be global timecode
            if "r" in keyframe:
                rotations[kf, joint_idx] = keyframe["r"]
                keyframes[1][joint_idx].append(kf)
            if "s" in keyframe:
                # set any numbers less than 1e-3 present in the scale to 1e-3 to avoid singularity
                kf_scale = np.array(keyframe["s"])
                kf_scale = np.where(kf_scale < 1e-3, 1e-3, kf_scale)
                scales[kf, joint_idx] = kf_scale

                keyframes[2][joint_idx].append(kf)

    for tf_idx in range(3):
        for joint_idx in range(len(base_joint_names)):
            keyframes[tf_idx][joint_idx] = sorted(set(keyframes[tf_idx][joint_idx]))

    for tf_idx in range(3):
        for joint_idx in range(len(base_joint_names)):
            joint_keyframes = keyframes[tf_idx][joint_idx]
            # We ensure that any given joint has a 0th keyframe, as it is the default pose.
            if 0 not in joint_keyframes:
                keyframes[tf_idx][joint_idx].insert(0, 0)

    if _system_config.use_euler_angles:
        rotations_quats = np.full((max_kf_idx + 1, len(base_joint_names), 4), np.nan, dtype=np.float64)
        for i in range(max_kf_idx + 1):
            for j in range(len(base_joint_names)):
                if not np.any(np.isnan(rotations[i, j])):
                    quat = make_quat_from_euler(rotations[i, j])
                    rotations_quats[i, j] = np.array([quat.GetReal(), *quat.GetImaginary()])
                else:
                    rotations_quats[i, j] = np.array([np.nan, np.nan, np.nan, np.nan])

        rotations = rotations_quats

    generated_animation_metadata = GeneratedAnimation(
        animation_json=generated_animation,
        translations=translations,
        rotations=rotations,
        scales=scales,
        keyframes=keyframes,
        joint_names=base_joint_names,
        fps=fps,
        end_frame=max_kf_idx,
        datetime=datetime.datetime.now().isoformat(),
    )

    logger.debug(f"Keyframes by tfs by joint: {keyframes}")
    logger.info(f"Number of generated keyframes: {sum(sum(len(kf) for kf in tf_idxs) for tf_idxs in keyframes)}")

    return generated_animation_metadata
