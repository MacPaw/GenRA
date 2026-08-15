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

from ..config import get_system_config
from ..data_structs import ModelMetadata

_system_config = get_system_config()


def get_object_json(metadata: ModelMetadata) -> dict:
    """
    Returns the hierarchy JSON for the given metadata.

    Parameters:
        metadata (ModelMetadata): An instance of ModelMetadata containing the joint transforms.

    Returns:
        dict: The hierarchy with populated transforms.
    """
    hierarchy = {}

    inc_p = _system_config.include_bind_translations
    inc_r = _system_config.include_bind_rotations
    inc_s = _system_config.include_bind_scales

    # When animation is absolute (not bind-relative), the Object JSON must show
    # joint-local bind values to match the animation's coordinate space.
    force_local = not _system_config.bind_relative_transformations
    bind_tfs = metadata.get_bind_transformations(world_space=False if force_local else None)

    for i, joint in enumerate(metadata.joint_names):
        translations = bind_tfs[0][i].astype(float)
        rotations = bind_tfs[1][i].astype(float)
        scales = bind_tfs[2][i].astype(float)

        entry: dict = {}
        if inc_p:
            entry["p"] = np.round(translations, 2).tolist()
        if inc_r:
            entry["r"] = np.round(rotations, 2).tolist()
        if inc_s:
            entry["s"] = np.round(scales, 2).tolist()
        hierarchy[joint] = entry

    return hierarchy
