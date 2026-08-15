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

from ..config import get_system_config
from ..data_structs import GeneratedAnimation
from .interpolation_utils import fcurve_interpolation, linear_interpolation, slerp_interpolation

logger = logging.getLogger(__name__)

_system_config = get_system_config()


def interpolate_all(generated_animation: GeneratedAnimation, interpolation_type: str = "auto"):
    keyframes = generated_animation.keyframes  # keyframes by transformations by joints

    if interpolation_type == "auto":
        fcurve_interpolation(generated_animation.translations, keyframes[0])
        slerp_interpolation(generated_animation.rotations, keyframes[1], _system_config.quat_interpolation_type)
        fcurve_interpolation(generated_animation.scales, keyframes[2])
    elif interpolation_type == "linear":
        linear_interpolation(generated_animation.translations, keyframes[0])
        slerp_interpolation(generated_animation.rotations, keyframes[1], _system_config.quat_interpolation_type)
        linear_interpolation(generated_animation.scales, keyframes[2])
    else:
        raise ValueError(f"Invalid interpolation type: {interpolation_type}")
