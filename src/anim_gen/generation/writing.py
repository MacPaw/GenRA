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

import json
import logging
import os
from typing import Any

import numpy as np

from .. import __version__ as _anim_gen_version
from ..builder.hierarchy import get_object_json
from ..config import GenerationMode, get_system_config
from ..data_structs import BaseFile, Config, GeneratedAnimation, InputAnimation
from ..generation.interpolation import interpolate_all
from ..utils import _apply_bind_transforms, apply_modified_animation

logger = logging.getLogger(__name__)

_system_config = get_system_config()


# Keys whose values are serialized on one line to reduce metadata file size
_METADATA_COMPACT_KEYS = frozenset({"animation_json", "keyframes", "object_json", "prompt_strings"})


def _keyframe_stats(keyframes: list[list[list[int]]]) -> dict[str, Any]:
    """Compute total and unique keyframe counts from keyframes[tf][joint] structure."""
    total = sum(
        len(keyframes[tf_i][j])
        for tf_i in range(len(keyframes))
        for j in range(len(keyframes[tf_i]) if keyframes else 0)
    )
    unique_frames: set[int] = set()
    for tf_i in range(len(keyframes)):
        for j in range(len(keyframes[tf_i]) if keyframes else 0):
            unique_frames.update(keyframes[tf_i][j])
    return {"total_keyframes": total, "unique_frame_count": len(unique_frames)}


def _dump_metadata_json(obj: Any, fp: Any, indent: int, indent_level: int, key: str | None) -> None:
    """Serialize obj to JSON; compact (one-line) for keys in _METADATA_COMPACT_KEYS."""
    if key in _METADATA_COMPACT_KEYS and isinstance(obj, (dict, list)):
        fp.write(json.dumps(obj, separators=(",", ":")))
        return
    if isinstance(obj, dict):
        fp.write("{\n")
        items = list(obj.items())
        for i, (k, v) in enumerate(items):
            fp.write(" " * (indent_level + indent) + json.dumps(k) + ": ")
            _dump_metadata_json(v, fp, indent, indent_level + indent, k)
            if i < len(items) - 1:
                fp.write(",")
            fp.write("\n")
        fp.write(" " * indent_level + "}")
        return
    if isinstance(obj, list):
        fp.write("[\n")
        for i, item in enumerate(obj):
            fp.write(" " * (indent_level + indent))
            _dump_metadata_json(item, fp, indent, indent_level + indent, None)
            if i < len(obj) - 1:
                fp.write(",")
            fp.write("\n")
        fp.write(" " * indent_level + "]")
        return
    fp.write(json.dumps(obj))


def write_metadata_file(metadata: dict[str, Any], filepath: str, indent: int = 4) -> None:
    """Write metadata dict to a JSON file with bulky keys (keyframes, animation_json, etc.) on one line."""
    with open(filepath, "w") as f:
        _dump_metadata_json(metadata, f, indent, 0, None)


def _revert_world_axis_remap(
    generated_animation: GeneratedAnimation, fixed_axis_order: tuple[int, int, int], scalars: tuple[int, int, int]
) -> None:
    """
    Revert metadata-space world axis remapping back to the source file axis orientation.
    """
    if fixed_axis_order == (0, 1, 2) and scalars == (1, 1, 1):
        return

    scalars_np = np.asarray(scalars, dtype=generated_animation.translations.dtype)

    reverted_translations = np.empty_like(generated_animation.translations)
    reverted_translations[:, :, fixed_axis_order] = generated_animation.translations * scalars_np
    generated_animation.translations = reverted_translations

    reverted_rotations = np.empty_like(generated_animation.rotations)
    reverted_rotations[:, :, 0] = generated_animation.rotations[:, :, 0]
    reverted_rotations[:, :, tuple(axis + 1 for axis in fixed_axis_order)] = (
        generated_animation.rotations[:, :, 1:] * scalars_np
    )
    generated_animation.rotations = reverted_rotations

    reverted_scales = np.empty_like(generated_animation.scales)
    reverted_scales[:, :, fixed_axis_order] = generated_animation.scales
    generated_animation.scales = reverted_scales


def save_generated_animation(
    generated_animation: GeneratedAnimation,
    anim_request: dict,
    base_file: BaseFile,
    animation_examples: list[InputAnimation],
    gen_anim_path: str = "./generated_animations/generated_anim.usda",
    config: Config | None = None,
    mode: GenerationMode = GenerationMode.GENERATE,
    auto_select_examples: bool = False,
    timing_stats: dict[str, float] | None = None,
    token_usage: dict[str, dict | None] | None = None,
    selected_example_names: list[str] | None = None,
    selection_reasoning: str | None = None,
    generation_succeeded_on_attempt: int | None = None,
) -> tuple[str, dict]:
    """
    Save the generated animation to a specified directory and create metadata files.

    Parameters:
        generated_animation (GeneratedAnimation): The generated animation data.
        anim_request (dict): The user's request details.
        base_file (BaseFile): The base file containing object structure and optionally animation.
        animation_examples (list[InputAnimation]): List of example animations (excludes base).
        gen_anim_path (str): The path to the generated animation file.
        config (Config): Generation configuration.
        mode (GenerationMode): The generation mode.
        auto_select_examples (bool): Whether examples were auto-selected.
        timing_stats (dict): Optional timing per step (total_seconds, selection_seconds, etc.).
        token_usage (dict): Optional token usage per step (selection, generation, motion_description).
        selected_example_names (list): Optional list of example names when auto_select_examples was used.
        selection_reasoning (str): Optional selection reasoning text when auto_select_examples was used.
        generation_succeeded_on_attempt (int): Optional 1-based attempt number on which generation succeeded.

    Returns:
        tuple[str, dict]: A tuple containing the path to the result file and metadata dict.
    """
    if config is None:
        config = Config()

    if not os.path.exists(os.path.dirname(gen_anim_path)):
        os.makedirs(os.path.dirname(gen_anim_path))

    original_filepath = base_file.filepath
    original_metadata = base_file.metadata

    logger.info(f"Applying modified animation to {original_filepath}")

    bind_fixed_axis_order = tuple(getattr(original_metadata, "bind_fixed_axis_order", (0, 1, 2)))
    bind_scalars = tuple(getattr(original_metadata, "bind_scalars", (1, 1, 1)))
    fixed_axis_order = tuple(getattr(original_metadata, "fixed_axis_order", (0, 1, 2)))
    scalars = tuple(getattr(original_metadata, "scalars", (1, 1, 1)))

    # Metadata-space applies world remap first and bind-root remap second.
    # Revert in reverse order to get back to source file orientation.
    _revert_world_axis_remap(generated_animation, bind_fixed_axis_order, bind_scalars)
    _revert_world_axis_remap(generated_animation, fixed_axis_order, scalars)

    if _system_config.bind_relative_transformations:
        _apply_bind_transforms(
            generated_animation.translations,
            generated_animation.rotations,
            generated_animation.scales,
            original_metadata.joint_local_bind_xforms,
        )

    interpolate_all(
        generated_animation, interpolation_type=config.interpolation_type or _system_config.interpolation_type
    )

    apply_modified_animation(
        original_filepath,
        gen_anim_path,
        "generated_animation",  # replace with summarized request later
        generated_animation,
        motion_description=generated_animation.motion_description,
        model_description=generated_animation.model_description,
        anim_gen_version=_anim_gen_version,
    )

    # Base file metadata (include object_json, animation_json only for REFINE)
    base_meta: dict[str, Any] = {
        "filepath": base_file.filepath,
        "caption": base_file.caption,
        "is_autogenerated": base_file.is_autogenerated,
        "object_json": base_file.object_json,
    }
    if mode == GenerationMode.REFINE and base_file.animation_json is not None:
        base_meta["animation_json"] = base_file.animation_json

    # Animation examples with object_json, keyframe counts and keyframes
    examples_meta: list[dict[str, Any]] = []
    for ex in animation_examples:
        ex_dict: dict[str, Any] = {
            "filepath": ex.filepath,
            "caption": ex.caption,
            "is_autogenerated": ex.is_autogenerated,
            "object_json": get_object_json(ex.metadata),
            "animation_json": ex.animation_json,
            "keyframe_count": _keyframe_stats(ex.keyframes)["total_keyframes"],
            "keyframes": ex.keyframes,
        }
        examples_meta.append(ex_dict)

    generated_kf_stats = _keyframe_stats(generated_animation.keyframes)

    generated_animation_meta: dict[str, Any] = {
        "animation_json": generated_animation.animation_json,
        "caption": {
            "motion_description": generated_animation.motion_description,
            "model_description": generated_animation.model_description,
        },
        "keyframes": generated_animation.keyframes,
        "keyframes_count": generated_kf_stats["total_keyframes"],
    }

    metadata: dict[str, Any] = {
        "library_version": _anim_gen_version,
        "mode": mode.value,
        "request": anim_request,
        "auto_select_examples": auto_select_examples,
        "user_config": config.to_dict(),
        "system_config": _system_config.to_dict(),
        "base_file": base_meta,
        "animation_examples": examples_meta,
        "generated_animation": generated_animation_meta,
        "output_filepath": gen_anim_path,
        # "prompt_strings": get_prompt_strings(mode),
    }

    if timing_stats is not None:
        metadata["timing_stats"] = timing_stats
    if token_usage is not None:
        metadata["token_usage"] = token_usage
    if selected_example_names is not None:
        metadata["selected_example_names"] = selected_example_names
    if selection_reasoning is not None:
        metadata["selection_reasoning"] = selection_reasoning
    if generation_succeeded_on_attempt is not None:
        metadata["generation_succeeded_on_attempt"] = generation_succeeded_on_attempt

    return gen_anim_path, metadata
