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

import datetime
import glob
import json
import logging
import os
import time
import traceback
from collections.abc import Callable
from datetime import timezone
from typing import Any

from ..config import GenerationMode, get_system_config
from ..data_structs import AnimationFile, BaseFile, Config, GeneratedAnimation, InputAnimation
from ..log import setup_loggers
from ..rendering.render_wrapper import convert_animation_to_glb, render_animation, render_rig_overlay
from .api import api_request
from .joint_cleanup import apply_joint_name_mapping, cleanup_joint_names, rename_json_keys
from .motion_description import optimize_motion_description_generate, optimize_motion_description_refine
from .parsing import parse_generated_animation
from .selection import select_examples
from .utils import build_prompt_strings, prepare_animation_example, prepare_base_file
from .validation import _validate_config, _validate_request
from .writing import save_generated_animation, write_metadata_file

logger = logging.getLogger(__name__)

_system_config = get_system_config()

# Progress event stages (for progress_callback)
STAGE_VALIDATION = "validation"
STAGE_EXAMPLE_SELECTION = "example_selection"
STAGE_KEYFRAME_SAMPLING_BASE = "keyframe_sampling_base"
STAGE_KEYFRAME_SAMPLING_EXAMPLES = "keyframe_sampling_examples"
STAGE_JOINT_CLEANUP = "joint_name_cleanup"
STAGE_GENERATION = "generation"
STAGE_MOTION_DESCRIPTION = "motion_description"
STAGE_SAVING = "saving"
STAGE_RIG_RENDERING = "rig_rendering"
STAGE_RENDERING = "rendering"
STAGE_GLTF_CONVERSION = "gltf_conversion"


def _emit(progress_callback: Callable[[dict], None] | None, event: str, payload: dict) -> None:
    """Build and invoke progress callback with a single event."""
    if progress_callback is None:
        return
    try:
        progress_callback(
            {
                "event": event,
                "ts": datetime.datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
        )
    except Exception as e:
        logger.debug("Progress callback failed: %s", e)


def _optimize_motion_description(
    base_file: BaseFile, request: dict, mode: GenerationMode
) -> tuple[str, str, dict | None]:
    """
    Optimize motion description based on the generation mode.

    Parameters:
        base_file (BaseFile): The base file containing caption information.
        request (dict): A dictionary containing the user's request details.
        mode (GenerationMode): The mode of generation (GENERATE or REFINE).

    Returns:
        tuple[str, str, dict | None]: (optimized_motion_description, model_description, token_usage).

    Raises:
        ValueError: If required caption information is missing.
    """
    logger.info("Optimizing motion description...")
    user_prompt = request.get("description", "")

    # Get model description from base file
    if base_file.caption is None:
        raise ValueError("Base file must have caption")
    model_description = base_file.caption.get("model_description")
    if model_description is None:
        raise ValueError("Base file must have model_description in caption")

    if mode == GenerationMode.GENERATE:
        optimized_motion_description, token_usage = optimize_motion_description_generate(
            user_prompt, model=_system_config.motion_description_model
        )
    elif mode == GenerationMode.REFINE:
        # Get base motion description
        base_motion_description = base_file.caption.get("motion_description")
        if base_motion_description is None:
            raise ValueError("Base file must have motion_description in caption for REFINE mode")

        optimized_motion_description, token_usage = optimize_motion_description_refine(
            base_motion_description, user_prompt, model=_system_config.motion_description_model
        )
    else:
        raise ValueError(f"Unsupported mode for motion description optimization: {mode}")

    return optimized_motion_description, model_description, token_usage


def _generate_animation(
    base_file: BaseFile,
    examples: list[InputAnimation],
    request: dict,
    config: Config,
    mode: GenerationMode = GenerationMode.GENERATE,
    num_retries: int = 2,
    progress_callback: Callable[[dict], None] | None = None,
    rig_render_dir: str | None = None,
) -> tuple[GeneratedAnimation, list[dict], dict]:
    """
    Generate an animation based on the provided base file and examples.

    Parameters:
        base_file (BaseFile): The base file containing object structure and optionally animation.
        examples (list[InputAnimation]): A list of InputAnimation objects containing example animations.
        request (dict): A dictionary containing the user's request details.
        config (Config): An instance of the Config class containing generation parameters.
        mode (GenerationMode): The mode of generation, either GENERATE with examples or REFINE.
        num_retries (int): The number of retries for the generation and parsing process in case of failure.
        rig_render_dir (str | None): Path to directory containing rig render images (view_*.png).
            When provided, images are sent to the model alongside the text prompts.

    Returns:
        tuple[GeneratedAnimation, list[dict], dict]: Generated animation, API context, and run_stats with
            "generation" and "motion_description" keys (time_seconds, token_usage each).
    """

    rig_render_paths: list[str] | None = None
    if rig_render_dir:
        rig_render_paths = sorted(glob.glob(os.path.join(rig_render_dir, "view_*.png")))
        if not rig_render_paths:
            logger.warning(f"No view_*.png files found in {rig_render_dir}, proceeding without rig renders")
            rig_render_paths = None

    gen_model = config.model or (
        _system_config.gen_model if mode != GenerationMode.REFINE else _system_config.refine_model
    )
    gen_temp = config.temperature or (
        _system_config.gen_temperature if mode != GenerationMode.REFINE else _system_config.refine_temperature
    )
    top_p = config.top_p or _system_config.top_p
    reasoning_effort = config.reasoning_effort or "medium"

    logger.info("Using system config: " + json.dumps(_system_config.to_dict()))
    logger.info("Using user config: " + json.dumps(config.to_dict()))

    has_rig_renders = rig_render_paths is not None and len(rig_render_paths) > 0
    system_prompt_str, user_prompt_str_list, user_request_str = build_prompt_strings(
        base_file, examples, request, mode, has_rig_renders=has_rig_renders
    )

    _emit(progress_callback, "stage", {"stage": STAGE_GENERATION})

    generation_time_seconds = 0.0
    generation_token_usage: dict | None = None
    generation_succeeded_on_attempt = 0

    for i in range(num_retries + 1):
        logger.info(f"Generating animation (attempt {i + 1} of {num_retries + 1})...")

        try:
            t0 = time.perf_counter()
            response_str, api_context, generation_token_usage = api_request(
                system_prompt_str,
                user_prompt_str_list,
                user_request_str,
                image_paths=rig_render_paths,
                use_code_interpreter=_system_config.use_code_interpreter,
                model=gen_model,
                temperature=gen_temp,
                top_p=top_p,
                mode=mode,
                reasoning_effort=reasoning_effort,
            )
            generation_time_seconds = time.perf_counter() - t0

            generated_animation = parse_generated_animation(
                response_str,
                base_file.metadata,  # Use base file metadata for generation
            )

            generation_succeeded_on_attempt = i + 1
            break
        except ValueError as e:
            logger.warning(f"Incorrect animation during generation or parsing attempt {i}: {e}\n")
            _emit(progress_callback, "message", {"level": "warning", "message": str(e)})
            continue
    else:
        logger.error(f"Failed to generate animation after {num_retries} retries.")
        _emit(
            progress_callback,
            "message",
            {"level": "error", "message": "Failed to generate animation after retries."},
        )
        raise ValueError("Failed to generate animation.")

    # Step 3: Optimize motion description
    _emit(progress_callback, "stage", {"stage": STAGE_MOTION_DESCRIPTION})
    t_motion = time.perf_counter()
    optimized_motion_description, model_description, motion_token_usage = _optimize_motion_description(
        base_file, request, mode
    )
    motion_description_time_seconds = time.perf_counter() - t_motion
    # Add motion and model descriptions to generated animation
    generated_animation.motion_description = optimized_motion_description
    generated_animation.model_description = model_description

    run_stats = {
        "generation": {
            "time_seconds": generation_time_seconds,
            "token_usage": generation_token_usage,
            "succeeded_on_attempt": generation_succeeded_on_attempt,
        },
        "motion_description": {
            "time_seconds": motion_description_time_seconds,
            "token_usage": motion_token_usage,
        },
    }
    return generated_animation, api_context, run_stats


def generate_animation(
    dest_path: str,
    request: dict,
    base_file: AnimationFile | None = None,
    animation_examples: list[AnimationFile] | None = None,
    mode: GenerationMode = GenerationMode.GENERATE,
    config: Config | None = None,
    logs_path: str = f"./logs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    metadata_path: str | None = None,
    render_rig: bool = False,
    render_video: bool = False,
    render_path: str | None = None,
    produce_gltf: bool = False,
    gltf_path: str | None = None,
    auto_select_examples: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """
    Generate an animation based on a base file and optional examples.

    Parameters:
        dest_path (str): Path where the generated animation will be saved.
        request (dict): User request containing description and other parameters.
        base_file (AnimationFile | None): Base file. Use AnimationFile.from_example(name) to create from example name.
                                         Can be None in GENERATE mode if auto_select_examples is True.
        animation_examples (list[AnimationFile] | None): Optional list of example animation files.
                                                       Use AnimationFile.from_example(name) to create from example name.
        mode (GenerationMode): GENERATE or REFINE mode.
        config (Config): Generation configuration.
        logs_path (str): Path to log file.
        metadata_path (str | None): Optional path to save metadata JSON.
        render_rig (bool): Whether to render rig overlay views of the input model.
        render_video (bool): Whether to render video.
        render_path (str | None): Path for rendered video.
        produce_gltf (bool): Whether to convert output USD to GLB.
        gltf_path (str | None): Path for converted GLB.
        auto_select_examples (bool): Whether to auto-select examples.

    Returns:
        dict: Dictionary with 'filepath' and 'metadata' keys.

    Raises:
        ValueError: If validation fails.
    """
    setup_loggers(logs_path)

    if config is None:
        config = Config()

    result_filepath = ""
    metadata: dict[str, Any] = {}

    try:
        # TODO: add validation for base_file and animation_examples

        _emit(progress_callback, "stage", {"stage": STAGE_VALIDATION})
        _validate_request(request)
        _validate_config(config)

        if animation_examples is None:
            animation_examples = []

        total_start = time.perf_counter()
        selection_time_seconds = 0.0
        selection_token_usage = None

        # Handle base_file validation and "first example as base" for generate with examples
        if base_file is None:
            if mode == GenerationMode.REFINE:
                raise ValueError("base_file must be provided in REFINE mode")
            if not auto_select_examples:
                if animation_examples:
                    # Use first example as base, rest remain as examples
                    base_file = animation_examples[0]
                    animation_examples = list(animation_examples[1:])
                else:
                    raise ValueError(
                        "base_file must be provided unless auto_select_examples is True or example files are provided"
                    )

        selected_example_names: list[str] = []
        selection_reasoning = ""

        # Handle auto-select examples
        if auto_select_examples:
            _emit(progress_callback, "stage", {"stage": STAGE_EXAMPLE_SELECTION})
            logger.info("Auto-selecting additional examples...")
            t_sel = time.perf_counter()
            selected_examples, selection_token_usage, selected_example_names, selection_reasoning = select_examples(
                request.get("description", ""), model=_system_config.selection_model
            )
            selection_time_seconds = time.perf_counter() - t_sel

            # If base_file is None, treat first selected example as base_file
            if base_file is None:
                if len(selected_examples) == 0:
                    logger.warning(
                        "No examples were selected by auto-selection. Falling back to predefined base example: bounce."
                    )
                    base_file = AnimationFile.from_example("bounce")
                else:
                    base_file = selected_examples[0]

            animation_examples.extend(selected_examples)

        # Prepare base file (handles GENERATE vs REFINE mode logic)
        if base_file is None:
            raise ValueError("base_file must be provided or auto_select_examples must be True")

        _emit(progress_callback, "stage", {"stage": STAGE_KEYFRAME_SAMPLING_BASE})
        base = prepare_base_file(base_file, mode)

        rig_output_dir: str | None = None
        if render_rig:
            _emit(progress_callback, "stage", {"stage": STAGE_RIG_RENDERING})
            rig_output_dir = os.path.join(os.path.dirname(dest_path), "input_renders")
            os.makedirs(rig_output_dir, exist_ok=True)
            if not render_rig_overlay(
                input_path=base.filepath,
                output_dir=rig_output_dir,
                up_axis=base.metadata.up_axis,
            ):
                logger.error(f"Failed to render rig overlay for {base.filepath}")
                raise RuntimeError(f"Failed to render rig overlay for {base.filepath}")

        # Validate base file has animation_json in REFINE mode
        if mode == GenerationMode.REFINE and base.animation_json is None:
            raise ValueError("Base file must have animation in REFINE mode")

        # Prepare example animations
        _emit(progress_callback, "stage", {"stage": STAGE_KEYFRAME_SAMPLING_EXAMPLES})
        examples = []
        for example_file in animation_examples:
            examples.append(prepare_animation_example(example_file))

        # Joint name cleanup: rename verbose joint names before prompting
        _emit(progress_callback, "stage", {"stage": STAGE_JOINT_CLEANUP})
        t_cleanup = time.perf_counter()
        forward_map, reverse_map, cleanup_token_usage = cleanup_joint_names(base.metadata.joint_names)
        cleanup_time_seconds = time.perf_counter() - t_cleanup

        is_identity = all(k == v for k, v in forward_map.items())
        if not is_identity:
            original_paths = list(base.metadata.joint_names)
            cleaned_paths = apply_joint_name_mapping(original_paths, forward_map)
            full_path_fwd = dict(zip(original_paths, cleaned_paths, strict=True))

            base.metadata.joint_names = cleaned_paths
            base.object_json = rename_json_keys(base.object_json, full_path_fwd)
            if base.animation_json is not None:
                base.animation_json = rename_json_keys(base.animation_json, full_path_fwd)

            for ex in examples:
                ex_original = list(ex.metadata.joint_names)
                ex_cleaned = apply_joint_name_mapping(ex_original, forward_map)
                ex_fwd = dict(zip(ex_original, ex_cleaned, strict=True))
                ex.metadata.joint_names = ex_cleaned
                ex.animation_json = rename_json_keys(ex.animation_json, ex_fwd)

        gen_animation, api_context, run_stats = _generate_animation(
            base_file=base,
            examples=examples,
            request=request,
            config=config,
            mode=mode,
            num_retries=_system_config.num_retries,
            progress_callback=progress_callback,
            rig_render_dir=rig_output_dir,
        )

        # Reverse-map joint names back to originals for USD writing
        if not is_identity:
            restored_paths = apply_joint_name_mapping(gen_animation.joint_names, reverse_map)
            full_path_rev = dict(zip(gen_animation.joint_names, restored_paths, strict=True))
            gen_animation.joint_names = restored_paths
            gen_animation.animation_json = rename_json_keys(gen_animation.animation_json, full_path_rev)

            # Restore base metadata so save_generated_animation uses original names
            base.metadata.joint_names = original_paths
            base.object_json = rename_json_keys(base.object_json, {v: k for k, v in full_path_fwd.items()})
            if base.animation_json is not None:
                base.animation_json = rename_json_keys(base.animation_json, {v: k for k, v in full_path_fwd.items()})

        total_time_seconds = time.perf_counter() - total_start

        timing_stats: dict[str, float] = {
            "total_seconds": round(total_time_seconds, 2),
            "selection_seconds": round(selection_time_seconds if auto_select_examples else 0.0, 2),
            "cleanup_seconds": round(cleanup_time_seconds, 2),
            "generation_seconds": round(run_stats["generation"]["time_seconds"], 2),
            "motion_description_seconds": round(run_stats["motion_description"]["time_seconds"], 2),
        }
        token_usage: dict[str, dict | None] = {
            "generation": run_stats["generation"]["token_usage"],
            "motion_description": run_stats["motion_description"]["token_usage"],
            "cleanup": cleanup_token_usage,
        }
        if auto_select_examples:
            token_usage["selection"] = selection_token_usage

        _emit(progress_callback, "stage", {"stage": STAGE_SAVING})
        result_filepath, metadata = save_generated_animation(
            gen_animation,
            request,
            base_file=base,
            animation_examples=examples,
            gen_anim_path=dest_path,
            config=config,
            mode=mode,
            auto_select_examples=auto_select_examples,
            timing_stats=timing_stats,
            token_usage=token_usage,
            selected_example_names=selected_example_names if auto_select_examples else None,
            selection_reasoning=selection_reasoning if auto_select_examples else None,
            generation_succeeded_on_attempt=run_stats["generation"]["succeeded_on_attempt"],
        )

        if metadata_path is not None:
            write_metadata_file(metadata, metadata_path)

        if render_video:
            _emit(progress_callback, "stage", {"stage": STAGE_RENDERING})
            if render_path is None:
                render_path = os.path.splitext(result_filepath)[0] + ".mp4"
            if not render_animation(gen_animation, result_filepath, render_path):
                logger.error(f"Failed to render animation from {result_filepath} to {render_path}")
                raise RuntimeError(f"Failed to render animation from {result_filepath} to {render_path}")

        if produce_gltf:
            _emit(progress_callback, "stage", {"stage": STAGE_GLTF_CONVERSION})
            if gltf_path is None:
                gltf_path = os.path.splitext(result_filepath)[0] + ".glb"
            if not convert_animation_to_glb(
                result_filepath,
                gltf_path,
                fps=gen_animation.fps,
                up_axis=base.metadata.up_axis,
            ):
                logger.error(f"Failed to convert animation from {result_filepath} to {gltf_path}")
                raise RuntimeError(f"Failed to convert animation from {result_filepath} to {gltf_path}")

    except Exception as e:
        logger.error(f"Error during animation generation: {e}\n{traceback.format_exc()}")
        _emit(progress_callback, "message", {"level": "error", "message": str(e)})
        raise RuntimeError(f"Failed to generate animation: {e}") from e

    result = {
        "filepath": result_filepath,
        "metadata": metadata,
    }

    return result
