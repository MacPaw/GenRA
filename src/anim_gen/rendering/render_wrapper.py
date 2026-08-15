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
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import get_system_config
from ..data_structs import GeneratedAnimation

logger = logging.getLogger(__name__)

RENDER_SCRIPT = Path(__file__).resolve().parent / "render.py"
CONVERT_SCRIPT = Path(__file__).resolve().parent / "convert.py"
RIG_RENDER_SCRIPT = Path(__file__).resolve().parent / "rig_rendering.py"
GIZMO_OVERLAY_SCRIPT = Path(__file__).resolve().parent / "gizmo_overlay.py"

_MACOS_BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"


def _resolve_blender_bin() -> str:
    """Return the Blender binary path: BLENDER_BIN env, then PATH lookup, then macOS default."""
    env = os.environ.get("BLENDER_BIN")
    if env:
        return env
    if shutil.which("blender"):
        return "blender"
    if platform.system() == "Darwin" and os.path.isfile(_MACOS_BLENDER_PATH):
        return _MACOS_BLENDER_PATH
    return "blender"


def render_animation(
    gen_animation: GeneratedAnimation,
    input_path,
    output_path,
    blender_bin: str | None = None,
    render_script=RENDER_SCRIPT,
    background: str | None = None,
) -> bool:
    if blender_bin is None:
        blender_bin = _resolve_blender_bin()

    logger.info(f"Rendering animation {input_path} to {output_path}")

    cmd = [
        blender_bin,
        "-b",
        "--log-level",
        "0",
        "-P",
        str(render_script),
        "--",
        "--usd",
        input_path,
        "--out",
        output_path,
        "--fps",
        str(int(gen_animation.fps)),
        "--frame_start",
        str(0),
        "--frame_end",
        str(gen_animation.end_frame + 1),
        "--step",
        "3",
        "--resx",
        "512",
        "--resy",
        "512",
        "--up_axis",
        "Z",
        "--margin",
        "0.8",
    ]

    if background is not None:
        cmd.extend(["--background", background])

    # Use Xvfb (virtual display) when requested (e.g. in Docker) to avoid EGL_BAD_MATCH headless errors
    if os.environ.get("BLENDER_USE_XVFB"):
        cmd = ["xvfb-run", "-a"] + cmd

    proc = subprocess.run(cmd, capture_output=True, text=True)
    status_code = proc.returncode
    process_stdout = proc.stdout
    process_stderr = proc.stderr
    process_output = "\n".join([s for s in [process_stdout, process_stderr] if s])

    # log the stderr
    if process_stderr:
        logger.warning(f"{process_stderr}")

    if status_code != 0:
        logger.warning(f"Error during rendering (exit code {status_code}):\n{process_output}")

    return status_code == 0


def convert_animation_to_glb(
    input_path,
    output_path,
    fps: float | int | None = None,
    up_axis: str | None = None,
    blender_bin: str | None = None,
    convert_script=CONVERT_SCRIPT,
) -> bool:
    if blender_bin is None:
        blender_bin = _resolve_blender_bin()

    logger.info(f"Converting animation {input_path} to {output_path}")

    cmd = [
        blender_bin,
        "-b",
        "--log-level",
        "0",
        "-P",
        str(convert_script),
        "--",
        "--usd",
        input_path,
        "--out",
        output_path,
        "--up_axis",
        up_axis,
    ]
    if fps is not None:
        cmd.extend(["--fps", str(int(fps))])

    if os.environ.get("BLENDER_USE_XVFB"):
        cmd = ["xvfb-run", "-a"] + cmd

    proc = subprocess.run(cmd, capture_output=True, text=True)
    status_code = proc.returncode
    process_stdout = proc.stdout
    process_stderr = proc.stderr
    process_output = "\n".join([s for s in [process_stdout, process_stderr] if s])

    if process_stderr:
        logger.warning(f"{process_stderr}")

    if status_code != 0:
        logger.warning(f"Error during GLB conversion (exit code {status_code}):\n{process_output}")

    return status_code == 0


def render_rig_overlay(
    input_path: str,
    output_dir: str,
    up_axis: str = "Z",
    view_angle_offset: float | None = None,
    render_bones: bool | None = None,
    hide_view_normal_axis: bool | None = None,
    blender_bin: str | None = None,
) -> bool:
    """Render rig overlay views of an input model with 2D gizmo overlays.

    Two-step process:
      1. Blender subprocess runs rig_rendering.py -> multi-view PNGs + views_meta.json
      2. Python subprocess runs gizmo_overlay.py -> overlays 2D gizmos onto PNGs in-place

    Returns True on success, False on failure.
    """
    cfg = get_system_config()
    if view_angle_offset is None:
        view_angle_offset = cfg.view_angle_offset
    if render_bones is None:
        render_bones = cfg.render_rig_bones
    if hide_view_normal_axis is None:
        hide_view_normal_axis = cfg.hide_view_normal_gizmo_axis

    if blender_bin is None:
        blender_bin = _resolve_blender_bin()

    logger.info(f"Rendering rig overlay for {input_path} into {output_dir}")

    # Step 1: Blender rig rendering
    blender_cmd = [
        blender_bin,
        "-b",
        "--log-level",
        "0",
        "-P",
        str(RIG_RENDER_SCRIPT),
        "--",
        "--usd",
        input_path,
        "--out",
        output_dir,
        "--up_axis",
        up_axis,
        "--view_angle_offset",
        str(view_angle_offset),
        "--frame_start",
        "0",
    ]
    if render_bones:
        blender_cmd.append("--render_bones")

    if os.environ.get("BLENDER_USE_XVFB"):
        blender_cmd = ["xvfb-run", "-a"] + blender_cmd

    proc = subprocess.run(blender_cmd, capture_output=True, text=True)
    if proc.stderr:
        logger.warning(f"Blender rig render stderr:\n{proc.stderr}")
    if proc.returncode != 0:
        output = "\n".join([s for s in [proc.stdout, proc.stderr] if s])
        logger.warning(f"Blender rig render failed (exit code {proc.returncode}):\n{output}")
        return False

    # Step 2: Python gizmo overlay
    overlay_cmd = [
        sys.executable,
        str(GIZMO_OVERLAY_SCRIPT),
        "--images_dir",
        output_dir,
    ]
    overlay_cmd.append("--hide-view-normal-axis" if hide_view_normal_axis else "--no-hide-view-normal-axis")

    proc = subprocess.run(overlay_cmd, capture_output=True, text=True)
    if proc.stderr:
        logger.warning(f"Gizmo overlay stderr:\n{proc.stderr}")
    if proc.returncode != 0:
        output = "\n".join([s for s in [proc.stdout, proc.stderr] if s])
        logger.warning(f"Gizmo overlay failed (exit code {proc.returncode}):\n{output}")
        return False

    logger.info(f"Rig overlay rendering completed successfully: {output_dir}")
    return True
