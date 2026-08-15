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

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from pxr import Gf, UsdSkel, Vt

from .config import get_system_config
from .quat_utils import make_euler_from_quat

_system_config = get_system_config()


@dataclass
class AnimationFile:
    """
    A dataclass representing an animation file with its path and caption.
    This is used throughout the library to represent input files.
    """

    path: str
    caption: dict | None = None

    def to_dict(self) -> dict:
        """
        Convert to dictionary representation.

        Returns:
            dict: Dictionary with 'path' and 'caption' keys.
        """
        return {"path": self.path, "caption": self.caption}

    @classmethod
    def from_dict(cls, data: dict) -> "AnimationFile":
        """
        Create AnimationFile from dictionary.

        Parameters:
            data (dict): Dictionary with 'path' and optional 'caption' keys.

        Returns:
            AnimationFile: AnimationFile instance.
        """
        return cls(path=data["path"], caption=data.get("caption"))

    @classmethod
    def from_example(cls, example_name: str) -> "AnimationFile":
        """
        Create AnimationFile from an internal example name.

        Parameters:
            example_name (str): Name of the internal example.

        Returns:
            AnimationFile: AnimationFile instance with path and caption.

        Raises:
            FileNotFoundError: If the example file or caption file doesn't exist.
        """
        from .generation.selection import example_to_animation_file

        return example_to_animation_file(example_name)


@dataclass
class ModelMetadata:
    """
    A class to hold metadata for a model.
    """

    # Skeleton
    joint_names: list

    # Bind transformations
    bind_xforms: Vt.Matrix4dArray
    joint_local_bind_xforms: Vt.Matrix4dArray

    # Animation details
    time_codes: np.ndarray
    start_frame: int
    end_frame: int
    fps: float
    up_axis: str

    fixed_axis_order: tuple[int, int, int]
    scalars: tuple[int, int, int]
    bind_fixed_axis_order: tuple[int, int, int]
    bind_scalars: tuple[int, int, int]
    root_joint_index: int

    # Transformations
    translations: np.ndarray | None = None
    rotations: np.ndarray | None = None
    rotations_euler: np.ndarray | None = None
    scales: np.ndarray | None = None

    # Bind relative transformations
    translations_bind_relative: np.ndarray | None = None
    rotations_bind_relative: np.ndarray | None = None
    rotations_euler_bind_relative: np.ndarray | None = None
    scales_bind_relative: np.ndarray | None = None

    # Skeleton-space transformations (optional)
    skel_translations: np.ndarray | None = None
    skel_rotations: np.ndarray | None = None
    skel_scales: np.ndarray | None = None

    def get_transformations(self) -> list[np.ndarray]:
        """
        Returns the transformations for the model.
        """
        if _system_config.bind_relative_transformations:
            translations = self.translations_bind_relative
            rotations = (
                self.rotations_euler_bind_relative if _system_config.use_euler_angles else self.rotations_bind_relative
            )
            scales = self.scales_bind_relative
            if translations is None or rotations is None or scales is None:
                raise ValueError("Metadata has missing bind relative transformations")
            return [translations, rotations, scales]
        else:
            translations = self.translations
            rotations = self.rotations_euler if _system_config.use_euler_angles else self.rotations
            scales = self.scales
            if translations is None or rotations is None or scales is None:
                raise ValueError("Metadata has missing transformations")
            return [translations, rotations, scales]

    def get_bind_transformations(self, world_space: bool | None = None) -> list[np.ndarray]:
        if world_space is None:
            world_space = _system_config.world_bind_transformations
        if world_space:
            bind_transforms = self.bind_xforms
        else:
            bind_transforms = self.joint_local_bind_xforms

        bind_translations, bind_rotations, bind_scales = UsdSkel.DecomposeTransforms(bind_transforms)
        bind_translations = np.array(bind_translations, dtype=float)
        bind_rotations = np.array([[q.GetReal(), *q.GetImaginary()] for q in bind_rotations], dtype=float)
        bind_rotations_euler = np.array([make_euler_from_quat(Gf.Quatf(*q)) for q in bind_rotations], dtype=float)
        bind_scales = np.array(bind_scales, dtype=float)

        # Apply world-axis remap followed by bind-root remap so bind data matches metadata-space transforms.
        bind_translations = bind_translations[:, self.fixed_axis_order] * np.asarray(self.scalars, dtype=float)
        bind_translations = bind_translations[:, self.bind_fixed_axis_order] * np.asarray(
            self.bind_scalars, dtype=float
        )

        bind_rotations = bind_rotations[:, (0, *(axis + 1 for axis in self.fixed_axis_order))]
        bind_rotations[:, 1:] = bind_rotations[:, 1:] * np.asarray(self.scalars, dtype=float)
        bind_rotations = bind_rotations[:, (0, *(axis + 1 for axis in self.bind_fixed_axis_order))]
        bind_rotations[:, 1:] = bind_rotations[:, 1:] * np.asarray(self.bind_scalars, dtype=float)

        bind_rotations_euler = bind_rotations_euler[:, self.fixed_axis_order] * np.asarray(self.scalars, dtype=float)
        bind_rotations_euler = bind_rotations_euler[:, self.bind_fixed_axis_order] * np.asarray(
            self.bind_scalars, dtype=float
        )

        bind_scales = bind_scales[:, self.fixed_axis_order]
        bind_scales = bind_scales[:, self.bind_fixed_axis_order]

        # Normalize the root bind rotation in metadata-space.
        bind_rotations[self.root_joint_index] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        bind_rotations_euler[self.root_joint_index] = np.array([0.0, 0.0, 0.0], dtype=float)

        return [
            bind_translations,
            bind_rotations_euler if _system_config.use_euler_angles else bind_rotations,
            bind_scales,
        ]


@dataclass
class BaseFile:
    """
    A class to hold a base animation with its metadata, object JSON, and animation JSON.
    """

    filepath: str
    metadata: ModelMetadata
    object_json: dict
    animation_json: dict | None = None  # only if intended to be used as a base for refinement
    caption: dict | None = None
    is_autogenerated: bool = False

    def to_dict(self, include_metadata: bool = True) -> dict:
        """
        Convert the BaseFile instance to a dictionary representation.
        """
        result: dict[str, Any] = {
            "filepath": self.filepath,
            "object_json": self.object_json,
            "caption": self.caption,
        }
        if include_metadata:
            result["metadata"] = asdict(self.metadata)
        return result


@dataclass
class InputAnimation:
    """
    A class to hold an animation example with its metadata, and animation JSON.
    """

    filepath: str
    metadata: ModelMetadata
    animation_json: dict
    keyframes: list[list[list[int]]]
    caption: dict | None = None
    is_autogenerated: bool = False

    def to_dict(self, include_metadata: bool = True) -> dict:
        """
        Convert the InputAnimation instance to a dictionary representation.

        Returns:
            dict: A dictionary containing the filename, metadata, object JSON, animation string, and caption.
        """
        result: dict[str, Any] = {
            "filepath": self.filepath,
            "animation_json": self.animation_json,
            "caption": self.caption,
        }
        if include_metadata:
            result["metadata"] = asdict(self.metadata)
        return result


@dataclass
class GeneratedAnimation:
    """
    A class to hold the generated animation data and metadata.
    """

    animation_json: dict
    translations: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    keyframes: list[list[list[int]]]
    joint_names: list[str]
    fps: float
    end_frame: int
    datetime: str
    motion_description: str | None = None
    model_description: str | None = None


@dataclass
class Config:
    """
    A class to hold the configuration for the animation generation process.
    """

    interpolation_type: str | None = None
    model: str | None = None # generation model
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    reasoning_effort: str | None = None

    @classmethod
    def from_dict(cls, data: dict | None = None) -> "Config":
        if data is None:
            return cls()
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
