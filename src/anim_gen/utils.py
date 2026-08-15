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

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

from .data_structs import GeneratedAnimation, ModelMetadata
from .quat_utils import make_euler_from_quat

logger = logging.getLogger(__name__)


def _compose_axis_remaps(
    first: tuple[tuple[int, int, int], tuple[int, int, int]],
    second: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Compose two axis remaps in execution order: apply first, then second.
    """
    first_order, first_scalars = first
    second_order, second_scalars = second
    composed_order = tuple(first_order[idx] for idx in second_order)
    composed_scalars = tuple(second_scalars[i] * first_scalars[second_order[i]] for i in range(3))
    return composed_order, composed_scalars  # type: ignore


def _get_single_axis_remap(axis_idx: int, angle: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Return remap for a single axis Euler angle (axis_idx: 0=x,1=y,2=z).
    """
    if angle == 0:
        return (0, 1, 2), (1, 1, 1)
    if axis_idx == 0:
        if angle == 90:
            return (0, 2, 1), (1, -1, 1)
        if angle == -90:
            return (0, 2, 1), (1, 1, -1)
        if abs(angle) == 180:
            return (0, 1, 2), (1, -1, -1)
    elif axis_idx == 1:
        if angle == 90:
            return (2, 1, 0), (1, 1, -1)
        if angle == -90:
            return (2, 1, 0), (-1, 1, 1)
        if abs(angle) == 180:
            return (0, 1, 2), (-1, 1, -1)
    elif axis_idx == 2:
        if angle == 90:
            return (1, 0, 2), (-1, 1, 1)
        if angle == -90:
            return (1, 0, 2), (1, -1, 1)
        if abs(angle) == 180:
            return (0, 1, 2), (-1, -1, 1)

    logger.error("Please align the model to have Z-up or Y-up axis.")
    raise ValueError("Please align the model to have Z-up or Y-up axis.")


def _snap_supported_axis_angle(angle: float) -> int | None:
    """
    Snap one Euler axis angle to supported values {0, +/-90, +/-180}.
    Returns None if no supported snap is close enough.
    """
    if np.isclose(angle, 0, atol=5):
        return 0
    if np.isclose(abs(angle), 90, atol=5):
        return int(np.sign(angle)) * 90
    if np.isclose(abs(angle), 180, atol=5):
        return int(np.sign(angle)) * 180 if angle != 0 else 180
    return None


def get_fixed_axis_order(quat):
    """
    Returns the fixed axis order and sign flips from root rotation.
    Supports consecutive per-axis remaps for multi-axis right-angle rotations.
    """
    euler_angles = make_euler_from_quat(quat)
    snapped_angles: list[int] = []
    for axis_idx, angle in enumerate(euler_angles):
        snapped = _snap_supported_axis_angle(float(angle))
        if snapped is None:
            logger.warning(
                f"Axis {axis_idx} angle {angle:.4f} is not close to 0/90/180 degrees; this axis remap is ignored."
            )
            snapped = 0
        snapped_angles.append(snapped)

    clamped_euler_angles = tuple(snapped_angles)
    logger.debug(f"Clamped euler angles: {clamped_euler_angles}")

    remap: tuple[tuple[int, int, int], tuple[int, int, int]] = ((0, 1, 2), (1, 1, 1))
    for axis_idx, snapped_angle in enumerate(clamped_euler_angles):
        axis_remap = _get_single_axis_remap(axis_idx, snapped_angle)
        remap = _compose_axis_remaps(remap, axis_remap)

    return remap


def get_skel_rotation(skelPrim, time_codes=None):
    """
    Returns the world-space rotation (as a quaternion) of the skeleton prim at skelPath,
    using its ancestor transforms (not including its own xform ops).
    In case of variable transformations, raises an error.
    """
    # Use an XformCache to evaluate the local‐to‐world matrix of the skeleton prim
    if time_codes is None:
        time_codes = [Usd.TimeCode(0)]

    xform_cache = UsdGeom.XformCache()
    translations = np.zeros((len(time_codes), 3))
    quats = np.zeros((len(time_codes), 4))
    scales = np.zeros((len(time_codes), 3))

    for i, time in enumerate(time_codes):
        xform_cache.SetTime(time)
        transforms = xform_cache.GetParentToWorldTransform(skelPrim)
        translations[i], quat, scales[i] = UsdSkel.DecomposeTransform(transforms)
        quats[i] = np.array([quat.GetReal(), *quat.GetImaginary()])

    # check if all transformations remain constant during the animation
    if (
        not np.allclose(translations, translations[0], rtol=1e-4)
        or not np.allclose(quats, quats[0], rtol=1e-3)
        or not np.allclose(scales, scales[0], rtol=1e-4)
    ):
        # max_tr_delta = np.max(np.abs(translations - translations[0]))
        # max_rot_delta = np.max(np.abs(quats - quats[0]))
        # max_scale_delta = np.max(np.abs(scales - scales[0]))
        logger.error("Transformations are not constant during the animation.")
        raise ValueError("Transformations are not constant during the animation.")

    return Gf.Quatf(quats[0][0], quats[0][1], quats[0][2], quats[0][3])


def _apply_bind_transforms(translations, rotations, scales, joint_local_bind_xforms):
    num_joints = len(joint_local_bind_xforms)

    _, _, bind_scales = UsdSkel.DecomposeTransforms(joint_local_bind_xforms)
    bind_scales_np = np.array(bind_scales)

    for i in range(translations.shape[0]):
        for j in range(num_joints):
            xform_matrix = Gf.Matrix4d()

            if not np.any(np.isnan(translations[i][j])):
                xform_matrix.SetTranslate(Gf.Vec3d(*translations[i][j]))

            joint_local_bind_combined_translation = xform_matrix * joint_local_bind_xforms[j]

            translations[i][j] = joint_local_bind_combined_translation.ExtractTranslation()

            xform_matrix = Gf.Matrix4d()

            if not np.any(np.isnan(rotations[i][j])):
                xform_matrix.SetRotate(
                    Gf.Quatd(rotations[i][j][0], Gf.Vec3d(rotations[i][j][1], rotations[i][j][2], rotations[i][j][3]))
                )

            joint_local_bind_combined_rotation = xform_matrix * joint_local_bind_xforms[j]

            rotations[i][j] = np.array(
                [
                    joint_local_bind_combined_rotation.ExtractRotationQuat().GetReal(),
                    *joint_local_bind_combined_rotation.ExtractRotationQuat().GetImaginary(),
                ]
            )

            if not np.any(np.isnan(scales[i][j])):
                scales[i][j] = scales[i][j] * bind_scales_np[j]


def _remap_xyz_components(values: np.ndarray, fixed_axis_order: tuple[int, int, int]) -> np.ndarray:
    """Remap XYZ components using the fixed axis order."""
    return values[:, :, fixed_axis_order]


def _remap_translations(
    translations: np.ndarray, fixed_axis_order: tuple[int, int, int], scalars: tuple[int, int, int]
) -> np.ndarray:
    """Remap translation axes and flip the requested axes."""
    return _remap_xyz_components(translations, fixed_axis_order) * np.asarray(scalars, dtype=translations.dtype)  # type: ignore


def _remap_rotations_quat(
    rotations: np.ndarray, fixed_axis_order: tuple[int, int, int], scalars: tuple[int, int, int]
) -> np.ndarray:
    """Remap quaternion vector components as [w, x, y, z] -> [w, reordered_xyz] and flip requested axes."""
    rotation_axis_order = (0, *(axis + 1 for axis in fixed_axis_order))
    remapped = rotations[:, :, rotation_axis_order]
    remapped[:, :, 1:] = remapped[:, :, 1:] * np.asarray(scalars, dtype=rotations.dtype)
    return remapped


def _remap_rotations_euler(
    rotations: np.ndarray, fixed_axis_order: tuple[int, int, int], scalars: tuple[int, int, int]
) -> np.ndarray:
    """Remap Euler XYZ components and flip requested axes."""
    return _remap_xyz_components(rotations, fixed_axis_order) * np.asarray(scalars, dtype=rotations.dtype)  # type: ignore


def remove_blendshapes_from_stage(stage: Usd.Stage) -> None:
    """
    Remove all blendshapes from a USD stage.

    This function removes:
    - BlendShapeWeightsAttr from all animation prims
    - All BlendShape prims
    - Blend shape relationships from skeleton binding API

    Parameters:
        stage (Usd.Stage): The USD stage to remove blendshapes from.
    """
    # Remove BlendShapeWeightsAttr from all animation prims
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Animation):
            anim = UsdSkel.Animation.Get(stage, prim.GetPath())
            if anim:
                blend_shape_weights_attr = anim.GetBlendShapeWeightsAttr()
                if blend_shape_weights_attr.IsAuthored():
                    blend_shape_weights_attr.Clear()
                    logger.debug(f"Removed BlendShapeWeightsAttr from animation prim: {prim.GetPath()}")

    # Remove any BlendShape prims
    blendshape_prims_to_remove = []
    for prim in stage.Traverse():
        # Check if it's a BlendShape prim by checking the type name
        type_name = prim.GetTypeName()
        if type_name == "BlendShape" or "BlendShape" in type_name:
            blendshape_prims_to_remove.append(prim.GetPath())

    for path in blendshape_prims_to_remove:
        stage.RemovePrim(path)
        logger.debug(f"Removed BlendShape prim: {path}")

    # Clear blend shape references from skeleton binding API if present
    skel_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Skeleton):
            skel_prim = prim
            break

    if skel_prim:
        # Clear blend shape relationships if they exist
        try:
            binding_api = UsdSkel.BindingAPI(skel_prim)
            if binding_api:
                blend_shapes_rel = binding_api.GetBlendShapesRel()
                if blend_shapes_rel and blend_shapes_rel.IsAuthored():
                    blend_shapes_rel.ClearTargets(True)
                    logger.debug(f"Cleared blend shapes relationship from skeleton: {skel_prim.GetPath()}")

                blend_shape_targets_rel = binding_api.GetBlendShapeTargetsRel()
                if blend_shape_targets_rel and blend_shape_targets_rel.IsAuthored():
                    blend_shape_targets_rel.ClearTargets(True)
                    logger.debug(f"Cleared blend shape targets relationship from skeleton: {skel_prim.GetPath()}")
        except Exception as e:
            # BindingAPI might not be available or prim might not have it applied
            logger.debug(f"Could not access BindingAPI for skeleton {skel_prim.GetPath()}: {e}")


def apply_modified_animation(
    src_filepath: str,
    dst_filepath: str,
    animation_name: str,
    generated_animation: GeneratedAnimation,
    motion_description: str | None = None,
    model_description: str | None = None,
    anim_gen_version: str | None = None,
):
    """
    Apply modified animation to a USD file and write custom metadata.

    Parameters:
        src_filepath (str): Path to source USD file.
        dst_filepath (str): Path to destination USD file.
        animation_name (str): Name for the animation prim.
        generated_animation (GeneratedAnimation): The generated animation data.
        motion_description (str | None): Motion description to write to custom metadata.
        model_description (str | None): Model description to write to custom metadata.
        anim_gen_version (str | None): Library version to write; if None, will not set autogenerated/version.
    """
    joint_names = generated_animation.joint_names
    translations = generated_animation.translations.astype(np.float64)
    rotations = generated_animation.rotations.astype(np.float64)
    scales = generated_animation.scales.astype(np.float64)
    fps = generated_animation.fps
    animation_json = generated_animation.animation_json
    generated_datetime = generated_animation.datetime
    end_frame = translations.shape[0] - 1

    src_stage = Usd.Stage.Open(src_filepath)
    if not src_stage:
        raise RuntimeError(f"Failed to open source stage: {src_filepath}")

    stage = Usd.Stage.CreateNew(dst_filepath)
    stage.GetRootLayer().TransferContent(src_stage.GetRootLayer())
    stage.SetMetadata("timeCodesPerSecond", fps)
    stage.SetMetadata("startTimeCode", 0)
    stage.SetMetadata("endTimeCode", end_frame)

    # Remove all blendshapes from the stage
    remove_blendshapes_from_stage(stage)

    # Find the skeleton prim
    skel_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Skeleton):
            skel_prim = prim
            break

    if skel_prim is None:
        raise ValueError("No skeleton prim found in stage")

    animRel = skel_prim.GetRelationship("skel:animationSource")
    if animRel:
        animRel.ClearTargets(True)

    for child in skel_prim.GetChildren():
        if child.IsA(UsdSkel.Animation):
            stage.RemovePrim(child.GetPath())

    anim_path = skel_prim.GetPath().AppendChild(animation_name)
    anim = UsdSkel.Animation.Define(stage, anim_path)
    anim.GetJointsAttr().Set(Vt.TokenArray(joint_names))

    # Set custom metadata on the animation prim
    anim_prim = anim.GetPrim()
    # Convert animation_json to JSON string if it's a dict
    if isinstance(animation_json, dict):
        animation_json_str = json.dumps(animation_json)
    else:
        animation_json_str = str(animation_json)

    anim_prim.SetCustomDataByKey("animation_json", animation_json_str)
    anim_prim.SetCustomDataByKey("generated_datetime", generated_datetime)

    # Mark as autogenerated and store library version when version is provided
    if anim_gen_version is not None:
        anim_prim.SetCustomDataByKey("autogenerated", True)
        anim_prim.SetCustomDataByKey("anim_gen_version", anim_gen_version)

    # Write motion and model descriptions if provided
    if motion_description is not None:
        anim_prim.SetCustomDataByKey("motion_description", motion_description)
    if model_description is not None:
        anim_prim.SetCustomDataByKey("model_description", model_description)

    animRel.AddTarget(anim.GetPath())

    num_joints = len(joint_names)

    for t in range(translations.shape[0]):
        usd_time = Usd.TimeCode(t)

        trans_list = [Gf.Vec3f(*translations[t, j]) for j in range(num_joints)]
        rot_list = [Gf.Quatf(*rotations[t, j]) for j in range(num_joints)]
        scale_list = [Gf.Vec3h(*scales[t, j]) for j in range(num_joints)]

        vt_trans = Vt.Vec3fArray(trans_list)
        vt_rot = Vt.QuatfArray(rot_list)
        vt_scale = Vt.Vec3hArray(scale_list)

        anim.GetTranslationsAttr().Set(vt_trans, usd_time)
        anim.GetRotationsAttr().Set(vt_rot, usd_time)
        anim.GetScalesAttr().Set(vt_scale, usd_time)

    stage.GetRootLayer().Save()
    logger.info(f"Wrote modified animation '{animation_name}' to {dst_filepath}")


def read_custom_metadata(
    filepath: str,
) -> tuple[dict | None, str | None, str | None, str | None, bool, str | None]:
    """
    Read custom metadata from a USD file.

    Parameters:
        filepath (str): Path to the USD file.

    Returns:
        tuple: (animation_json, generated_datetime, motion_description, model_description,
                is_autogenerated, anim_gen_version).
        is_autogenerated is True only if the prim has "autogenerated" set to True.
        anim_gen_version is the string stored or None.
    """
    try:
        stage = Usd.Stage.Open(filepath)
    except Exception as e:
        logger.warning(f"Failed to open USD file to read custom metadata: {e}")
        return None, None, None, None, False, None

    # Find the animation prim
    anim_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Animation):
            anim_prim = prim
            break

    if not anim_prim:
        return None, None, None, None, False, None

    try:
        animation_json_str = anim_prim.GetCustomDataByKey("animation_json")
        generated_datetime = anim_prim.GetCustomDataByKey("generated_datetime")
        motion_description = anim_prim.GetCustomDataByKey("motion_description")
        model_description = anim_prim.GetCustomDataByKey("model_description")
        autogenerated_val = anim_prim.GetCustomDataByKey("autogenerated")
        anim_gen_version = anim_prim.GetCustomDataByKey("anim_gen_version")

        animation_json = None
        if animation_json_str is not None:
            try:
                animation_json = json.loads(animation_json_str)
            except json.JSONDecodeError:
                logger.warning("Failed to parse animation_json from custom metadata")

        is_autogenerated = bool(autogenerated_val) if autogenerated_val is not None else False
        version_str = str(anim_gen_version) if anim_gen_version is not None else None

        return (
            animation_json,
            generated_datetime,
            motion_description,
            model_description,
            is_autogenerated,
            version_str,
        )
    except (KeyError, Exception) as e:
        logger.warning(f"Failed to read custom metadata from USD file: {e}")
        return None, None, None, None, False, None


def parse_metadata_pxr(
    filepath: str, parse_transformations: bool = True, include_skel_transforms: bool = False
) -> ModelMetadata:
    try:
        stage = Usd.Stage.Open(filepath)
    except Exception as e:
        raise ValueError(f"Failed to open USD file: {e}") from e

    up_axis = str(UsdGeom.GetStageUpAxis(stage))

    skel_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Skeleton) and skel_prim is None:
            skel_prim = prim
        elif prim.IsA(UsdSkel.Skeleton) and skel_prim is not None:
            raise ValueError("Multiple skeleton prims found in the USD stage")

    if skel_prim is None:
        raise ValueError("No skeleton prim found in the USD stage")

    try:
        skel_cache = UsdSkel.Cache()
        skel = UsdSkel.Skeleton.Get(stage, skel_prim.GetPath())
    except Exception as e:
        raise ValueError(f"Failed to get skeleton from prim: {e}") from e

    try:
        skel_query = skel_cache.GetSkelQuery(skel)
        anim_query = skel_query.GetAnimQuery()
    except Exception as e:
        raise ValueError(f"Failed to create skeleton or animation query: {e}") from e

    try:
        times = anim_query.GetJointTransformTimeSamples()
        time_codes = [Usd.TimeCode(t) for t in times]
        time_codes_np = np.array([tc.GetValue() for tc in time_codes])
    except Exception as e:
        raise ValueError(f"Failed to get time samples: {e}") from e

    try:
        # rest_transforms = np.array(skel.GetRestTransformsAttr().Get(), dtype=float)
        bind_transforms = skel.GetBindTransformsAttr().Get()
    except Exception as e:
        raise ValueError(f"Failed to get bind transforms: {e}") from e

    try:
        joints_anim = list(anim_query.GetJointOrder())
        if not joints_anim:
            raise ValueError("Skeleton has no joints defined")
    except Exception as e:
        raise ValueError(f"Failed to get joint names: {e}") from e

    if len(joints_anim) != len(set(joints_anim)):
        raise ValueError("Joint names are not unique")

    try:
        topology = UsdSkel.Topology(joints_anim)
        valid, why_not = topology.Validate()
        if not valid:
            raise ValueError(f"Invalid skeleton topology: {why_not}")
    except Exception as e:
        raise ValueError(f"Failed to validate skeleton topology: {e}") from e

    try:
        start_frame = int(stage.GetMetadata("startTimeCode"))
        end_frame = int(stage.GetMetadata("endTimeCode"))
        fps = stage.GetMetadata("timeCodesPerSecond")
    except Exception as e:
        raise ValueError(f"Failed to get frame metadata: {e}") from e

    try:
        has_blend_shapes = False

        anim = UsdSkel.Animation.Get(stage, anim_query.GetPrim().GetPath())
        if anim and anim.GetBlendShapeWeightsAttr().IsAuthored():
            weights = anim.GetBlendShapeWeightsAttr()

            if weights:
                time_samples = weights.GetTimeSamples()

                for ts in time_samples:
                    values = weights.Get(ts)
                    for v in values:
                        if abs(v) > 1e-7:
                            has_blend_shapes = True
                            break

        if has_blend_shapes:
            logger.error("Blend shapes are not supported")
            # raise ValueError("Blend shapes are not supported")
    except Exception as e:
        raise ValueError(f"Failed to check for blend shapes: {e}") from e

    bind_transforms_np = np.array(bind_transforms, dtype=float)
    joint_local_bind_xforms = Vt.Matrix4dArray(bind_transforms_np.shape[0])

    try:
        for i in range(bind_transforms_np.shape[0]):
            parent_id = topology.GetParent(i)
            bind = Gf.Matrix4d(bind_transforms_np[i])
            if parent_id >= 0:
                parent_bind = Gf.Matrix4d(bind_transforms_np[parent_id])
                local_bind = bind * parent_bind.GetInverse()
            else:
                local_bind = bind
            joint_local_bind_xforms[i] = local_bind
    except Exception as e:
        raise ValueError(f"Failed to compute joint local bind transforms: {e}") from e

    if parse_transformations:
        translations = np.zeros((len(time_codes), len(joints_anim), 3))
        translations_bind_relative = np.zeros((len(time_codes), len(joints_anim), 3))
        skel_translations = np.zeros((len(time_codes), len(joints_anim), 3))
        rotations = np.zeros((len(time_codes), len(joints_anim), 4))
        rotations_euler = np.zeros((len(time_codes), len(joints_anim), 3))
        rotations_bind_relative = np.zeros((len(time_codes), len(joints_anim), 4))
        rotations_euler_bind_relative = np.zeros((len(time_codes), len(joints_anim), 3))
        skel_rotations = np.zeros((len(time_codes), len(joints_anim), 4))
        scales = np.zeros((len(time_codes), len(joints_anim), 3))
        skel_scales = np.zeros((len(time_codes), len(joints_anim), 3))
        scales_bind_relative = np.zeros((len(time_codes), len(joints_anim), 3))

        for si in range(len(time_codes)):
            try:
                joint_local_xforms = skel_query.ComputeJointLocalTransforms(time_codes[si], False)
            except Exception as e:
                raise ValueError(f"Failed to compute joint local transform components: {e}") from e

            try:
                joint_bind_relative_xforms = Vt.Matrix4dArray(len(joint_local_xforms))

                for j in range(len(joint_local_xforms)):
                    joint_bind_relative_xforms[j] = joint_local_xforms[j] * joint_local_bind_xforms[j].GetInverse()
            except Exception as e:
                raise ValueError(f"Failed to compute joint bind relative transforms: {e}") from e

            try:
                for arr, trs in [
                    ((translations, rotations, scales), UsdSkel.DecomposeTransforms(joint_local_xforms)),
                    (
                        (translations_bind_relative, rotations_bind_relative, scales_bind_relative),
                        UsdSkel.DecomposeTransforms(joint_bind_relative_xforms),
                    ),
                    (
                        (skel_translations, skel_rotations, skel_scales),
                        UsdSkel.DecomposeTransforms(skel_query.ComputeJointSkelTransforms(time_codes[si], False)),
                    ),
                ]:
                    arr[0][si] = np.array(trs[0])
                    arr[1][si] = np.array([[q.GetReal(), *q.GetImaginary()] for q in trs[1]])
                    arr[2][si] = np.array(trs[2])

            except Exception as e:
                raise ValueError(f"Failed to decompose joint transform components: {e}") from e

            try:
                for j in range(len(joint_local_xforms)):
                    rotations_euler[si, j] = make_euler_from_quat(Gf.Quatf(*rotations[si, j]))
                    rotations_euler_bind_relative[si, j] = make_euler_from_quat(
                        Gf.Quatf(*rotations_bind_relative[si, j])
                    )
            except Exception as e:
                raise ValueError(f"Failed to convert rotations to euler angles: {e}") from e

    try:
        skel_ancestor_rotation = get_skel_rotation(skel_prim, time_codes)
        logger.debug(f"Skel ancestor rotation: {make_euler_from_quat(skel_ancestor_rotation)}")
        fixed_axis_order, scalars = get_fixed_axis_order(skel_ancestor_rotation)
        logger.debug(f"Fixed axis order: {fixed_axis_order}, scalars: {scalars}")
    except Exception as e:
        raise ValueError(f"Failed to get fixed axis order: {e}") from e

    root_joint_index = next((i for i in range(len(joints_anim)) if topology.GetParent(i) < 0), 0)
    try:
        root_bind_quat = joint_local_bind_xforms[root_joint_index].ExtractRotationQuat()
        root_bind_quat_np = np.array([[[root_bind_quat.GetReal(), *root_bind_quat.GetImaginary()]]], dtype=float)
        root_bind_quat_np = _remap_rotations_quat(root_bind_quat_np, fixed_axis_order, scalars)
        bind_root_rotation = Gf.Quatf(*root_bind_quat_np[0, 0])
        logger.debug(f"Bind root rotation (after world remap): {make_euler_from_quat(bind_root_rotation)}")
        bind_fixed_axis_order, bind_scalars = get_fixed_axis_order(bind_root_rotation)
        logger.debug(f"Bind fixed axis order: {bind_fixed_axis_order}, bind scalars: {bind_scalars}")
    except Exception as e:
        raise ValueError(f"Failed to get bind root fixed axis order: {e}") from e

    logger.debug(f"Parsed joint names: {joints_anim}")

    metadata = ModelMetadata(
        joint_names=joints_anim,
        start_frame=start_frame,
        end_frame=end_frame,
        time_codes=time_codes_np,
        fps=fps,
        bind_xforms=bind_transforms,
        joint_local_bind_xforms=joint_local_bind_xforms,
        fixed_axis_order=fixed_axis_order,
        scalars=scalars,
        bind_fixed_axis_order=bind_fixed_axis_order,
        bind_scalars=bind_scalars,
        # bind_fixed_axis_order=(0, 1, 2),
        # bind_scalars=(1, 1, 1),
        root_joint_index=root_joint_index,
        up_axis=up_axis,
    )

    if parse_transformations:
        metadata.translations = _remap_translations(
            _remap_translations(translations, fixed_axis_order, scalars), bind_fixed_axis_order, bind_scalars
        )
        metadata.translations_bind_relative = _remap_translations(
            _remap_translations(translations_bind_relative, fixed_axis_order, scalars),
            bind_fixed_axis_order,
            bind_scalars,
        )
        metadata.rotations = _remap_rotations_quat(
            _remap_rotations_quat(rotations, fixed_axis_order, scalars),
            bind_fixed_axis_order,
            bind_scalars,
        )
        # unwrap the rotations_euler for each joint and axis on the whole range of the animation
        # to get the continuous motion of the angle values
        metadata.rotations_euler = np.unwrap(
            np.around(
                _remap_rotations_euler(
                    _remap_rotations_euler(rotations_euler, fixed_axis_order, scalars),
                    bind_fixed_axis_order,
                    bind_scalars,
                ),
                5,
            ),
            period=360,
            axis=0,
        )
        metadata.rotations_bind_relative = _remap_rotations_quat(
            _remap_rotations_quat(rotations_bind_relative, fixed_axis_order, scalars),
            bind_fixed_axis_order,
            bind_scalars,
        )
        metadata.rotations_euler_bind_relative = np.unwrap(
            np.around(
                _remap_rotations_euler(
                    _remap_rotations_euler(rotations_euler_bind_relative, fixed_axis_order, scalars),
                    bind_fixed_axis_order,
                    bind_scalars,
                ),
                5,
            ),
            period=360,
            axis=0,
        )
        metadata.scales = _remap_xyz_components(_remap_xyz_components(scales, fixed_axis_order), bind_fixed_axis_order)
        metadata.scales_bind_relative = _remap_xyz_components(
            _remap_xyz_components(scales_bind_relative, fixed_axis_order), bind_fixed_axis_order
        )

    if include_skel_transforms:
        metadata.skel_translations = _remap_translations(
            _remap_translations(skel_translations, fixed_axis_order, scalars), bind_fixed_axis_order, bind_scalars
        )
        metadata.skel_rotations = _remap_rotations_quat(
            _remap_rotations_quat(skel_rotations, fixed_axis_order, scalars), bind_fixed_axis_order, bind_scalars
        )
        metadata.skel_scales = _remap_xyz_components(
            _remap_xyz_components(skel_scales, fixed_axis_order), bind_fixed_axis_order
        )

    return metadata
