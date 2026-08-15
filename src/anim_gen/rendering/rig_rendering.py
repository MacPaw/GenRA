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

# Blender 4.5.x
# Render rigged USDA to a single PNG with a visible skeleton overlay

import json
import math
import os
import sys

import bpy
import mathutils
from mathutils import Vector

# ----------------------------- CLI parsing -----------------------------------


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    import argparse

    p = argparse.ArgumentParser(description="Render USD first frame with skeleton overlay as PNG (background-safe)")
    p.add_argument("--usd", required=True, help="Path to .usd/.usda/.usdc")
    p.add_argument("--out", required=True, help="Output image path (.png recommended)")
    p.add_argument("--resx", type=int, default=512)
    p.add_argument("--resy", type=int, default=512)
    p.add_argument("--frame_start", type=int, default=0, help="Start frame (default: 0)")
    p.add_argument("--margin", type=float, default=0.9, help="Fill fraction of viewport (0-1)")
    p.add_argument("--up_axis", type=str, default="Z", help="Up axis of the input USD (Y or Z)")
    p.add_argument("--bone_radius", type=float, default=0.005, help="Radius of bone cylinders (scene units)")
    p.add_argument("--joint_radius", type=float, default=0.009, help="Radius of joint spheres (scene units)")
    p.add_argument("--render_bones", action="store_true", default=False, help="Draw bone cylinders between joints")
    p.add_argument("--mesh_alpha", type=float, default=1.0, help="0=fully transparent, 1=opaque")
    p.add_argument(
        "--gizmo_3d", action="store_true", default=False, help="Enable 3D axis gizmo in the scene (disabled by default)"
    )
    p.add_argument(
        "--view_angle_offset",
        type=float,
        default=0.0,
        help="Azimuth offset in degrees around the up axis (e.g. 20 for a slight 3D perspective)",
    )
    return p.parse_args(argv)


# ----------------------------- Utilities -------------------------------------


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def ensure_engine():
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_WORKBENCH"
    sh = scn.display.shading
    sh.light = "MATCAP"  # nice soft light
    sh.color_type = "OBJECT"  # use object colors
    sh.show_backface_culling = False
    scn.view_settings.view_transform = "Standard"


def setup_render(frame_start, resx, resy):
    scn = bpy.context.scene
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGBA"
    scn.render.resolution_x = resx
    scn.render.resolution_y = resy
    scn.frame_set(frame_start)

    if scn.world is None:
        scn.world = bpy.data.worlds.new("World")
    scn.world.color = (1, 1, 1)

    scn.use_nodes = False
    scn.render.use_sequencer = False


def import_usd(path):
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.usd_import(filepath=path, create_collection=True, import_cameras=False, import_lights=False)
    bpy.context.view_layer.update()
    after = set(bpy.data.objects.keys())
    new_names = list(after - before)
    return [bpy.data.objects[n] for n in new_names]


def depsgraph():
    return bpy.context.evaluated_depsgraph_get()


def compute_aabb(objs):
    inf = float("inf")
    bb_min = Vector((inf, inf, inf))
    bb_max = Vector((-inf, -inf, -inf))

    dg = depsgraph()

    def expand_with_mesh(eval_obj, world_matrix):
        nonlocal bb_min, bb_max
        if eval_obj.type != "MESH":
            return
        me = eval_obj.to_mesh(depsgraph=dg)
        if me is None:
            return
        for v in me.vertices:
            w = world_matrix @ v.co
            bb_min.x = min(bb_min.x, w.x)
            bb_max.x = max(bb_max.x, w.x)
            bb_min.y = min(bb_min.y, w.y)
            bb_max.y = max(bb_max.y, w.y)
            bb_min.z = min(bb_min.z, w.z)
            bb_max.z = max(bb_max.z, w.z)
        eval_obj.to_mesh_clear()

    dg.update()
    for inst in dg.object_instances:
        base = inst.object
        if base.original not in objs and base not in objs:
            continue
        eval_obj = base.evaluated_get(dg)
        expand_with_mesh(eval_obj, inst.matrix_world)

    return bb_min, bb_max


def axis_permutation_for_view(half_extents, view_axis, up_axis):
    hx, hy, hz = half_extents.x, half_extents.y, half_extents.z

    up = up_axis.upper()
    v = view_axis.upper()

    if up == "Z":
        if v in ["+Y", "-Y"]:
            return hx, hz, hy
        if v in ["+X", "-X"]:
            return hz, hy, hx

    if up == "Y":
        if v in ["+Z", "-Z"]:
            return hx, hy, hz
        if v in ["+X", "-X"]:
            return hz, hy, hx

    if up == "X":
        if v in ["+Y", "-Y"]:
            return hx, hz, hy
        if v in ["+Z", "-Z"]:
            return hx, hy, hz

    raise ValueError(f"Unsupported combination up={up_axis}, view={view_axis}")


def direction_vector(view_axis):
    if view_axis == "+X":
        return Vector((-1, 0, 0))
    if view_axis == "-X":
        return Vector((1, 0, 0))
    if view_axis == "+Y":
        return Vector((0, -1, 0))
    if view_axis == "-Y":
        return Vector((0, 1, 0))
    if view_axis == "+Z":
        return Vector((0, 0, -1))
    if view_axis == "-Z":
        return Vector((0, 0, 1))
    raise ValueError(f"Bad view axis: {view_axis}")


def compute_camera_distance(cam_data, half_w, half_h, half_d, margin):
    camera_angle = max(cam_data.angle_x, cam_data.angle_y)

    dist_x = half_w / (math.tan(camera_angle * 0.5) * margin)
    dist_y = half_h / (math.tan(camera_angle * 0.5) * margin)

    dist_frame = max(dist_x, dist_y)
    return dist_frame + half_d


def make_camera(view_axis):
    cam_data = bpy.data.cameras.new(f"Cam_{view_axis}")
    cam = bpy.data.objects.new(f"Cam_{view_axis}", cam_data)
    bpy.context.collection.objects.link(cam)

    cam_data.lens = 100.0
    cam_data.sensor_width = 36
    return cam


def position_camera(cam, view_axis, dist, azimuth_deg=0.0, up_vector=None):
    dir_vec = direction_vector(view_axis)
    pos = dir_vec * dist
    if azimuth_deg and up_vector:
        rot = mathutils.Matrix.Rotation(math.radians(azimuth_deg), 3, up_vector)
        pos = rot @ pos
    cam.location = pos
    view_vec = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = view_vec.to_track_quat("-Z", "Y").to_euler()
    return cam


def out_with_suffix(path, suffix):
    return f"{path}/{suffix}.png"


def filter_unbound_meshes(imported):
    keep = []
    skeleton = None
    for obj in imported:
        if obj.type == "ARMATURE":
            skeleton = obj
            keep.append(obj)
            break
    if not skeleton:
        raise RuntimeError("No Skeleton (Armature) found in USD import")

    for obj in imported:
        if obj.type == "MESH":
            if obj.parent == skeleton:
                keep.append(obj)
                continue
            for mod in obj.modifiers:
                if mod.type == "ARMATURE" and mod.object == skeleton:
                    keep.append(obj)
                    break
        else:
            keep.append(obj)

    for obj in imported:
        if obj not in keep:
            print(f"[INFO] Removing imported object: {obj.name} ({obj.type})")
            bpy.data.objects.remove(obj, do_unlink=True)

    return skeleton, keep


def build_transform_hierarchy(imported_objs):
    root = bpy.data.objects.new("ROOT_SCALE", None)
    root.empty_display_type = "PLAIN_AXES"
    bpy.context.collection.objects.link(root)

    centerer = bpy.data.objects.new("CENTER_TRANSLATE", None)
    centerer.empty_display_type = "PLAIN_AXES"
    bpy.context.collection.objects.link(centerer)
    centerer.parent = root

    for o in imported_objs:
        if o.parent is None:
            o.parent = centerer
    return root, centerer


def apply_normalization(root, centerer, bb_min, bb_max):
    size = bb_max - bb_min
    max_dim = max(size.x, size.y, size.z)
    if max_dim < 1e-12:
        max_dim = 1.0
    scale = 1.0 / max_dim
    center = (bb_min + bb_max) * 0.5

    root.scale = (scale, scale, scale)
    centerer.location = -center

    size_n = size * scale
    half_n = size_n * 0.5
    return half_n, center, scale


def replace_mesh_materials(mesh_objs, alpha):
    for obj in mesh_objs:
        obj.color = (0.9, 0.9, 0.9, alpha)


def _move_to_collection(obj, target_coll, in_front=False):
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    target_coll.objects.link(obj)
    if in_front:
        obj.show_in_front = True


def _add_to_overlay(obj, overlay_coll):
    _move_to_collection(obj, overlay_coll, in_front=True)


def build_axis_gizmo(half_extents, origin=None):
    """Create RGB XYZ-axis arrows at the given origin, sized to fit within the model bounding box."""
    if origin is None:
        origin = Vector((0, 0, 0))

    gizmo_coll = bpy.data.collections.new("AxisGizmo")
    bpy.context.scene.collection.children.link(gizmo_coll)

    max_half = max(half_extents.x, half_extents.y, half_extents.z)
    axis_length = max_half * 0.30
    shaft_radius = axis_length * 0.04
    cone_height = axis_length * 0.22
    cone_radius = shaft_radius * 2.8
    label_size = axis_length * 0.18

    axes = [
        ("X", Vector((1, 0, 0)), (0.9, 0.05, 0.05, 1.0), mathutils.Euler((math.pi / 2, 0, math.pi / 2))),
        ("Y", Vector((0, 1, 0)), (0.05, 0.75, 0.05, 1.0), mathutils.Euler((math.pi / 2, 0, math.pi))),
        ("Z", Vector((0, 0, 1)), (0.15, 0.3, 0.95, 1.0), mathutils.Euler((math.pi / 2, 0, 0))),
    ]

    z_up = Vector((0, 0, 1))

    for name, direction, color, text_rot in axes:
        shaft_center = origin + direction * (axis_length / 2)
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=shaft_radius, depth=axis_length, location=shaft_center)
        shaft = bpy.context.active_object
        shaft.name = f"Gizmo_Shaft_{name}"
        shaft.rotation_mode = "QUATERNION"
        shaft.rotation_quaternion = z_up.rotation_difference(direction)
        shaft.color = color
        _move_to_collection(shaft, gizmo_coll, in_front=True)

        cone_center = origin + direction * (axis_length + cone_height / 2)
        bpy.ops.mesh.primitive_cone_add(
            vertices=16, radius1=cone_radius, radius2=0, depth=cone_height, location=cone_center
        )
        cone = bpy.context.active_object
        cone.name = f"Gizmo_Arrow_{name}"
        cone.rotation_mode = "QUATERNION"
        cone.rotation_quaternion = z_up.rotation_difference(direction)
        cone.color = color
        _move_to_collection(cone, gizmo_coll, in_front=True)

        label_offset = axis_length + cone_height + label_size * 0.6
        bpy.ops.object.text_add(location=origin + direction * label_offset)
        txt = bpy.context.active_object
        txt.data.body = name
        txt.data.size = label_size
        txt.data.align_x = "CENTER"
        txt.data.align_y = "CENTER"
        txt.data.extrude = shaft_radius * 0.5
        txt.name = f"Gizmo_Label_{name}"
        txt.color = color
        txt.rotation_euler = text_rot
        bpy.ops.object.select_all(action="DESELECT")
        txt.select_set(True)
        bpy.context.view_layer.objects.active = txt
        bpy.ops.object.convert(target="MESH")
        txt = bpy.context.active_object
        _move_to_collection(txt, gizmo_coll, in_front=True)


def build_skeleton_overlay(arm_obj, joint_radius=1.0, bone_radius=0.5, render_bones=True):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    arm_eval = arm_obj.evaluated_get(dg)
    root_bone = arm_obj.pose.bones[0]

    overlay_coll = bpy.data.collections.get("SkeletonOverlay")
    if overlay_coll is None:
        overlay_coll = bpy.data.collections.new("SkeletonOverlay")
        bpy.context.scene.collection.children.link(overlay_coll)

    z_axis = Vector((0, 0, 1))

    # joints
    for pb in arm_eval.pose.bones:
        head_world = arm_eval.matrix_world @ pb.head

        is_root = pb.name == root_bone.name

        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=24, ring_count=12, radius=joint_radius if not is_root else joint_radius * 1.5, location=head_world
        )

        sph = bpy.context.active_object
        sph.name = f"Joint_{pb.name}"

        if is_root:
            sph.color = (0.0, 1.0, 0.0, 1.0)
        else:
            sph.color = (1.0, 0.0, 0.0, 1.0)

        _add_to_overlay(sph, overlay_coll)

    if not render_bones:
        return

    # bones
    for pb in arm_eval.pose.bones:
        if not pb.parent:
            continue
        head_world = arm_eval.matrix_world @ pb.head
        parent_world = arm_eval.matrix_world @ pb.parent.head
        vec = head_world - parent_world
        length = vec.length
        if length < 1e-6:
            continue

        mid = (head_world + parent_world) * 0.5
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=bone_radius, depth=length, location=mid)
        cyl = bpy.context.active_object
        cyl.name = f"Bone_{pb.parent.name}_to_{pb.name}"
        cyl.rotation_mode = "QUATERNION"
        cyl.rotation_quaternion = z_axis.rotation_difference(vec.normalized())
        cyl.color = (0.0, 0.0, 1.0, 1.0)

        _add_to_overlay(cyl, overlay_coll)


def main():
    args = parse_args()

    clean_scene()
    ensure_engine()
    setup_render(args.frame_start, args.resx, args.resy)

    imported = import_usd(args.usd)
    if not imported:
        raise RuntimeError("USD import produced no objects.")

    skeleton, kept = filter_unbound_meshes(imported)
    skeleton.data.pose_position = "REST"

    root, centerer = build_transform_hierarchy(kept)

    bb_min_raw, bb_max_raw = compute_aabb(kept)
    half_n, bb_center, norm_scale = apply_normalization(root, centerer, bb_min_raw, bb_max_raw)

    mesh_objs = [o for o in kept if o.type == "MESH"]
    replace_mesh_materials(mesh_objs, alpha=args.mesh_alpha)

    build_skeleton_overlay(
        skeleton,
        joint_radius=args.joint_radius,
        bone_radius=args.bone_radius,
        render_bones=args.render_bones,
    )

    if args.gizmo_3d:
        gizmo_origin = -bb_center * norm_scale
        build_axis_gizmo(half_n, origin=gizmo_origin)

    bpy.context.view_layer.update()

    up = args.up_axis.upper()
    if up == "Z":
        views = ["+Y", "+X", "-Y", "-X"]
    elif up == "Y":
        views = ["-Z", "+X", "+Z", "-X"]
    elif up == "X":
        views = ["+Y", "+Z", "-Y", "-Z"]
    else:
        raise ValueError("up_axis must be X, Y, or Z")

    up_vec = {"X": Vector((1, 0, 0)), "Y": Vector((0, 1, 0)), "Z": Vector((0, 0, 1))}[up]
    views_meta = {"up_axis": up, "views": []}

    for i, v in enumerate(views):
        cam = make_camera(v)
        cam_data = cam.data

        half_w, half_h, half_d = axis_permutation_for_view(half_n, v, up_axis=up)

        dist = compute_camera_distance(cam_data, half_w, half_h, half_d, args.margin)
        cam = position_camera(cam, v, dist, azimuth_deg=args.view_angle_offset, up_vector=up_vec)

        bpy.context.scene.camera = cam
        bpy.context.scene.render.filepath = out_with_suffix(args.out, f"view_{i}")
        bpy.ops.render.render(write_still=True)

        mat = cam.matrix_world
        views_meta["views"].append(
            {
                "image": f"view_{i}.png",
                "view_axis": v,
                "camera_right": [mat[0][0], mat[1][0], mat[2][0]],
                "camera_up": [mat[0][1], mat[1][1], mat[2][1]],
            }
        )

        print(f"[DONE] Render {v} -> {bpy.context.scene.render.filepath}")

    meta_path = os.path.join(args.out, "views_meta.json")
    with open(meta_path, "w") as f:
        json.dump(views_meta, f, indent=2)
    print(f"[DONE] View metadata -> {meta_path}")


if __name__ == "__main__":
    main()
