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

# Blender 4.5.x or higher
# Render rigged + animated .usda to MP4.

import math
import sys

import bpy
from mathutils import Vector  # pyright: ignore[reportMissingImports]


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    import argparse

    p = argparse.ArgumentParser(description="Render animated USD to MP4 w/ auto fit-to-camera scale")
    p.add_argument("--usd", required=True, help="Path to .usd/.usda/.usdc")
    p.add_argument("--out", required=True, help="Output .mp4 path")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--frame_start", type=int, default=-1, help="First frame (-1 = auto-detect from animation)")
    p.add_argument("--frame_end", type=int, default=-1, help="Last frame (-1 = auto-detect from animation)")
    p.add_argument("--step", type=int, default=2, help="Bounding-scan step (>=1)")
    p.add_argument("--resx", type=int, default=512)
    p.add_argument("--resy", type=int, default=512)
    p.add_argument("--camera_dist", type=float, default=10.0, help="Fixed camera distance from scene center")
    p.add_argument(
        "--camera_azimuth", type=float, default=20.0, help="Horizontal orbit angle in degrees (positive = from right)"
    )
    p.add_argument(
        "--camera_elevation", type=float, default=15.0, help="Vertical orbit angle in degrees (positive = from above)"
    )
    p.add_argument("--margin", type=float, default=0.9, help="Fill fraction of viewport (0-1)")
    p.add_argument("--up_axis", type=str, default="Z", help="Up axis of the input USD (Y or Z)")
    p.add_argument(
        "--background",
        type=str,
        default="black",
        help="Background color: 'black', 'white', 'transparent', or hex '#RRGGBB'",
    )
    return p.parse_args(argv)


_NAMED_COLORS = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
}


def parse_background(value: str) -> tuple | str:
    """Return (r, g, b) floats in 0-1 range, or 'transparent'."""
    v = value.strip().lower()
    if v == "transparent":
        return "transparent"
    if v in _NAMED_COLORS:
        return _NAMED_COLORS[v]
    hex_str = v.lstrip("#")
    if len(hex_str) == 6:
        r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
        return (r / 255.0, g / 255.0, b / 255.0)
    raise ValueError(f"Unrecognised background color: '{value}'. Use a name, hex '#RRGGBB', or 'transparent'.")


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def ensure_engine(use_eevee=False):
    if use_eevee:
        preferred = ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"]
        for idname in preferred:
            try:
                bpy.context.scene.render.engine = idname
                return
            except Exception:
                print(f"Engine {idname} not found, trying next")
                continue
        print("No engine found, using CYCLES")
        bpy.context.scene.render.engine = "CYCLES"
    else:
        bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
        sh = bpy.context.scene.display.shading
        sh.color_type = "TEXTURE"


def setup_render(out_path, resx, resy, fps, frame_start, frame_end, background="black"):
    scn = bpy.context.scene
    scn.render.image_settings.media_type = "VIDEO"
    scn.render.image_settings.file_format = "FFMPEG"
    scn.render.ffmpeg.format = "MPEG4"
    scn.render.ffmpeg.codec = "H264"
    scn.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scn.render.ffmpeg.ffmpeg_preset = "GOOD"
    scn.render.ffmpeg.gopsize = 12
    scn.render.ffmpeg.max_b_frames = 2
    scn.render.use_file_extension = True
    scn.render.resolution_x = resx
    scn.render.resolution_y = resy
    scn.render.fps = fps
    scn.frame_set(frame_start)
    scn.render.filepath = out_path

    bg = parse_background(background)

    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")

    if bg == "transparent":
        scn.render.film_transparent = True
        bpy.context.scene.world.color = (0, 0, 0)
    else:
        scn.render.film_transparent = False
        bpy.context.scene.world.color = bg

    scn.render.use_sequencer = False
    scn.view_settings.view_transform = "Standard"


def import_usd(path):
    before = set(bpy.data.objects.keys())

    bpy.ops.wm.usd_import(filepath=path, create_collection=True, import_cameras=False, import_lights=False)
    bpy.context.view_layer.update()

    after = set(bpy.data.objects.keys())
    new_names = list(after - before)

    return [bpy.data.objects[n] for n in new_names]


def make_fixed_camera(
    dist: float,
    up_axis: str = "Z",
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
):
    cam_data = bpy.data.cameras.new("Cam")
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_data.lens = 100.0
    cam_data.sensor_width = 36

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cos_el = math.cos(el)
    sin_el = math.sin(el)

    axis = up_axis.upper()
    if axis == "Z":
        # Front = -Y, azimuth orbits around Z, elevation lifts along Z
        x = dist * cos_el * math.sin(az)
        y = -dist * cos_el * math.cos(az)
        z = dist * sin_el
    elif axis == "Y":
        # Front = +Z, azimuth orbits around Y, elevation lifts along Y
        x = dist * cos_el * math.sin(az)
        y = dist * sin_el
        z = dist * cos_el * math.cos(az)
    else:
        raise ValueError("up_axis must be one of 'Y' or 'Z'")

    cam_obj.location = (x, y, z)

    target = bpy.data.objects.new("CamTarget", None)
    target.location = (0, 0, 0)
    bpy.context.collection.objects.link(target)

    track = cam_obj.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    bpy.context.scene.camera = cam_obj
    bpy.context.view_layer.update()

    return cam_obj


def depsgraph():
    return bpy.context.evaluated_depsgraph_get()


def compute_global_bounds_across_animation(objs, frame_start, frame_end, step):
    inf = float("inf")
    bb_min = Vector((inf, inf, inf))
    bb_max = Vector((-inf, -inf, -inf))

    scene = bpy.context.scene
    cur = scene.frame_current
    dg = depsgraph()

    def expand_with_mesh(eval_obj, world_matrix):
        nonlocal bb_min, bb_max
        # only MESH contributes (USD import typically yields meshes)
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

    frames = range(frame_start, frame_end + 1, max(1, step))
    if frame_end not in frames:
        frames = list(frames) + [frame_end]

    for f in frames:
        scene.frame_set(f)
        dg.update()

        for inst in dg.object_instances:
            base = inst.object
            if base.original not in objs and base not in objs:
                continue
            eval_obj = base.evaluated_get(dg)
            expand_with_mesh(eval_obj, inst.matrix_world)

    scene.frame_set(cur)
    return bb_min, bb_max


def fit_scale_for_camera(cam_obj, root, bb_min, bb_max, dist, margin):
    cam = cam_obj.data
    half = (bb_max - bb_min) * 0.5

    bpy.context.view_layer.update()
    rot_inv = cam_obj.matrix_world.inverted().to_3x3()

    # Project all 8 centered-box corners into camera-local space (rotation only,
    # since the model will be centered at origin by apply_fit_transform)
    corners = [Vector((sx * half.x, sy * half.y, sz * half.z)) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    local = [rot_inv @ c for c in corners]

    half_w = max(abs(v.x) for v in local)
    half_h = max(abs(v.y) for v in local)
    half_d = max(abs(v.z) for v in local)

    camera_angle = max(cam.angle_x, cam.angle_y)
    half_frame_limit = dist * math.tan(camera_angle * 0.5) * margin
    s1 = half_frame_limit / half_w if half_w > 0 else float("inf")
    s2 = half_frame_limit / half_h if half_h > 0 else float("inf")
    s3 = (dist - cam.clip_start) / half_d if half_d > 0 else float("inf")
    s = min(s1, s2, s3)

    print(f"[INFO] Fit scale: {s:.3f}")

    root.scale = (s, s, s)
    return s


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

    # collect meshes parented to / deformed by skeleton
    for obj in imported:
        if obj.type == "MESH":
            # keep if parented to skeleton or has armature modifier pointing to skeleton
            if obj.parent == skeleton:
                keep.append(obj)
                continue
            for mod in obj.modifiers:
                if mod.type == "ARMATURE" and mod.object == skeleton:
                    keep.append(obj)
                    break
        else:
            keep.append(obj)

    # delete everything else
    for obj in imported:
        if obj not in keep:
            print(f"[INFO] Removing imported object: {obj.name} ({obj.type})")
            bpy.data.objects.remove(obj, do_unlink=True)

    return skeleton, keep


def _iter_fcurves(action):
    """Yield all fcurves from an action, handling both Blender 4.x and 5.0+ APIs."""
    # Blender 5.0+: layered action system
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, "channelbags"):
                    for bag in strip.channelbags:
                        yield from bag.fcurves
    # Blender 4.x: direct fcurves attribute
    if hasattr(action, "fcurves"):
        yield from action.fcurves


def detect_animation_range():
    """Auto-detect frame range from all imported animation data."""
    frame_min = float("inf")
    frame_max = float("-inf")
    for action in bpy.data.actions:
        for fcurve in _iter_fcurves(action):
            for kp in fcurve.keyframe_points:
                frame_min = min(frame_min, kp.co[0])
                frame_max = max(frame_max, kp.co[0])
    if frame_min == float("inf"):
        print("[WARN] No animation keyframes found, falling back to 0-250")
        return 0, 250
    f_start, f_end = int(frame_min), int(frame_max)
    print(f"[INFO] Auto-detected animation range: {f_start}-{f_end}")
    return f_start, f_end


def build_transform_hierarchy(imported_objs):
    # root (for scaling)
    root = bpy.data.objects.new("ROOT_SCALE", None)
    root.empty_display_type = "PLAIN_AXES"
    bpy.context.collection.objects.link(root)

    # centerer (for translation)
    centerer = bpy.data.objects.new("CENTER_TRANSLATE", None)
    centerer.empty_display_type = "PLAIN_AXES"
    bpy.context.collection.objects.link(centerer)

    centerer.parent = root

    # parent only top-level objects to centerer
    for o in imported_objs:
        if o.parent is None:
            o.parent = centerer

    return root, centerer


def apply_fit_transform(root, centerer, bb_min, bb_max, scale):
    center = (bb_min + bb_max) * 0.5
    print(f"[INFO] Centering at: {center}, scale: {scale}")
    root.scale = (scale, scale, scale)
    centerer.location = -center


def main():
    args = parse_args()
    if args.step < 1:
        args.step = 1

    clean_scene()
    ensure_engine()

    imported = import_usd(args.usd)
    if not imported:
        raise RuntimeError("USD import produced no objects.")

    _, imported = filter_unbound_meshes(imported)

    frame_start = args.frame_start
    frame_end = args.frame_end
    if frame_start < 0 or frame_end < 0:
        auto_start, auto_end = detect_animation_range()
        if frame_start < 0:
            frame_start = auto_start
        if frame_end < 0:
            frame_end = auto_end

    setup_render(args.out, args.resx, args.resy, args.fps, frame_start, frame_end, args.background)

    cam_obj = make_fixed_camera(
        args.camera_dist,
        up_axis=args.up_axis,
        azimuth_deg=args.camera_azimuth,
        elevation_deg=args.camera_elevation,
    )

    bb_min, bb_max = compute_global_bounds_across_animation(imported, frame_start, frame_end, args.step)

    root, centerer = build_transform_hierarchy(imported)

    s = fit_scale_for_camera(cam_obj, root, bb_min, bb_max, args.camera_dist, args.margin)
    apply_fit_transform(root, centerer, bb_min, bb_max, s)

    # final render
    scn = bpy.context.scene
    scn.frame_start = frame_start
    scn.frame_end = frame_end

    bpy.context.view_layer.update()

    bpy.ops.render.render(animation=True, use_viewport=True)
    print("[DONE] Render finished:", args.out)


if __name__ == "__main__":
    main()
