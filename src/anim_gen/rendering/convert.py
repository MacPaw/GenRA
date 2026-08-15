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
# Convert rigged + animated USD to GLB.

import sys
from math import pi

import bpy
from mathutils import Matrix


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    import argparse

    p = argparse.ArgumentParser(description="Convert animated USD to GLB")
    p.add_argument("--usd", required=True, help="Path to .usd/.usda/.usdc")
    p.add_argument("--out", required=True, help="Output .glb path")
    p.add_argument("--fps", type=int, default=None, help="Scene FPS to preserve animation timing")
    p.add_argument("--up_axis", type=str, default="Z", help="Up axis of the input USD (Y or Z)")
    return p.parse_args(argv)


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_usd(path):
    existing_ptrs = {obj.as_pointer() for obj in bpy.data.objects}
    bpy.ops.wm.usd_import(filepath=path, create_collection=True, import_cameras=False, import_lights=False)
    bpy.context.view_layer.update()
    return [obj for obj in bpy.data.objects if obj.as_pointer() not in existing_ptrs]


def rotate_imported_roots_x(imported_objects, angle_radians):
    """Apply a world-space X rotation to imported root objects."""
    imported_ptrs = {obj.as_pointer() for obj in imported_objects}
    roots = [obj for obj in imported_objects if obj.parent is None or obj.parent.as_pointer() not in imported_ptrs]
    rot_x = Matrix.Rotation(angle_radians, 4, "X")
    for obj in roots:
        obj.matrix_world = rot_x @ obj.matrix_world
    bpy.context.view_layer.update()


def export_glb(path):
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        export_animations=True,
        export_apply=True,
        export_cameras=False,
        export_lights=False,
    )


def main():
    args = parse_args()

    clean_scene()

    imported_objects = import_usd(args.usd)
    if args.up_axis == "Y":
        rotate_imported_roots_x(imported_objects, -pi / 2)

    if args.fps is not None:
        bpy.context.scene.render.fps = int(args.fps)
    export_glb(args.out)

    print("[DONE] GLB conversion finished:", args.out)


if __name__ == "__main__":
    main()
