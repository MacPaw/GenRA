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

"""Overlay a 3D-projected axis gizmo on rendered view images.

Reads views_meta.json (produced by rig_rendering.py) to obtain the camera
orientation for each view, then projects the world axes onto screen
space.  The result is a perspective-correct gizmo drawn in the bottom-left
corner with:
  - foreshortened arrow lengths
  - depth-sorted drawing order (back-to-front)
  - dimmed colours for axes that recede into the screen

Usage:
    python gizmo_overlay.py --images_dir /path/to/renders
    python gizmo_overlay.py --images_dir /path/to/renders --out_dir /path/to/output
"""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

AXIS_COLORS = {
    "X": (230, 25, 25),
    "Y": (25, 190, 25),
    "Z": (50, 80, 240),
}

WORLD_AXES = {
    "X": (1, 0, 0),
    "Y": (0, 1, 0),
    "Z": (0, 0, 1),
}


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dim_color(color, factor):
    """Blend *color* toward white.  factor=1 → original, factor=0 → white."""
    return tuple(int(c * factor + 255 * (1 - factor)) for c in color)


def _load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFCompactRounded.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def project_axes(cam_right, cam_up, elevation_deg=15.0):
    """Project every world axis onto screen space.

    An optional *elevation_deg* tilts the gizmo's virtual viewpoint downward
    (as if looking from slightly above), which separates horizontal axes that
    would otherwise overlap when the render camera is perfectly level.

    Returns ``{name: (sx, sy, depth, mag)}`` where *sx*, *sy* are the raw
    dot-products (screen-right / screen-up), *depth* is the dot-product with
    the camera forward vector (positive = into screen), and *mag* is the
    screen-space magnitude ``hypot(sx, sy)``.
    """
    cam_fwd = _cross(cam_up, cam_right)

    if elevation_deg:
        rad = math.radians(elevation_deg)
        c, s = math.cos(rad), math.sin(rad)
        up_orig, fwd_orig = cam_up, cam_fwd
        cam_up = tuple(c * u + s * f for u, f in zip(up_orig, fwd_orig, strict=True))
        cam_fwd = tuple(-s * u + c * f for u, f in zip(up_orig, fwd_orig, strict=True))

    result = {}
    for name, wdir in WORLD_AXES.items():
        sx = _dot(wdir, cam_right)
        sy = _dot(wdir, cam_up)
        depth = _dot(wdir, cam_fwd)
        mag = math.hypot(sx, sy)
        result[name] = (sx, sy, depth, mag)
    return result


def get_view_normal_axis(cam_right, cam_up):
    """Return the world axis that points most toward/away from the camera."""
    cam_fwd = _cross(cam_up, cam_right)
    return max(WORLD_AXES.items(), key=lambda item: abs(_dot(item[1], cam_fwd)))[0]


def _draw_arrow(draw, origin, tip_offset, color, line_width, head_len, label, font):
    """Draw an arrow from *origin* with the tip at *origin + tip_offset*.

    *tip_offset* is in image-pixel coordinates (x-right, y-down) so that
    foreshortening is baked in.
    """
    ox, oy = origin
    tip_x = ox + tip_offset[0]
    tip_y = oy + tip_offset[1]

    arrow_len = math.hypot(tip_offset[0], tip_offset[1])
    if arrow_len < 2:
        return

    ux = tip_offset[0] / arrow_len
    uy = tip_offset[1] / arrow_len

    actual_head = min(head_len, arrow_len * 0.8)
    base_x = tip_x - ux * actual_head
    base_y = tip_y - uy * actual_head

    px, py = -uy, ux
    hw = actual_head * 0.38
    tri = [
        (tip_x, tip_y),
        (base_x + px * hw, base_y + py * hw),
        (base_x - px * hw, base_y - py * hw),
    ]

    outline = (0, 0, 0)
    draw.line([(ox, oy), (base_x, base_y)], fill=outline, width=line_width + 2)
    draw.polygon(tri, fill=outline)
    draw.line([(ox, oy), (base_x, base_y)], fill=color, width=line_width)
    draw.polygon(tri, fill=color)

    if label and font:
        lx = tip_x + ux * (actual_head * 0.9)
        ly = tip_y + uy * (actual_head * 0.9)
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = lx - tw / 2, ly - th / 2
        for ddx, ddy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            draw.text((tx + ddx, ty + ddy), label, fill=outline, font=font)
        draw.text((tx, ty), label, fill=color, font=font)


def overlay_gizmo(img, projected, gizmo_size=None, padding=None, hidden_axes=None):
    """Draw a 3D-projected axis gizmo in the bottom-left corner of *img*."""
    if hidden_axes is None:
        hidden_axes = set()

    w, h = img.size
    short = min(w, h)
    if gizmo_size is None:
        gizmo_size = int(short * 0.14)
    if padding is None:
        padding = int(short * 0.02)

    base_lw = max(2, gizmo_size // 20)
    head_len = max(8, gizmo_size // 5)
    font_size = max(12, gizmo_size // 4)
    font = _load_font(font_size)

    label_extra = head_len * 0.9 + font_size
    dot_r = max(3, base_lw + 1)

    extents_x = [-dot_r, dot_r]
    extents_y = [-dot_r, dot_r]
    for axis_name, (sx, sy, _depth, mag) in projected.items():
        if axis_name in hidden_axes:
            continue
        if mag < 0.01:
            continue
        tip_dx = sx * gizmo_size
        tip_dy = -sy * gizmo_size
        arrow_len = math.hypot(tip_dx, tip_dy)
        if arrow_len < 2:
            continue
        ux, uy = tip_dx / arrow_len, tip_dy / arrow_len
        lbl_dx = tip_dx + ux * label_extra
        lbl_dy = tip_dy + uy * label_extra
        extents_x += [tip_dx, lbl_dx]
        extents_y += [tip_dy, lbl_dy]

    cx = max(padding, padding - min(extents_x))
    cy = min(h - padding, h - padding - max(extents_y))
    origin = (cx, cy)

    draw = ImageDraw.Draw(img)

    sorted_axes = sorted(projected.items(), key=lambda item: -item[1][2])

    for name, (sx, sy, depth, mag) in sorted_axes:
        if name in hidden_axes:
            continue
        if mag < 0.01:
            continue

        tip_dx = sx * gizmo_size
        tip_dy = -sy * gizmo_size

        receding = depth > 0.05
        if receding:
            color = _dim_color(AXIS_COLORS[name], 0.55)
            lw = max(1, base_lw - 1)
        else:
            color = AXIS_COLORS[name]
            lw = base_lw

        _draw_arrow(draw, origin, (tip_dx, tip_dy), color, lw, head_len, name, font)

    draw.ellipse(
        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
        fill=(50, 50, 50),
        outline=(0, 0, 0),
    )

    return img


def parse_args():
    p = argparse.ArgumentParser(description="Overlay 3D-projected axis gizmo on rendered view images")
    p.add_argument("--images_dir", required=True, help="Directory containing view_*.png images")
    p.add_argument("--meta", default=None, help="Path to views_meta.json (default: <images_dir>/views_meta.json)")
    p.add_argument("--out_dir", default=None, help="Output directory (default: overwrite in-place)")
    p.add_argument("--gizmo_size", type=int, default=None, help="Arrow length in pixels (default: 18%% of image)")
    p.add_argument("--padding", type=int, default=None, help="Padding from image corner in pixels")
    p.add_argument(
        "--gizmo_elevation",
        type=float,
        default=15.0,
        help="Virtual elevation in degrees for the gizmo viewpoint (tilts gizmo as if seen from above, default: 15)",
    )
    p.add_argument(
        "--hide-view-normal-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide the axis that points into/out of the view plane for each render (default: true)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    meta_path = Path(args.meta) if args.meta else images_dir / "views_meta.json"
    out_dir = Path(args.out_dir) if args.out_dir else images_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path) as f:
        meta = json.load(f)

    for view in meta["views"]:
        img_name = view["image"]
        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"[WARN] {img_path} not found, skipping")
            continue

        projected = project_axes(view["camera_right"], view["camera_up"], elevation_deg=args.gizmo_elevation)
        hidden_axes = set()
        if args.hide_view_normal_axis:
            hidden_axes.add(get_view_normal_axis(view["camera_right"], view["camera_up"]))

        img = Image.open(img_path).convert("RGBA")
        img = overlay_gizmo(
            img,
            projected,
            gizmo_size=args.gizmo_size,
            padding=args.padding,
            hidden_axes=hidden_axes,
        )

        out_path = out_dir / img_name
        img.save(out_path)
        shown = [n for n, (_, _, _, m) in projected.items() if m >= 0.01 and n not in hidden_axes]
        print(f"[DONE] {out_path}  axes: {', '.join(shown)}")


if __name__ == "__main__":
    main()
