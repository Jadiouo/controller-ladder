"""Planar geometry of the Push-T object (shared by the task, the quasi-static push model and the teacher).

Shape frame: origin = centroid of the T, u = x, v = z (the plane of the arm). Pose (x, z, th) of a body maps a shape point p to
world  pos + R(th) p  with R(th) = [[cos, sin], [-sin, cos]]  (this is the rotation of a MuJoCo hinge about +y: positive th turns +x toward -z).
T = bar (W x B, on top, +z) + stem (S x L, below). Same 4:1 / 1:3 proportions as the Diffusion Policy Push-T (120x30 + 30x90 px), 1 px = 1.667 mm.
"""
from __future__ import annotations

import numpy as np

W, B, S, L = 0.20, 0.05, 0.05, 0.15                     # bar width / thickness, stem width / length (m)
_A1, _A2 = W * B, S * L
V_BAR = (_A2 * (B / 2 + L / 2)) / (_A1 + _A2)           # bar centre height above the centroid (0.0429 m)
V_STEM = V_BAR - (B / 2 + L / 2)                        # stem centre (-0.0571 m)
AREA = _A1 + _A2
# rectangles as (centre u, centre v, half u, half v)
RECTS = ((0.0, V_BAR, W / 2, B / 2), (0.0, V_STEM, S / 2, L / 2))
# counter-clockwise outline (u right, v up), 8 vertices
POLY = np.array([(-W / 2, V_BAR + B / 2), (-W / 2, V_BAR - B / 2), (-S / 2, V_BAR - B / 2), (-S / 2, V_STEM - L / 2),
                 (S / 2, V_STEM - L / 2), (S / 2, V_BAR - B / 2), (W / 2, V_BAR - B / 2), (W / 2, V_BAR + B / 2)])
# uniform-density inertia per unit mass about the centroid (m^2): rho^2 = I / m
RHO2 = (_A1 * ((W ** 2 + B ** 2) / 12 + V_BAR ** 2) + _A2 * ((S ** 2 + L ** 2) / 12 + V_STEM ** 2)) / AREA


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s], [-s, c]])


def rot_batch(th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)       # (..., 2, 2)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rect_corners(x, z, th, rect):
    cu, cv, hu, hv = rect
    loc = np.array([(cu - hu, cv - hv), (cu + hu, cv - hv), (cu + hu, cv + hv), (cu - hu, cv + hv)])   # ccw in (u, v)
    return np.array([x, z]) + loc @ rot(th).T


def _clip(subject, a, b):
    """Sutherland-Hodgman: keep the part of polygon `subject` on the left of the directed edge a->b (clip polygon is ccw in (x, z)
    with z up, so 'left' is the inside; rotated shapes stay ccw because R is a proper rotation)."""
    out = []
    ex, ez = b[0] - a[0], b[1] - a[1]

    def side(p):
        return ex * (p[1] - a[1]) - ez * (p[0] - a[0])

    for i in range(len(subject)):
        p, q = subject[i], subject[(i + 1) % len(subject)]
        sp, sq = side(p), side(q)
        if sp >= 0:
            out.append(p)
        if (sp >= 0) != (sq >= 0):
            t = sp / (sp - sq)
            out.append(p + t * (q - p))
    return out


def _poly_area(pts):
    if len(pts) < 3:
        return 0.0
    p = np.asarray(pts)
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def convex_overlap(c1, c2):
    """Area of the intersection of two convex ccw quadrilaterals (arrays (4, 2))."""
    poly = [p for p in c1]
    for i in range(len(c2)):
        if not poly:
            return 0.0
        poly = _clip(poly, c2[i], c2[(i + 1) % len(c2)])
    return _poly_area(poly)


def overlap_area(pose_a, pose_b) -> float:
    """Area (m^2) of the intersection of two T shapes at poses (x, z, th). Exact: the bar and the stem of one T are disjoint, so the
    intersection is the sum of the four convex rectangle-rectangle intersections."""
    ca = [rect_corners(*pose_a, r) for r in RECTS]
    cb = [rect_corners(*pose_b, r) for r in RECTS]
    return float(sum(convex_overlap(a, b) for a in ca for b in cb))


def coverage(pose, goal) -> float:
    """Diffusion Policy Push-T score: overlap area / area of the goal T (1 = perfect, 0 = disjoint)."""
    return overlap_area(pose, goal) / AREA


def poly_world(pose):
    x, z, th = pose
    return np.array([x, z]) + POLY @ rot(th).T


def closest_on_polygon(pts, poly):
    """Closest points of polygon `poly` (n, 2) to the points `pts` (B, 2): returns (B, 2) points and (B,) distances."""
    a = poly
    ab = np.roll(poly, -1, 0) - a
    inv = 1.0 / (ab ** 2).sum(-1)
    diff = pts[:, None, :] - a[None]                                   # (B, n, 2)
    t = np.clip((diff * ab[None]).sum(-1) * inv[None], 0.0, 1.0)
    off = diff - t[..., None] * ab[None]                               # pts - projection
    d2 = (off ** 2).sum(-1)
    i = np.argmin(d2, 1)
    ar = np.arange(len(pts))
    return pts - off[ar, i], np.sqrt(d2[ar, i])


def inside_polygon(pts, poly):
    """Boolean (B,): points strictly inside the (ccw) polygon (even-odd rule)."""
    x, y = pts[:, 0:1], pts[:, 1:2]
    xa, ya = poly[None, :, 0], poly[None, :, 1]
    xb, yb = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    cond = (ya > y) != (yb > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = xa + (y - ya) * (xb - xa) / (yb - ya)
    return (cond & (x < xint)).sum(1) % 2 == 1
