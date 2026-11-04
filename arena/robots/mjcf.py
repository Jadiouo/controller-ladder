"""MJCF generation for an n-link planar arm. Pure function of RobotSpec (+ optional task attachment)."""
from __future__ import annotations

import re

from .spec import RobotSpec


def _tool_xml(spec: RobotSpec) -> str:
    if spec.tool == "paddle":     # flat plate perpendicular to the last link, centred on the tip
        return (f'<geom name="paddle" type="box" size="0.01 0.03 0.10" pos="0 0 0" mass="{spec.tool_mass!r}" '
                f'contype="1" conaffinity="1"/>')
    if spec.tool == "gripper":    # two fixed fingers (passive; not actuated in this version)
        m = spec.tool_mass / 2
        return "".join(f'<geom name="finger{i}" type="box" size="0.05 0.01 0.01" pos="0.05 0 {z}" mass="{m!r}" '
                       f'contype="0" conaffinity="0"/>' for i, z in enumerate((0.04, -0.04)))
    if spec.tool == "net":        # shallow catching basket, opening toward local +z (up when the tool is level); contact enabled
        m = spec.tool_mass / 3
        walls = "".join(f'<geom name="net_wall{i}" type="box" size="0.008 0.05 0.035" pos="{sx * 0.108!r} 0 0.035" mass="{m / 2!r}" '
                        f'rgba="0.15 0.65 0.35 1" contype="1" conaffinity="1"/>' for i, sx in enumerate((-1, 1)))
        return (f'<geom name="net_floor" type="box" size="0.10 0.05 0.008" pos="0 0 -0.008" mass="{m!r}" '
                f'rgba="0.15 0.65 0.35 1" contype="1" conaffinity="1"/>' + walls)
    if spec.tool == "claw":       # visual C-shaped holder around a ball of radius <= 0.04 at the tip (no contact)
        m = spec.tool_mass / 3
        prongs = "".join(f'<geom name="claw{i}" type="box" size="0.04 0.012 0.008" pos="-0.005 0 {z!r}" mass="{m / 2!r}" '
                         f'rgba="0.9 0.55 0.1 1" contype="0" conaffinity="0"/>' for i, z in enumerate((0.055, -0.055)))
        return prongs + (f'<geom name="claw_back" type="box" size="0.008 0.012 0.063" pos="-0.045 0 0" mass="{m!r}" '
                         f'rgba="0.9 0.55 0.1 1" contype="0" conaffinity="0"/>')
    return ""


_SECTION = re.compile(r"<!--@(\w+)-->")
SECTIONS = ("world", "equality")


def split_sections(xml: str) -> tuple[str, dict]:
    """`tip_xml` may carry sections after sentinel comments: `<!--@world-->...` goes into <worldbody> (e.g. a free ball),
    `<!--@equality-->...` into <equality> (e.g. a connect constraint that holds an object to the tip). Text before the first
    sentinel stays inside the tip body, so tip_xml without sentinels behaves exactly as before."""
    parts = _SECTION.split(xml)
    out = {name: "" for name in SECTIONS}
    for name, body in zip(parts[1::2], parts[2::2]):
        if name not in out:
            raise ValueError(f"unknown tip_xml section '{name}'; known: {SECTIONS}")
        out[name] += body
    return parts[0], out


def build_mjcf(spec: RobotSpec, tip_xml: str = "") -> str:
    """`tip_xml` is extra XML inserted inside the 'tip' body (e.g. a passive pole). Site 'tip' marks the tool point."""
    n = spec.n
    tip_xml, sec = split_sections(tip_xml)
    tip = ""
    if spec.payload_mass > 0:
        tip += (f'<geom type="sphere" size="{spec.payload_radius!r}" mass="{spec.payload_mass!r}" '
                f'contype="0" conaffinity="0"/>')
    tip += _tool_xml(spec) + '<site name="tip" size="0.01"/>' + tip_xml
    xml = f'<body name="tip" pos="{spec.link_lengths[-1]!r} 0 0">{tip}</body>'
    for i in reversed(range(n)):
        L = spec.link_lengths[i]
        pos = "0 0 0" if i == 0 else f"{spec.link_lengths[i - 1]!r} 0 0"
        xml = (f'<body name="l{i + 1}" pos="{pos}">'
               f'<joint name="j{i + 1}" type="hinge" axis="0 1 0" damping="{spec.damping[i]!r}" armature="{spec.armature!r}"/>'
               f'<geom type="capsule" fromto="0 0 0 {L!r} 0 0" size="{spec.link_radii[i]!r}" mass="{spec.link_masses[i]!r}" '
               f'contype="0" conaffinity="0"/>{xml}</body>')
    motors = "".join(f'<motor joint="j{i + 1}" gear="1" ctrllimited="true" ctrlrange="-{spec.tau_max!r} {spec.tau_max!r}"/>'
                     for i in range(n))
    return (f'<mujoco model="arm{n}"><option timestep="{spec.dt_sim!r}" gravity="0 0 -9.81" integrator="RK4"/>'
            f'<worldbody>{xml}{sec["world"]}</worldbody>'
            + (f'<equality>{sec["equality"]}</equality>' if sec["equality"] else "")
            + f'<actuator>{motors}</actuator></mujoco>')
