"""MJX model of the arena 2-link arm, built from the SAME MJCF generator as the CPU env (`arena.robots.build_mjcf`).

Per-env payload mass: body_mass / body_inertia / body_subtreemass are affine in the payload mass, so we put two
reference models on the device, take the difference and interpolate per env. (`check_payload_fields` verifies that no
other model field changes with the payload except constraint-solver-only ones that this contact-free task never reads.)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from arena.robots import RobotSpec, build_mjcf

PAYLOAD_FIELDS = ("body_mass", "body_inertia", "body_subtreemass")
IRRELEVANT = {"actuator_acc0", "body_invweight0", "dof_invweight0", "dof_M0"}   # only used by the constraint solver / reference accel


def cpu_model(spec: RobotSpec | None = None, payload: float | None = None) -> mujoco.MjModel:
    spec = spec or RobotSpec()
    if payload is not None:
        spec = spec.with_(payload_mass=payload)
    return mujoco.MjModel.from_xml_string(build_mjcf(spec))


def _cast(m, dtype):
    return jax.tree_util.tree_map(lambda x: x.astype(dtype) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x, m)


class ArmModel:
    """Nominal MJX model + the means to build a per-env batched model."""

    def __init__(self, spec: RobotSpec | None = None):
        self.spec = spec or RobotSpec()
        self.mj = cpu_model(self.spec)
        self.m = mjx.put_model(self.mj)
        m1 = mjx.put_model(cpu_model(self.spec, self.spec.payload_mass + 1.0))
        self.p0, self.p1 = self.spec.payload_mass, self.spec.payload_mass + 1.0
        self._delta = {f: getattr(m1, f) - getattr(self.m, f) for f in PAYLOAD_FIELDS}
        self._m1 = m1

    def check_payload_fields(self):
        """names of model fields (other than PAYLOAD_FIELDS) that differ between payload values; must be IRRELEVANT"""
        bad = []
        for f in self.m.__dataclass_fields__ if hasattr(self.m, "__dataclass_fields__") else []:
            a, b = getattr(self.m, f), getattr(self._m1, f)
            if hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape and a.dtype.kind == "f":
                if f not in PAYLOAD_FIELDS and bool(jnp.any(a != b)):
                    bad.append(f)
        return sorted(set(bad) - IRRELEVANT)

    def batched(self, payload):
        """(model with leading env axis on the payload-dependent fields, in_axes pytree for vmap). payload: (E,) kg."""
        payload = jnp.asarray(payload)
        s = (payload - self.p0)
        upd = {f: getattr(self.m, f)[None] + s.reshape((-1,) + (1,) * getattr(self.m, f).ndim) * self._delta[f][None]
               for f in PAYLOAD_FIELDS}
        mb = self.m.replace(**{k: v.astype(getattr(self.m, k).dtype) for k, v in upd.items()})
        axes = jax.tree_util.tree_map(lambda _: None, self.m).replace(**{f: 0 for f in PAYLOAD_FIELDS})
        return mb, axes
