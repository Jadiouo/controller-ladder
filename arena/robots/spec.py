"""RobotSpec: parameters of an n-link planar arm moving in the x-z plane (hinges about +y)."""
from __future__ import annotations

from dataclasses import dataclass, replace

TOOLS = ("none", "paddle", "gripper", "net", "claw")


@dataclass(frozen=True)
class RobotSpec:
    n: int = 2
    link_lengths: tuple = (0.5, 0.4)
    link_masses: tuple = (1.0, 0.8)
    link_radii: tuple = (0.03, 0.025)
    damping: tuple = (0.5, 0.3)
    armature: float = 0.01
    payload_mass: float = 1.5      # point-like sphere at the tip (kg); 0 = none
    payload_radius: float = 0.04
    tool: str = "none"             # 'none' | 'paddle' | 'gripper' | 'net' (shallow catching basket) | 'claw' (passive geometry, mass below)
    tool_mass: float = 0.2
    tau_max: float = 80.0          # actuator ctrlrange (Nm), all joints
    dt_sim: float = 0.0025
    substeps: int = 2              # control period = dt_sim * substeps (200 Hz by default)

    def __post_init__(self):
        for name in ("link_lengths", "link_masses", "link_radii", "damping"):
            if len(getattr(self, name)) != self.n:
                raise ValueError(f"{name} must have n={self.n} entries")
        if self.tool not in TOOLS:
            raise ValueError(f"tool must be one of {TOOLS}")

    @property
    def dt(self) -> float:
        return self.dt_sim * self.substeps

    def with_(self, **kw) -> "RobotSpec":
        return replace(self, **kw)

    @classmethod
    def two_link(cls, **kw) -> "RobotSpec":
        """The spike01-03 arm (default)."""
        return cls(**kw)

    @classmethod
    def three_link(cls, **kw) -> "RobotSpec":
        base = dict(n=3, link_lengths=(0.4, 0.35, 0.25), link_masses=(1.0, 0.8, 0.5),
                    link_radii=(0.03, 0.025, 0.02), damping=(0.5, 0.3, 0.2))
        base.update(kw)
        return cls(**base)
