"""arena_mjx: MJX (JAX) port of the 2-DoF tracking task of `arena/` (no contacts). See results/mjx/README.md.

Import this package BEFORE creating any jax array, and set the XLA env vars first when a GPU is used
(`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`); `arena_mjx.config.setup()` does it.
"""
from .config import setup  # noqa: F401
