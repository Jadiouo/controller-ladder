"""Process-wide JAX setup. Call `setup(x64=...)` before the first jax computation."""
import os


def setup(x64: bool = False, mem_fraction: float = 0.5):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))
    import jax
    jax.config.update("jax_enable_x64", bool(x64))
    # TF32 / bf16 matmul shortcuts on Ampere+ would corrupt the mass-matrix solves: force full precision.
    jax.config.update("jax_default_matmul_precision", "highest")
    return jax
