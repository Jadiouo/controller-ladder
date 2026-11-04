import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"; os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.5"
import jax, jax.numpy as jnp
print(jax.__version__, jax.devices())
x=jnp.ones((256,256)); print((x@x).sum())
from mujoco import mjx; print(hasattr(mjx,'inverse'))
d=jax.devices()[0]; print(d.device_kind, getattr(d,'compute_capability',None)); print(d.memory_stats())
