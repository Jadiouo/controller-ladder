"""L6: imitation-learning students (BC-MLP, MIP, Diffusion Policy, flow matching) and the teacher -> demonstrations ->
training -> evaluation pipeline.  Needs PyTorch (not a dependency of the core arena; the core never imports this
package except lazily from `controllers/student.py`).  Pure-numpy parts (features, dataset I/O) work without torch."""
from .features import FeatureSpec, layout_of
from .dataset import ShardedDataset, collect

__all__ = ["FeatureSpec", "layout_of", "ShardedDataset", "collect"]
