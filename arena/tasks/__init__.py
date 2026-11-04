import sys

from ..registry import autoload
from .base import TASKS, Task, make_task, register_task

autoload(sys.modules[__name__])   # imports every task file here; a new file with @register_task needs no other edit

__all__ = ["Task", "TASKS", "register_task", "make_task"]
