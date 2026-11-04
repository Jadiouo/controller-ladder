import sys

from ..registry import autoload
from .base import CONTROLLERS, Controller, as_factory, controller_factory, register_controller

autoload(sys.modules[__name__])   # imports every controller file; a new file with @register_controller needs no other edit

__all__ = ["Controller", "CONTROLLERS", "register_controller", "controller_factory", "as_factory"]
