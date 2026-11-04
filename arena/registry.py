"""Tiny name -> object registry plus auto-import of every module in a package.

Adding a task or controller = adding a file to arena/tasks or arena/controllers that uses the
`@register_*` decorator; `autoload` imports every module in the package so no core file is edited.
"""
import importlib
import pkgutil


class Registry:
    def __init__(self, kind):
        self.kind, self._items = kind, {}

    def register(self, name):
        def deco(obj):
            if name in self._items and self._items[name] is not obj:
                raise ValueError(f"{self.kind} '{name}' already registered")
            self._items[name] = obj
            obj.registry_name = name
            return obj
        return deco

    def get(self, name):
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. Registered: {sorted(self._items)}")
        return self._items[name]

    def names(self):
        return sorted(self._items)

    def __contains__(self, name):
        return name in self._items


def autoload(package):
    """Import every submodule of `package` (a module object) so their decorators run."""
    for m in pkgutil.iter_modules(package.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"{package.__name__}.{m.name}")
