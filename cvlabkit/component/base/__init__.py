import os
import importlib
import inspect
import sys
from abc import ABCMeta


_current_dir = os.path.dirname(__file__)
mod_prefix = __name__
_loaded = {}

for fname in os.listdir(_current_dir):
    if fname.endswith(".py") and fname != "__init__.py":
        modname = fname[:-3]
        module = importlib.import_module(f"{mod_prefix}.{modname}")
        _loaded.update({
            name: cls for name, cls in inspect.getmembers(module, inspect.isclass)
            if isinstance(cls, ABCMeta)
        })

globals().update(_loaded)
__all__ = list(_loaded.keys())
