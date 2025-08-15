"""Dynamically loads abstract base classes (ABCs) for components.

This module makes ABCs defined within the 'cvlabkit/component/base' directory
directly accessible when 'cvlabkit.component.base' is imported, simplifying
the process of inheriting from these foundational component interfaces.
"""

import os
import importlib
import inspect
from abc import ABCMeta


_current_dir = os.path.dirname(__file__)
_loaded = {}

for fname in os.listdir(_current_dir):
    if fname.endswith(".py") and fname != "__init__.py":
        modname = fname[:-3]
        # Use absolute import for robustness.
        full_module_name = f"{__name__}.{modname}"
        module = importlib.import_module(full_module_name)
        _loaded.update({
            name: cls for name, cls in inspect.getmembers(module, inspect.isclass)
            if isinstance(cls, ABCMeta) # Only load ABCs.
        })

globals().update(_loaded)
__all__ = list(_loaded.keys())
