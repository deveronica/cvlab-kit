"""Provides a mechanism for generating configuration templates.

This module allows for the creation of configuration templates by tracking
missing configuration keys during a "dry run" of an application. It features a
`Placeholder` class that stands in for missing values, allowing the program to
run without crashing. The `ConfigProxy` class manages these placeholders and
records all accessed keys, which can then be used to generate a complete YAML
configuration template.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Set


class Placeholder:
    """A placeholder for a missing configuration key.

    This class is used by `ConfigProxy` during a dry run. When code attempts
    to access a configuration key that doesn't exist, a `Placeholder` is
    returned. It captures any operations performed on it (e.g., attribute
    access, method calls) and records the key as missing. This allows the
    program to continue running without crashing, for the primary purpose of
    generating a complete configuration template.

    Attributes:
        key: The dot-separated configuration key this placeholder represents.
        proxy: The `ConfigProxy` that created this placeholder.
    """

    def __init__(self, key: str, proxy: ConfigProxy) -> None:
        """Initializes a Placeholder instance.

        Args:
            key: The dot-separated configuration key (e.g., 'model.name').
            proxy: The `ConfigProxy` instance that created this placeholder.
        """
        self.key = key
        self.proxy = proxy

    def _resolve(self) -> Any:
        """Resolves the placeholder's value by inferring it from the context.

        If this is the first time the placeholder is being resolved, it asks the
        proxy to infer a default value based on the code context (i.e., the
        function that requested the key). This value is then stored in the
        proxy's `missing` dictionary. Subsequent calls for the same key will
        return the already inferred/stored value.

        Returns:
            The inferred or stored value for the placeholder's key.
        """
        # To prevent redundant inference or circular loops, only resolve once.
        if self.key not in self.proxy.resolved:
            self.proxy.resolved.add(self.key)
            value = self.proxy.infer_from_signature(self.key)
            self.proxy.missing[self.key] = value
            return value
        return self.proxy.missing.get(self.key)

    # --- Magic methods to intercept operations ---

    def __getattr__(self, name: str) -> Any:
        """Intercepts attribute access (e.g., `cfg.model.name`)."""
        try:
            return getattr(self._resolve(), name)
        except Exception:
            self.proxy.missing[self.key] = None
            return None

    def __getitem__(self, item: Any) -> Any:
        """Intercepts item access (e.g., `cfg.dataset[0]`)."""
        try:
            return self._resolve()[item]
        except Exception:
            self.proxy.missing[self.key] = {}
            return {}

    def __call__(self, *args, **kwargs) -> Any:
        """Intercepts function calls (e.g., `cfg.model.parameters()`)."""
        try:
            return self._resolve()(*args, **kwargs)
        except Exception:
            self.proxy.missing[self.key] = lambda *a, **kw: None
            return self.proxy.missing[self.key]

    def __repr__(self) -> str:
        """Returns the official string representation of the placeholder."""
        return repr(self._resolve())

    def __str__(self) -> str:
        """Returns the informal string representation of the placeholder."""
        try:
            return str(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = "<missing>"
            return "<missing>"

    def __int__(self) -> int:
        """Converts the placeholder to an integer."""
        try:
            return int(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = 1
            return 1

    def __float__(self) -> float:
        """Converts the placeholder to a float."""
        try:
            return float(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = 0.0
            return 0.0

    def __bool__(self) -> bool:
        """Converts the placeholder to a boolean."""
        try:
            return bool(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = False
            return False

    def __len__(self) -> int:
        """Returns the length of the placeholder (if it's a collection)."""
        try:
            return len(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = []
            return 0

    def __iter__(self):
        """Enables iteration over the placeholder (if it's iterable)."""
        try:
            return iter(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = []
            return iter([])

    def __deepcopy__(self, memo):
        """Handles deepcopy operations for the placeholder."""
        return Placeholder(self.key, self.proxy)

    def __reduce__(self):
        """Helps in pickling/unpickling the Placeholder object."""
        return (Placeholder, (self.key, self.proxy))

    def __fspath__(self) -> str:
        """Returns the file system path representation of the placeholder."""
        val = self._resolve()
        if isinstance(val, (str, bytes, Path)):
            return val
        self.proxy.missing[self.key] = "/missing/path"
        return "/missing/path"


class ConfigProxy:
    """Tracks missing config keys during a dry run to generate a template.

    This class is designed for a specific workflow: when running a script in a
    "dry-run" mode, this proxy allows the code to request configuration values
    that don't exist yet. Instead of raising a `KeyError`, it returns a
    `Placeholder` object. This allows the script to run to completion, and at
    the end, the proxy can dump a YAML template containing all the keys that
    were requested but not found.

    Attributes:
        missing: A dictionary to store the keys of missing parameters
            and their inferred default values.
        resolved: A set to keep track of keys that have already been resolved
            to avoid redundant processing.
        active: A flag to control the proxy's behavior. When active, it
            returns placeholders for missing keys. When inactive, it raises a
            `KeyError`.
    """

    def __init__(self, config: Config = None) -> None:
        """Initializes the ConfigProxy instance."""
        self.config = config
        self.missing: Dict[str, Any] = {}
        self.resolved: Set[str] = set()
        self.active: bool = True

    def resolve_missing(self, key: str) -> Any:
        """Handles requests for missing configuration keys.

        If the proxy is active, it returns a `Placeholder` for the given key.
        If the proxy is inactive, it raises a `KeyError`.

        Args:
            key: The dot-separated configuration key being requested.

        Returns:
            A `Placeholder` object if the proxy is active.

        Raises:
            KeyError: If the proxy is not active and a missing key is requested.
        """
        if not self.active:
            raise KeyError(f"Missing config key: {key}")

        print(f"Missing key '{key}' accessed. Using ConfigProxy to resolve.")
        if key in self.missing:
            # If the key has already been resolved, return the stored value.
            print(f"Key '{key}' already resolved. Returning stored value.")
            return self.missing[key]
        # Create a new Placeholder for the missing key.
        print(f"Creating Placeholder for missing key '{key}'.")
        return Placeholder(key, self)

    def infer_from_signature(self, key: str) -> Any:
        """Tries to infer a default value for a key by inspecting the call stack.

        This method finds the function that requested the missing key and checks
        its signature for a parameter matching the key. If a default value is
        found, it is used. Otherwise, it infers a default from type hints.

        The search for the function is performed in the following order:
        1.  Method on a class instance (`self`).
        2.  Global function in the module.
        3.  Local function (e.g., a nested function).

        Args:
            key: The dot-separated configuration key (e.g., 'model.name').

        Returns:
            An inferred default value, or `None` if no default could be found.
        """
        param_key = key.split(".")[-1]
        # We inspect the call stack, skipping the frames for this method and _resolve.
        for frame_info in inspect.stack()[2:]:
            try:
                frame = frame_info.frame
                func = None

                # 1. Prioritize finding a method on a class instance.
                if "self" in frame.f_locals:
                    instance = frame.f_locals["self"]
                    if hasattr(instance, frame_info.function):
                        func = getattr(instance, frame_info.function)

                # 2. Fallback to global functions.
                if not func and frame_info.function in frame.f_globals:
                    func = frame.f_globals[frame_info.function]

                # 3. Fallback to local functions (like nested functions).
                if not func and frame_info.function in frame.f_locals:
                    func = frame.f_locals[frame_info.function]

                if func:
                    sig = inspect.signature(func)
                    if param_key in sig.parameters:
                        param = sig.parameters[param_key]
                        if param.default is not inspect.Parameter.empty:
                            return param.default
                        # If param exists but has no default, infer from type and stop.
                        return self._infer_default_from_type(param.annotation)

            except (AttributeError, KeyError, ValueError, TypeError):
                # Some objects (e.g., built-ins) are not inspectable, so we
                # safely continue to the next frame.
                continue
        return None

    def _infer_default_from_type(self, param_type: Any) -> Any:
        """Provides a basic default value based on a given type annotation.

        Args:
            param_type: The type annotation of a parameter (e.g., `int`, `str`).

        Returns:
            A default value corresponding to the type (e.g., 1 for `int`,
            "" for `str`), or `None` if the type is not recognized.
        """
        if param_type is int:
            return 1
        if param_type is float:
            return 0.0
        if param_type is bool:
            return False
        if param_type is str:
            return ""
        if param_type is list:
            return []
        if param_type is dict:
            return {}
        return None

    def has_missing(self) -> bool:
        """Checks if any missing configuration keys were tracked.

        Returns:
            `True` if there are any missing keys, `False` otherwise.
        """
        return bool(self.missing)

    def activate(self) -> None:
        """Activates the proxy.

        Once activated, the proxy will return placeholders for missing keys.
        """
        self.active = True

    def deactivate(self) -> None:
        """Deactivates the proxy.

        Once deactivated, any further attempts to access missing keys will
        result in a `KeyError`.
        """
        self.active = False

    def __getstate__(self):
        """Returns the object's state for pickling.
        Required for deepcopy to work with this custom class.
        """
        return {
            "missing": self.missing,
            "resolved": self.resolved,
            "active": self.active,
        }

    def __setstate__(self, state):
        """Restores the object's state from pickling.
        Required for deepcopy to work with this custom class.
        """
        self.missing = state["missing"]
        self.resolved = state["resolved"]
        self.active = state["active"]
