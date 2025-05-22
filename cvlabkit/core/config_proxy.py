import inspect
import yaml
from pathlib import Path


class Placeholder:
    def __init__(self, key, proxy):
        self.key = key
        self.proxy = proxy

    def __getattr__(self, name):
        try:
            return getattr(self._resolve(), name)
        except Exception:
            self.proxy.missing[self.key] = None
            return None

    def __getitem__(self, item):
        try:
            return self._resolve()[item]
        except Exception:
            self.proxy.missing[self.key] = {}
            return {}

    def __call__(self, *args, **kwargs):
        try:
            return self._resolve()(*args, **kwargs)
        except Exception:
            self.proxy.missing[self.key] = lambda *a, **kw: None
            return self.proxy.missing[self.key]

    def __repr__(self):
        return repr(self._resolve())

    def __str__(self):
        try:
            return str(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = "<missing>"
            return "<missing>"

    def __int__(self):
        try:
            return int(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = 1
            return 1

    def __float__(self):
        try:
            return float(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = 0.0
            return 0.0

    def __bool__(self):
        try:
            return bool(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = False
            return False

    def __len__(self):
        try:
            return len(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = []
            return 0

    def __iter__(self):
        try:
            return iter(self._resolve())
        except Exception:
            self.proxy.missing[self.key] = []
            return iter([])

    def __fspath__(self):
        val = self._resolve()
        if isinstance(val, (str, bytes, type(Path(".")))):
            return val
        self.proxy.missing[self.key] = "/missing/path"
        return "/missing/path"

    def _resolve(self):
        if self.key not in self.proxy.resolved:
            self.proxy.resolved.add(self.key)
            value = self.proxy.infer_from_signature(self.key)
            self.proxy.missing[self.key] = value
            return value
        return self.proxy.missing.get(self.key, None)


class ConfigProxy:
    """Tracks missing config keys during dry-run and emits a YAML template.

    During dry-run  ➜  allow missing keys and guess defaults.
    After validation ➜  deactivate(): further missing keys raise KeyError.
    """

    def __init__(self):
        self.missing = {}
        self.resolved = set()
        self.active = True

    def resolve_missing(self, key):
        if not self.active:
            raise KeyError(f"Missing config key: {key}")
        if key in self.missing:
            return self.missing[key]
        return Placeholder(key, self)

    def infer_from_signature(self, key):
        for frame_info in inspect.stack()[2:]:
            func = frame_info.function
            frame = frame_info.frame
            module = inspect.getmodule(frame)

            if not module or func not in module.__dict__:
                continue

            try:
                target_func = module.__dict__[func]
                sig = inspect.signature(target_func)

                param_key = key.split(".")[-1]
                if param_key in sig.parameters:
                    param = sig.parameters[param_key]
                    if param.default is not inspect._empty:
                        return param.default
                    return self._infer_default_from_type(param.annotation)
            except Exception:
                continue

        return None

    def _infer_default_from_type(self, _type):
        try:
            if issubclass(_type, int):
                return 1
            if issubclass(_type, float):
                return 0.0
            if issubclass(_type, bool):
                return False
            if issubclass(_type, str):
                return ""
            if issubclass(_type, list):
                return []
            if issubclass(_type, dict):
                return {}
        except TypeError:
            pass
        return None

    def has_missing(self):
        return len(self.missing) > 0

    def deactivate(self):
        self.active = False

    def dump_template(self, path, original=None):
        full = self._flatten(original or {})
        full.update(self.missing)
        nested = self._unflatten(full)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(nested, f)

    def _flatten(self, d, parent_key="", sep="."):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten(v, new_key, sep))
            else:
                items[new_key] = v
        return items

    def _unflatten(self, flat, sep="."):
        result = {}
        for k, v in flat.items():
            parts = k.split(sep)
            cur = result
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = v
        return result