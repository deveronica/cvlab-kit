import importlib
import inspect

from cvlabkit.core.config import Config


class Creator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __getattr__(self, category: str):
        return _CategoryLoader(category, self.cfg)


class _CategoryLoader:
    def __init__(self, category: str, cfg: Config):
        self.category = category
        self.cfg = cfg

        if category == "agent":
            base_mod = importlib.import_module("cvlabkit.core.agent")
        else:
            base_mod = importlib.import_module("cvlabkit.component.base")

        self.base_classes = {
            name: cls
            for name, cls in inspect.getmembers(base_mod, inspect.isclass)
        }

    def __call__(self, *args, **kwargs):
        value = self.cfg.get(self.category)
        cls = self._resolve_class(value)
        return cls(self.cfg, *args, **kwargs)

    def __getattr__(self, option: str):
        value = self.cfg.get(f"{self.category}.{option}")
        cls = self._resolve_class(value)
        def constructor(*args, **kwargs):
            return cls(self.cfg, *args, **kwargs)
        return constructor

    def _resolve_class(self, value):
        if self.category == "agent":
            mod_path = f"cvlabkit.agent.{value}"
        else:
            mod_path = f"cvlabkit.component.{self.category}.{value}"

        mod = importlib.import_module(mod_path)
        candidates = [
            cls for _, cls in inspect.getmembers(mod, inspect.isclass)
            if any(issubclass(cls, base) and cls is not base
                   for base in self.base_classes.values())
        ]
        candidates = [cls for cls in candidates if cls.__name__ not in self.base_classes]
        if not candidates:
            raise ValueError(f"No class in '{mod_path}' inherits from base")
        if len(candidates) > 1:
            raise ValueError(f"Multiple base-class subclasses found in '{mod_path}'")
        return candidates[0]
