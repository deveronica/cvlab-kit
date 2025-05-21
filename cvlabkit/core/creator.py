import importlib
import inspect

from cvlabkit.core.config import Config


class Creator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __getattr__(self, category: str):
        # Dynamically return a loader for a given category (e.g., agent, model, optimizer)
        return _CategoryLoader(category, self.cfg)


class _CategoryLoader:
    def __init__(self, category: str, cfg: Config):
        self.category = category
        self.cfg = cfg

        # Determine base module path
        if category == "agent":
            # agent interface is in core.agent
            base_module_path = "cvlabkit.core.agent"
        else:
            # component interface is in component.base.{category}
            base_module_path = f"cvlabkit.component.base.{category}"

        # Try importing base module to extract interface classes
        try:
            base_module = importlib.import_module(base_module_path)
            base_class = next(
                cls for _, cls in inspect.getmembers(base_module, inspect.isclass)
                if cls.__module__ == base_module.__name__
            )
            self.base_class = base_class
        except (ModuleNotFoundError, StopIteration):
            self.base_class = None

        # Determine implementation package path
        if category == "agent":
            impl_package = "cvlabkit.agent"
        else:
            impl_package = f"cvlabkit.component.{category}"

        # Try importing implementation package and scan for subclasses
        try:
            impl_package_module = importlib.import_module(impl_package)
        except ModuleNotFoundError:
            raise ValueError(f"[Creator] Package '{impl_package}' not found.")

        # Search all .py files in the implementation package for valid subclasses
        self.implementations = {}
        for _, module_name, ispkg in pkgutil.iter_modules(impl_package_module.__path__):
            if ispkg:
                continue

            try:
                module = importlib.import_module(f"{impl_package}.{module_name}")
                for name, cls in inspect.getmembers(module, inspect.isclass):
                    # Skip if class is defined in a different module (avoid transitive imports)
                    if cls.__module__ != module.__name__:
                        continue
                    if self.base_class:
                        if issubclass(cls, self.base_class) and cls is not self.base_class:
                            self.implementations[module_name] = cls
                    else:
                        self.implementations[module_name] = cls
            except Exception as e:
                print(f"[Creator] Skipped '{module_name}': {e}")

    def __getattr__(self, option: str) -> Callable[..., Any]:
        # Resolve configuration optionable key like "model.backbone"
        key = f"{self.category}.{option}"
        impl_name = self.cfg.get(key)

        # Raise error with suggestions if implementation is invalid and show importable implementations
        if impl_name not in self.implementations:
            available = ", ".join(sorted(self.implementations.keys()))
            raise ValueError(f"[Creator] No implementation for '{impl_name}' in '{key}'. Available: {available}")

        constructor = self.implementations[impl_name]
        def _call(**kwargs):
            return constructor(self.cfg, **kwargs)

        return _call

    def __call__(self, **kwargs) -> Any:
        # Handle direct call like create.agent(), using cfg.category as the key
        impl_name = self.cfg.get(self.category)

        # Raise error if the value is not one of the discovered implementations
        if impl_name not in self.implementations:
            available = ", ".join(sorted(self.implementations.keys()))
            raise ValueError(f"[Creator] No implementation for '{impl_name}' in '{self.category}'. Available: {available}")

        constructor = self.implementations[impl_name]
        return constructor(self.cfg, **kwargs)

    