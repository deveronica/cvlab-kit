# cvlabkit/core/creator.py

"""Manages dynamic creation of agents and components based on configuration."""

import importlib
import inspect
import pkgutil
import re
from typing import Any, Callable, Dict, Optional, Tuple
import ast
import os

from cvlabkit.core.agent import Agent
from cvlabkit.component import base as component_base
from cvlabkit.core.config import Config

# Regex to parse a string like "name(key1=value1, key2='string_val')"
# - Group 1: The component name (e.g., "cifar10")
# - Group 2: The comma-separated parameters (e.g., "split=train, shuffle=True")
PARAM_PATTERN = re.compile(r"(\w+)\((.*)\)")
# Regex to parse "key=value" pairs from a parameter string.
KEY_VALUE_PATTERN = re.compile(r"(\w+)\s*=\s*([^,]+)")


class Creator:
    """Central factory for creating agents and components from configuration.

    This class dynamically instantiates all necessary objects for an experiment
    based on a configuration file, following a convention-over-configuration approach.

    Key Responsibilities:
    - Creates the main `Agent` for the experiment.
    - Provides a `ComponentCreator` to the agent, enabling it to build
      other components like models, optimizers, and data loaders.

    Usage Example:
        cfg = Config("config.yaml")
        create = Creator(cfg)
        agent = create.agent() # Agent can now use create.* to build its parts.
    """

    def __init__(self, cfg: Config):
        """Initializes the Creator with the main configuration.

        Args:
            cfg: The configuration object driving the creation process.
        """
        self.cfg = cfg
        self.component_creator = ComponentCreator(cfg)

    def __getattr__(self, category: str) -> Any:
        """Provides access to the appropriate loader for a given component category.

        This method is the entry point for creating any component:
        - `create.agent()` is handled by `_AgentLoader`.
        - `create.model()`, `create.optimizer()`, etc., are delegated to `ComponentCreator`.

        Args:
            category: The component category name (e.g., 'agent', 'model').

        Returns:
            A loader object capable of creating instances for that category.
        """
        if category == "agent":
            # Agents are special. They get their own loader which injects the
            # component_creator, allowing the agent to build its own components.
            return _AgentLoader(self.cfg, self.component_creator)
        else:
            # All other component types are handled by the ComponentCreator.
            return getattr(self.component_creator, category)


class ComponentCreator:
    """Creator Implementation for creating non-agent specific components
    (e.g., model, optimizer, dataset, etc.).

    This class discovers and instantiates components from the `cvlabkit.component`
    package based on the provided configuration.
    """

    def __init__(self, cfg: Config):
        """Initializes the ComponentCreator.

        Args:
            cfg: The main configuration object.
        """
        self.cfg = cfg
        self._base_classes = self._get_all_base_classes()

    def _get_all_base_classes(self) -> Dict[str, type]:
        """Finds and returns all component base classes from `cvlabkit.component.base`.

        This method inspects the `base` module and creates a mapping from a
        category name (e.g., "model") to its base class (e.g., `Model`).
        This avoids hardcoding the relationship between categories and their
        base classes.
        """
        base_classes = {}
        for name, cls in inspect.getmembers(component_base, inspect.isclass):
            if hasattr(cls, "__module__") and cls.__module__.startswith("cvlabkit.component.base"):
                base_classes[name.lower()] = cls
        return base_classes

    # TODO: Add support for recursive agent creation, where agents can create other agents.
    def __getattr__(self, category: str) -> "_ComponentCategoryLoader":
        """Returns a specialized loader for a specific component category.

        This method is called when `create.model` or `create.optimizer` is accessed.

        Args:
            category: The component category name (e.g., 'model', 'optimizer').

        Returns:
            A loader instance for the specified category.

        Raises:
            AttributeError: If the category name is 'agent',
                            or if no base class is found for the category.
        """
        if category == "agent":
            raise AttributeError(
                f"'{type(self).__name__}' can not support 'agent' category."
            )
        
        base_class = self._base_classes.get(category)
        if base_class is None:
             raise AttributeError(f"No base class found for component category '{category}'.")

        return _ComponentCategoryLoader(self.cfg, category, base_class)


class _BaseLoader:
    """Base class for all implementation loaders, providing common utilities."""

    def __init__(self, cfg: Config, category: str):
        """Initializes the base loader."""
        self.cfg = cfg
        self.category = category
        self.implementations: Dict[str, type] = {}

    def _get_component_info(self, config_value: Any) -> Tuple[Optional[str], Config]:
        """Parses a configuration value to extract the implementation name and
        its specific parameters.
        
        Supports two main syntaxes:
        1. A string with parameters: "name(key1=val1, key2='val2')"

        Args:
            config_value: The configuration value to parse, which can be a string or a dictionary.

        Returns:
            A tuple containing:
                - The implementation name (str) or None if not specified.
                - A Config object with the parameters for this component.
        """

        PARAM_PATTERN = re.compile(r'^\s*([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*$')

        def _coerce(node: ast.AST) -> Any:
            # Fast path: safe literal evaluation
            try:
                return ast.literal_eval(node)
            except Exception:
                pass
            # Bare identifiers: map booleans/none, otherwise treat as string
            if isinstance(node, ast.Name):
                name = node.id
                low = name.lower()
                if low == "true":
                    return True
                if low == "false":
                    return False
                if low in ("none", "null"):
                    return None
                return name
            # Attribute chains: a.b.c -> "a.b.c"
            if isinstance(node, ast.Attribute):
                parts = []
                cur = node
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                    return ".".join(reversed(parts))
            # Containers: recurse
            if isinstance(node, ast.List):
                return [_coerce(e) for e in node.elts]
            if isinstance(node, ast.Tuple):
                return tuple(_coerce(e) for e in node.elts)
            if isinstance(node, ast.Set):
                return set(_coerce(e) for e in node.elts)
            if isinstance(node, ast.Dict):
                return {_coerce(k): _coerce(v) for k, v in zip(node.keys, node.values)}
            raise ValueError(f"Unsupported expression in config: {ast.dump(node, include_attributes=False)}")

        def _parse_kwargs(params: str) -> Dict[str, Any]:
            call = ast.parse(f"f({params})", mode="eval").body
            if not isinstance(call, ast.Call):
                raise ValueError("Invalid parameter string")
            if call.args:
                raise ValueError("Positional arguments are not supported")
            out: Dict[str, Any] = {}
            for kw in call.keywords:
                if kw.arg is None:
                    raise ValueError("**kwargs expansion is not supported")
                out[kw.arg] = _coerce(kw.value)
            return out

        if isinstance(config_value, str):
            m = PARAM_PATTERN.match(config_value)
            if not m:
                return config_value.strip(), Config({})
            name, params_str = m.group(1), m.group(2)
            if not params_str or params_str.strip() == "":
                return name, Config({})
            return name, Config(_parse_kwargs(params_str))
        return None, self.cfg

    def _create_instance(
        self, constructor: Callable, component_cfg: Config, *args: Any, **kwargs: Any
        ) -> Any:
        """Creates a component instance, injecting dependencies.

        This method inspects the component's constructor (`__init__`) and provides
        only the arguments it explicitly accepts, including the merged `cfg` object.

        Args:
            constructor: The component class constructor.
            component_cfg: Configuration specific to this component instance.
            *args: Positional arguments to pass to the constructor.
            **kwargs: Keyword arguments to pass to the component's constructor.

        Returns:
            An instance of the component.
        """
        runtime_kwargs = dict(kwargs)
        component_creator_instance = runtime_kwargs.pop("component_creator", None)

        final_params = self.cfg.to_dict()
        final_params.update(component_cfg.to_dict())
        final_params.update(runtime_kwargs)
        final_cfg = Config(final_params, proxy=self.cfg.proxy)

        sig = inspect.signature(constructor)
        constructor_kwargs = dict(kwargs)

        for param_name in sig.parameters:
            if param_name == "self":
                continue
            elif param_name == "component_creator":
                constructor_kwargs["component_creator"] = component_creator_instance
            elif param_name in final_params:
                constructor_kwargs[param_name] = final_params[param_name]

        return constructor(final_cfg, *args, **constructor_kwargs)


class _ComponentCategoryLoader(_BaseLoader):
    """Loads implementations for a specific component category (e.g., 'model')."""

    def __init__(self, cfg: Config, category: str, base_class: type):
        """Initializes the loader for a specific component category.

        Prepares for lazy discovery of implementations for this category.

        Args:
            category: The component category name.
            cfg: The configuration object.
            base_class: The base class for this component category.
        """
        super().__init__(cfg, category)
        self.base_class = base_class
        self.implementations: Dict[Tuple[str, str], type] = {}

    def _load_implementation(self, impl_name: str) -> type:
        """Lazily loads and returns the class for a given implementation name.

        Applies smart loading logic: prefers base_class subclass, falls back to
        single class with a warning.

        Args:
            impl_name: The name of the implementation to load.

        Returns:
            The loaded class constructor.

        Raises:
            ValueError: If the module is not found, or if multiple suitable classes
                        are found without a clear base class inheritance.
        """
        cache_key = (self.category, impl_name)
        if cache_key in self.implementations:
            return self.implementations[cache_key]

        package_path = f"cvlabkit.component.{self.category}"
        module_full_name = f"{package_path}.{impl_name}"

        try:
            module = importlib.import_module(module_full_name)

            candidate_classes = []
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ == module.__name__ and not inspect.isabstract(cls):
                    candidate_classes.append(cls)

            found_class = None
            for cls in candidate_classes:
                if issubclass(cls, self.base_class):
                    found_class = cls
                    break
            
            if found_class is None and len(candidate_classes) == 1:
                found_class = candidate_classes[0]
                print(f"[Creator] Warning: Component '{impl_name}' in module '{module_full_name}' does not inherit from '{self.base_class.__name__}'. Importing anyway as it's the only concrete class in the module.")
            
            if found_class:
                self.implementations[cache_key] = found_class
                return found_class
            else:
                if len(candidate_classes) > 1:
                    raise ValueError(
                        f"Module '{module_full_name}' contains multiple concrete classes "
                        f"and none inherit from '{self.base_class.__name__}'. "
                        f"Please ensure the component class inherits from the correct base class or is the only class defined."
                    )
                else:
                    raise ValueError(
                        f"No suitable concrete class found in module '{module_full_name}'. "
                        f"Expected a class inheriting from '{self.base_class.__name__}' or a single concrete class."
                    )

        except ModuleNotFoundError:
            available_modules = []
            try:
                package = importlib.import_module(package_path)
                for _, name, _ in pkgutil.iter_modules(package.__path__):
                    available_modules.append(name)
            except (ModuleNotFoundError, IndexError):
                pass

            if available_modules:
                raise ValueError(f"Component module '{impl_name}' not found for category '{self.category}'. Available modules: {', '.join(available_modules)}")
            else:
                raise ValueError(f"Component module '{impl_name}' not found for category '{self.category}'. No modules found in {package_path}.")
        except Exception as e:
            raise ValueError(f"Failed to load component '{impl_name}' from '{module_full_name}': {e}")

    def __getattr__(self, option: str) -> Callable[..., Any]:
        """Handles calls for named components, e.g., `create.model.generator()`.

        This method is used when the config groups multiple components under one category.

        Args:
            option: The specific named option for the component.

        Returns:
            A callable that, when invoked, creates an instance of the component.

        Raises:
            ValueError: If configuration is missing or implementation cannot be determined.
        """
        key = f"{self.category}.{option}"
        config_value = self.cfg.get(key)
        if not config_value:
            raise ValueError(f"Missing configuration for '{key}'.")

        impl_name, component_cfg = self._get_component_info(config_value)
        if not impl_name:
            raise ValueError(f"Could not determine implementation for '{key}'.")

        constructor = self._load_implementation(impl_name)
        
        return lambda *a, **kw: self._create_instance(constructor, component_cfg, *a, **kw)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Handles calls for a single component, e.g., `create.model()`.

        This method is used when the config specifies a single implementation for the category.

        Args:
            *args: Positional arguments to pass to the component's constructor.
            **kwargs: Keyword arguments to pass to the component's constructor.

        Returns:
            An instance of the component.

        Raises:
            ValueError: If configuration is invalid or implementation cannot be determined.
        """
        config_value = self.cfg.get(self.category)
        if not isinstance(config_value, (str, dict)):
            raise ValueError(
                f"Cannot create '{self.category}' directly. "
                f"Configuration must be a string or a dictionary."
            )

        impl_name, component_cfg = self._get_component_info(config_value)
        if not impl_name:
            raise ValueError(f"Could not determine implementation for '{self.category}'.")

        constructor = self._load_implementation(impl_name)
        
        return self._create_instance(constructor, component_cfg, *args, **kwargs)


class _AgentLoader(_BaseLoader):
    """Specialized loader for Agent classes."""

    def __init__(self, cfg: Config, component_creator: ComponentCreator):
        """Initializes the agent loader.

        Args:
            cfg: The configuration object.
            component_creator: The ComponentCreator instance for building other components.
        """
        super().__init__(cfg, "agent")
        self.component_creator = component_creator
        self.base_class = Agent
        self.implementations: Dict[str, type] = {} # Cache for loaded agents

    def _load_implementation(self, impl_name: str) -> type:
        """Lazily loads and returns the class for a given agent implementation name.

        Args:
            impl_name: The name of the implementation to load.

        Returns:
            The loaded class constructor.

        Raises:
            ValueError: If the module is not found or no suitable class is found.
        """
        if impl_name in self.implementations:
            return self.implementations[impl_name]

        package_path = "cvlabkit.agent"
        module_full_name = f"{package_path}.{impl_name}"

        try:
            module = importlib.import_module(module_full_name)
            
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ == module.__name__ and issubclass(cls, self.base_class) and not inspect.isabstract(cls):
                    self.implementations[impl_name] = cls
                    return cls
            
            raise ValueError(f"No concrete class inheriting from 'Agent' found in module '{module_full_name}'.")

        except ModuleNotFoundError:
            available_modules = []
            try:
                package = importlib.import_module(package_path)
                for _, name, _ in pkgutil.iter_modules(package.__path__):
                    available_modules.append(name)
            except (ModuleNotFoundError, IndexError):
                pass
            
            if available_modules:
                raise ValueError(f"Agent implementation '{impl_name}' not found. Available: {', '.join(available_modules)}")
            else:
                raise ValueError(f"Agent implementation '{impl_name}' not found. No modules found in {package_path}.")
        except Exception as e:
            raise ValueError(f"Failed to load agent '{impl_name}' from '{module_full_name}': {e}")

    def __call__(self, **kwargs: Any) -> Agent:
        """Creates the main agent instance specified by `cfg.agent`.

        Args:
            **kwargs: Keyword arguments to pass to the agent's constructor.

        Returns:
            An instance of the Agent.

        Raises:
            ValueError: If the 'agent' configuration key is not specified,
                        or if the specified agent implementation is not found.
        """
        impl_name = self.cfg.get("agent")
        if not impl_name:
            raise ValueError("Configuration key 'agent' is not specified.")

        constructor = self._load_implementation(impl_name)
        
        kwargs["component_creator"] = self.component_creator
        return self._create_instance(constructor, Config({}), **kwargs)