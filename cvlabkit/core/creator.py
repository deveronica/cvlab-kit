# cvlabkit/core/creator.py

"""Manages dynamic creation of agents and components based on configuration."""

import ast
import importlib
import inspect
import pkgutil
import re
from typing import Any, Callable, Dict, Tuple

from cvlabkit.component import base as component_base
from cvlabkit.core.agent import Agent
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
            if hasattr(cls, "__module__") and cls.__module__.startswith(
                "cvlabkit.component.base"
            ):
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
            raise AttributeError(
                f"No base class found for component category '{category}'."
            )

        return _ComponentCategoryLoader(self.cfg, category, base_class)


class _BaseLoader:
    """Base class for all implementation loaders, providing common utilities."""

    def __init__(self, cfg: Config, category: str):
        """Initializes the base loader."""
        self.cfg = cfg
        self.category = category
        self.implementations: Dict[str, type] = {}

    def _get_component_info(self, config_value: Any) -> Tuple[str, Config]:
        """Parses a configuration value to extract the implementation name and
        its specific parameters using abstract syntax trees (AST).
        This handles simple names, function-call syntax, and dictionary-based configs.
        """

        def _safe_eval_node(node):
            """Safely evaluates an AST node to a Python literal,
            treating unquoted names as strings.
            """
            try:
                # Handles numbers, strings, lists, dicts, tuples, True, False, None
                return ast.literal_eval(node)
            except ValueError:
                # If literal_eval fails, it might be an unquoted string like 'train'
                if isinstance(node, ast.Name):
                    # Return the name's ID as a string.
                    # e.g., for `split=train`, node.id will be 'train'.
                    return node.id
                # If it's not a simple name, it's an unsupported expression.
                raise ValueError(
                    f"Unsupported expression in config DSL: {ast.dump(node)}"
                )

        if isinstance(config_value, str):
            tree = ast.parse(config_value, mode="eval")
            if isinstance(tree.body, ast.Call):  # e.g., "resize(size=128)"
                impl_name = tree.body.func.id
                kwargs = {
                    kw.arg: _safe_eval_node(kw.value) for kw in tree.body.keywords
                }
                return impl_name, Config(kwargs)
            elif isinstance(tree.body, ast.Name):  # e.g., "resnet18"
                return tree.body.id, Config({})

        elif isinstance(config_value, dict):  # e.g., { _type: "resnet18", ... }
            if "_type" not in config_value:
                raise ValueError(
                    f"Dictionary-based config for '{self.category}' must have a '_type' key.\n"
                    f"Example:\n"
                    f"  {self.category}:\n"
                    f"    _type: component_name\n"
                    f"    param1: value1\n"
                    f"    param2: value2\n"
                    f"Current config: {config_value}"
                )
            impl_name = config_value.pop("_type")
            return impl_name, Config(config_value)

        raise TypeError(
            f"Unsupported config format for '{self.category}': {type(config_value)}"
        )

    def _resolve_placeholders_recursive(self, value: Any) -> Any:
        """Recursively traverses a data structure and resolves placeholders
        like {{key}} using values from the main configuration.
        """
        if isinstance(value, str):
            if "{{" not in value:
                return value

            # Handle cases where the entire string is a placeholder,
            # which might resolve to a non-string value (e.g., a list or dict).
            match = re.fullmatch(r"\s*{{s*(.*?)s*}}\s*", value)
            if match:
                key = match.group(1).strip()
                return self.cfg.get(key)  # Return the raw value

            # Otherwise, substitute placeholders within the string.
            return re.sub(
                r"{{s*(.*?)s*}}", lambda m: str(self.cfg.get(m.group(1).strip())), value
            )

        elif isinstance(value, dict):
            return {
                k: self._resolve_placeholders_recursive(v) for k, v in value.items()
            }
        elif isinstance(value, list):
            return [self._resolve_placeholders_recursive(v) for v in value]
        else:
            return value

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

        # 1. Merge all configurations
        final_params = self.cfg.to_dict()
        final_params.update(component_cfg.to_dict())
        final_params.update(runtime_kwargs)

        # 2. Resolve all placeholders recursively
        resolved_params = self._resolve_placeholders_recursive(final_params)

        # 3. Create the final config object from the resolved parameters
        final_cfg = Config(resolved_params, proxy=self.cfg.proxy)

        sig = inspect.signature(constructor)
        constructor_kwargs = dict(kwargs)

        # 4. Prepare constructor arguments, injecting dependencies as needed
        for param_name in sig.parameters:
            if param_name == "self":
                continue
            elif param_name == "component_creator":
                constructor_kwargs["component_creator"] = component_creator_instance
            elif param_name in resolved_params:
                constructor_kwargs[param_name] = resolved_params[param_name]

        return constructor(final_cfg, *args, **constructor_kwargs)


class _ComponentCategoryLoader(_BaseLoader):
    """Loads implementations for a specific component category (e.g., 'model')."""

    def __init__(self, cfg: Config, category: str, base_class: type):
        super().__init__(cfg, category)
        self.base_class = base_class
        self.implementations: Dict[Tuple[str, str], type] = {}

    def _load_implementation(self, impl_name: str) -> type:
        # This method is correct and does not need changes.
        # Keep your original _load_implementation method here.
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
                print(
                    f"[Creator] Warning: Component '{impl_name}' in module '{module_full_name}' does not inherit from '{self.base_class.__name__}'. Importing anyway as it's the only concrete class in the module."
                )

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
                    # Skip legacy folders
                    if name != "legacy":
                        available_modules.append(name)
            except (ModuleNotFoundError, IndexError):
                pass

            if available_modules:
                raise ValueError(
                    f"Component module '{impl_name}' not found for category '{self.category}'. Available modules: {', '.join(available_modules)}"
                )
            else:
                raise ValueError(
                    f"Component module '{impl_name}' not found for category '{self.category}'. No modules found in {package_path}."
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load component '{impl_name}' from '{module_full_name}': {e}"
            )

    def _create_from_dsl(self, dsl_string: str, *args, **kwargs) -> Any:
        """Parses a pipeline DSL string, creates a list of component instances,
        and wraps them in a 'Compose' component for that category.
        """
        component_instances = []
        component_dsls = [s.strip() for s in dsl_string.split("|")]

        for component_dsl in component_dsls:
            impl_name, component_cfg = self._get_component_info(component_dsl)
            constructor = self._load_implementation(impl_name)
            # Pass runtime args only if the component's constructor accepts them
            sig = inspect.signature(constructor)
            if (
                "params" in sig.parameters or "model" in sig.parameters
            ):  # A bit of a hack for optimizer
                instance = self._create_instance(
                    constructor, component_cfg, *args, **kwargs
                )
            else:
                instance = self._create_instance(constructor, component_cfg)

            component_instances.append(instance)

        # Dynamically find and instantiate the 'Compose' class for the category
        try:
            compose_constructor = self._load_implementation("compose")
            # The 'Compose' component's __init__ should accept the list of components
            return compose_constructor(self.cfg, component_instances)
        except ValueError as e:
            raise ValueError(
                f"Failed to create a composite component for category '{self.category}'. "
                f"A 'compose' implementation is required for DSL pipelines. Details: {e}"
            )

    def __getattr__(self, option: str) -> Callable[..., Any]:
        """Handles calls for named components, e.g., `create.transform.weak()`.
        This returns a callable that can accept runtime arguments like `model.parameters()`.
        """
        key = f"{self.category}.{option}"
        config_value = self.cfg.get(key)
        if not config_value:
            raise ValueError(f"Missing configuration for '{key}'.")

        # This lambda ensures that any runtime arguments (*a, **kw) are passed along.
        def creator_lambda(*args, **kwargs):
            if isinstance(config_value, str) and "|" in config_value:
                # DSL pipelines do not accept runtime args.
                return self._create_from_dsl(config_value)

            # For single components, pass the runtime args through.
            impl_name, component_cfg = self._get_component_info(config_value)
            constructor = self._load_implementation(impl_name)
            return self._create_instance(constructor, component_cfg, *args, **kwargs)

        return creator_lambda

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Handles calls for a single top-level component, e.g., `create.optimizer()`.
        This method correctly passes runtime arguments.
        """
        config_value = self.cfg.get(self.category)
        if not config_value:
            raise ValueError(
                f"Missing configuration for top-level category '{self.category}'."
            )

        if isinstance(config_value, str) and "|" in config_value:
            return self._create_from_dsl(config_value)

        impl_name, component_cfg = self._get_component_info(config_value)
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
        self.implementations: Dict[str, type] = {}  # Cache for loaded agents

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
                if (
                    cls.__module__ == module.__name__
                    and issubclass(cls, self.base_class)
                    and not inspect.isabstract(cls)
                ):
                    self.implementations[impl_name] = cls
                    return cls

            raise ValueError(
                f"No concrete class inheriting from 'Agent' found in module '{module_full_name}'."
            )

        except ModuleNotFoundError:
            available_modules = []
            try:
                package = importlib.import_module(package_path)
                for _, name, _ in pkgutil.iter_modules(package.__path__):
                    # Skip legacy folders
                    if name != "legacy":
                        available_modules.append(name)
            except (ModuleNotFoundError, IndexError):
                pass

            if available_modules:
                raise ValueError(
                    f"Agent implementation '{impl_name}' not found. Available: {', '.join(available_modules)}"
                )
            else:
                raise ValueError(
                    f"Agent implementation '{impl_name}' not found. No modules found in {package_path}."
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load agent '{impl_name}' from '{module_full_name}': {e}"
            )

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
