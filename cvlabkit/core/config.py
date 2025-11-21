"""This module defines the `Config` class, a central component for managing
experiment configurations within the cvlab-kit framework.

It provides functionalities for loading configurations from YAML files or
dictionaries, accessing parameters using both dictionary-style and attribute-style
syntax, merging configurations, and expanding configurations for grid search
experiments. The `Config` class also integrates with `ConfigProxy` for
config validation during dry runs.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import yaml

from cvlabkit.core.config_proxy import ConfigProxy


class Config:
    """A configuration class that handles loading, accessing, and manipulating parameters.

    This class provides a unified interface for managing experiment configurations.
    It can load settings from a YAML file or a dictionary, provides attribute-style
    access to parameters, and supports advanced features like grid search expansion
    and special syntax parsing for component-specific parameters.

    Attributes:
        _data (Dict[str, Any]): The internal dictionary storing the configuration
            parameters.
        proxy (ConfigProxy): An object that tracks missing configuration keys
            during a dry run to help generate a template. It is primarily used
            by the `__getattr__` method when a requested key is not found.
    """

    def __init__(
        self, source: Union[str, Dict[str, Any]], proxy: ConfigProxy = None
    ) -> None:
        """Initializes the Config object.

        Args:
            source: A file path to a YAML file (str) or a dictionary containing
                the configuration parameters (Dict[str, Any]).
            proxy: An optional `ConfigProxy` instance for advanced scenarios like
                generating configuration templates during dry runs. If not provided,
                a new `ConfigProxy` will be initialized.

        Raises:
            TypeError: If the source is not a file path (str) or a dictionary.
        """
        if isinstance(source, str):
            # Load configuration from a YAML file.
            with open(source) as f:
                self._data = yaml.load(f, Loader=yaml.FullLoader)
        elif isinstance(source, dict):
            # Use the provided dictionary directly, creating a deep copy to avoid
            # external modifications.
            self._data = deepcopy(source)
        else:
            # Raise an error for unsupported source types.
            raise TypeError("Config expects a YAML file path (str) or a dictionary.")

        # Initialize the ConfigProxy. This proxy is used for advanced scenarios
        # like generating configuration templates during a dry run.
        self.proxy = proxy if proxy else ConfigProxy(self)

    def __getitem__(self, key: str) -> Any:
        """Enables dictionary-style access to configuration parameters.

        Allows accessing configuration values using bracket notation, e.g., `config["model.name"]`.

        Args:
            key (str): The dot-separated string representing the nested key
                (e.g., "model.params.lr").

        Returns:
            Any: The configuration value associated with the key.
        """
        return self.get(key)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieves a value using a dot-separated key.

        This method allows accessing nested configuration values using a single
        dot-separated string. For example, `config.get("model.params.lr")`.

        Args:
            key (str): A dot-separated string representing the nested key
                (e.g., "model.params.lr").
            default (Optional[Any]): The value to return if the key is not found.
                Defaults to `None`.

        Returns:
            Any: The configuration value associated with the key, or the
                specified `default` value if the key is not found.
        """
        keys = key.split(".")
        current_level = self._data
        # Traverse the nested dictionary using the split keys.
        for k in keys:
            # Check if the current level is a dictionary and contains the key.
            if isinstance(current_level, dict) and k in current_level:
                current_level = current_level[k]
            else:
                # If any part of the key path is not found, return the default.
                return default
        return current_level

    def __getattr__(self, key: str) -> Any:
        """Enables attribute-style access to configuration parameters.

        Allows accessing configuration values using dot notation, e.g., `config.model.name`.
        If the accessed attribute is a dictionary, it is wrapped in a new
        `Config` object to allow for nested attribute access.

        Args:
            key (str): The name of the attribute to access.

        Returns:
            Any: The configuration value, or a new `Config` object for nested
                dictionaries, or a `Placeholder` if the key is missing and the
                proxy is active.

        Raises:
            AttributeError: If the attribute is not a configuration key and not
                a standard Python attribute.
        """
        # Check if the key exists in the internal data.
        if key in self._data:
            value = self._data[key]
            # If the value is a dictionary, wrap it in a new Config object
            # to enable further attribute-style access.
            return Config(value, proxy=self.proxy) if isinstance(value, dict) else value

        # If the key is not found in _data, try to get it as a regular attribute
        # of the Config object itself (e.g., `proxy`).
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            # If it's not a regular attribute, then it must be a missing config key.
            # Delegate to the proxy. This is used in dry-run scenarios to track missing keys.
            if self.proxy.active:
                return self.proxy.resolve_missing(key)
            # If the proxy is not active, raise an AttributeError.
            raise AttributeError(
                f"Config has no attribute '{key}' and fast mode is not active."
            )

    def __contains__(self, key: str) -> bool:
        """Checks if a key exists in the configuration.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        """Returns a deep copy of the internal configuration data as a dictionary.

        This is useful for obtaining a mutable copy of the configuration that
        can be modified without affecting the original `Config` object.

        Returns:
            Dict[str, Any]: A deep copy of the configuration data.
        """
        return deepcopy(self._data)

    def merge(self, new_params: Dict[str, Any]) -> "Config":
        """Merges a dictionary of new parameters into the current configuration.

        This method creates a new `Config` object by combining the current
        configuration with the provided `new_params`. Parameters in `new_params`
        will overwrite existing ones at the top level.

        Args:
            new_params (Dict[str, Any]): A dictionary of parameters to merge.

        Returns:
            Config: A new `Config` object with the merged parameters.
        """
        merged_data = deepcopy(self._data)
        # Update the merged data with new parameters. Note that this is a shallow
        # merge at the top level; nested dictionaries are replaced, not merged.
        merged_data.update(new_params)
        return Config(merged_data, proxy=self.proxy)

    def expand(self) -> List["Config"]:
        """Expands the configuration into multiple `Config` objects for a grid search.

        This method identifies all parameters in the configuration that are lists
        and creates a Cartesian product of their values. Each combination results
        in a new `Config` object, effectively generating all configurations for
        a grid search experiment.

        Returns:
            List[Config]: A list of `Config` objects, one for each parameter
                combination. If no list-based parameters are found, it returns
                a list containing only the original `Config` object.
        """
        # Importing product from itertools to generate combinations.
        from itertools import product

        # Flatten the nested configuration into a single-level dictionary.
        flat_config = self._flatten(self._data)
        # Identify keys whose values are lists, indicating parameters for grid search.
        grid_keys = [k for k, v in flat_config.items() if isinstance(v, list)]

        # If no list-based parameters are found, return the original config in a list.
        if not grid_keys:
            return [self]

        # Get the values for each grid key to form combinations.
        combinations = [flat_config[k] for k in grid_keys]
        expanded_configs = []
        # Generate all possible combinations using `itertools.product`.
        for combo in product(*combinations):
            flat_instance = deepcopy(flat_config)
            # Apply the current combination of values to the flattened config.
            for key, value in zip(grid_keys, combo):
                flat_instance[key] = value
            # Unflatten the modified dictionary back into a nested structure.
            nested_config = self._unflatten(flat_instance)
            # Create a new Config object for each combination.
            expanded_configs.append(Config(nested_config, proxy=self.proxy))
        return expanded_configs

    @staticmethod
    def _flatten(
        d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flattens a nested dictionary into a single-level dictionary.

        Keys are concatenated using the specified separator (default: '.').

        Args:
            d (Dict[str, Any]): The dictionary to flatten.
            parent_key (str): The base key for the current level of recursion.
                Defaults to an empty string.
            sep (str): The separator to use for concatenating keys. Defaults to '.'.

        Returns:
            Dict[str, Any]: A flattened dictionary where keys represent the full
                path to the original nested values.
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            # Recursively flatten if the value is a dictionary.
            if isinstance(v, dict):
                items.update(Config._flatten(v, new_key, sep=sep))
            else:
                # Otherwise, add the key-value pair to the flattened items.
                items[new_key] = v
        return items

    @staticmethod
    def _unflatten(flat_dict: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
        """Converts a flattened dictionary back into a nested dictionary.

        Args:
            flat_dict (Dict[str, Any]): The flattened dictionary to unflatten.
            sep (str): The separator used to concatenate keys during flattening.
                Defaults to '.'.

        Returns:
            result (Dict[str, Any]): A nested dictionary reconstructed from
                the flattened one.
        """
        # Initialize an empty dictionary to hold the nested structure.
        result = {}
        for k, v in flat_dict.items():
            # Split the flattened key into its component parts.
            keys = k.split(sep)
            d = result
            # Traverse or create nested dictionaries until the last key.
            for part in keys[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            # Assign the value to the innermost key.
            d[keys[-1]] = v
        return result

    def dump_template(self, file_path: str) -> None:
        """Dumps the current configuration, including inferred missing keys, to a YAML file.

        This method is primarily used during dry runs to generate a template
        configuration file that includes all parameters accessed by the system,
        even those not explicitly defined in the initial config.

        Args:
            file_path (str): The path to the YAML file where the template will be saved.
        """
        # Merge the initial data with the missing keys resolved by the proxy.
        # The proxy's `missing` dictionary contains the inferred default values.
        template_data = deepcopy(self._data)
        for key, value in self.proxy.missing.items():
            # Convert dot-separated key back to nested dictionary structure
            keys = key.split(".")
            current_level = template_data
            for i, k in enumerate(keys):
                if i == len(keys) - 1:  # Last key
                    current_level[k] = value
                else:
                    if k not in current_level or not isinstance(current_level[k], dict):
                        current_level[k] = {}
                    current_level = current_level[k]

        with open(file_path, "w") as f:
            yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)

    def __getstate__(self):
        """Returns the state of the Config object for pickling/deepcopying."""
        return {"_data": self._data, "proxy": self.proxy}

    def __setstate__(self, state):
        """Restores the state of the Config object from pickling/deepcopying."""
        self._data = state["_data"]
        self.proxy = state["proxy"]
