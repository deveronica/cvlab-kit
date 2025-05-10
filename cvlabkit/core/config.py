import yaml
from copy import deepcopy

from cvlabkit.core.config_proxy import ConfigProxy


class Config:
    def __init__(self, source):
        if isinstance(source, str):
            with open(source) as f:
                self._data = yaml.safe_load(f)
        elif isinstance(source, dict):
            self._data = deepcopy(source)
        else:
            raise TypeError("Config expects a YAML path or a dictionary")

        self.proxy = ConfigProxy()

    def get(self, key):
        keys = key.split(".")
        cur = self._data
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return self.proxy.resolve_missing(key)
        return cur

    def __getattr__(self, key):
        if key in self._data:
            val = self._data[key]
            return Config(val) if isinstance(val, dict) else val
        return self.proxy.resolve_missing(key)

    def has_missing(self):
        if not self.proxy.has_missing():
            self.proxy.deactivate()
            return False
        return True

    def dump_template(self, path):
        full = self._flatten(self._data)
        full.update(self.proxy.missing)
        nested = self._unflatten(full)
        with open(path, "w") as f:
            yaml.safe_dump(nested, f)

    def to_dict(self):
        return deepcopy(self._data)

    def expand(self):
        from itertools import product

        flat = self._flatten(self._data)
        grid_keys = [k for k, v in flat.items() if isinstance(v, list)]

        if not grid_keys:
            return [self]

        values = [flat[k] for k in grid_keys]
        configs = []
        for combo in product(*values):
            flat_copy = deepcopy(flat)
            for k, v in zip(grid_keys, combo):
                flat_copy[k] = v
            nested = self._unflatten(flat_copy)
            configs.append(Config(nested))
        return configs

    @staticmethod
    def _flatten(d, parent_key="", sep="."):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(Config._flatten(v, new_key, sep))
            else:
                items[new_key] = v
        return items

    @staticmethod
    def _unflatten(flat_dict, sep="."):
        result = {}
        for k, v in flat_dict.items():
            keys = k.split(sep)
            d = result
            for part in keys[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[keys[-1]] = v
        return result
