import yaml
from pathlib import Path
from collections.abc import Mapping


def load_config(*paths) -> dict:
    config = {}
    for path in paths:
        with open(path) as f:
            override = yaml.safe_load(f)
            config = deep_merge(config, override)
    return config


def deep_merge(base, override):
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, Mapping) and k in result:
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result
