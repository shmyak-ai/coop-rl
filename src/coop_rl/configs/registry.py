"""Helpers for discovering and loading experiment configs."""

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

_CONFIGS_PACKAGE = "coop_rl.configs"
_EXCLUDED_MODULES = {"__init__", "registry"}


def list_available_configs() -> tuple[str, ...]:
    """Return the available config module names."""
    package = import_module(_CONFIGS_PACKAGE)
    names = []
    for module_info in iter_modules(package.__path__):
        if module_info.ispkg or module_info.name in _EXCLUDED_MODULES:
            continue
        names.append(module_info.name)
    return tuple(sorted(names))


def get_config_module(config_name: str) -> ModuleType:
    """Import a config module by name."""
    available_configs = set(list_available_configs())
    if config_name not in available_configs:
        available = ", ".join(sorted(available_configs))
        raise ValueError(f"Unknown config '{config_name}'. Available configs: {available}")
    return import_module(f"{_CONFIGS_PACKAGE}.{config_name}")


def get_config(config_name: str):
    """Build a ConfigDict for the named experiment."""
    module = get_config_module(config_name)
    return module.get_config()
