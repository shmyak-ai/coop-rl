"""Experiment configuration discovery and loading."""

from coop_rl.configs.registry import get_config, get_config_module, list_available_configs

__all__ = ["get_config", "get_config_module", "list_available_configs"]
