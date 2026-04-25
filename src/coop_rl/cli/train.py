"""Training CLI entry point."""

import argparse
from pathlib import Path

from coop_rl.configs import list_available_configs
from coop_rl.runtime.training import run_training


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for distributed training."""
    parser = argparse.ArgumentParser(description="Cooperative reinforcement learning.")
    parser.add_argument(
        "--debug-ray",
        action="store_true",
        help="Ray debug environment activation.",
    )
    parser.add_argument("--debug-log", action="store_true", help="Enable debug logs.")
    parser.add_argument(
        "--backend",
        type=str,
        default="thread",
        choices=["ray", "thread"],
        help="Execution backend: distributed Ray actors or local Python threads.",
    )
    parser.add_argument("--config", required=True, type=str, choices=list_available_configs())
    parser.add_argument(
        "--workdir",
        type=str,
        default=str(Path.home() / "coop-rl_results"),
        help="Path to the tensorboard logs, checkpoints, etc.",
    )
    parser.add_argument(
        "--orbax-checkpoint-dir",
        type=str,
        help="The absolute path to the orbax checkpoint dir",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run distributed training from the command line."""
    run_training(parse_args(argv))
