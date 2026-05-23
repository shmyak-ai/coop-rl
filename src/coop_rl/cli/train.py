"""Training CLI entry point."""

import argparse
import os
from pathlib import Path

from coop_rl.configs import list_available_configs
from coop_rl.runtime.training import RUNTIME_ENV_THREAD, TF_LOG_SUPPRESS_ENV_VARS, run_training


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for distributed training."""
    parser = argparse.ArgumentParser(description="Cooperative reinforcement learning.")
    parser.add_argument("--config", required=True, type=str, choices=list_available_configs())
    parser.add_argument(
        "--backend",
        type=str,
        default="thread",
        choices=["ray", "thread"],
        help="Execution backend: distributed Ray actors or local Python threads.",
    )
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug environment.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run distributed training from the command line."""
    args = parse_args(argv)
    for key, value in TF_LOG_SUPPRESS_ENV_VARS.items():
        os.environ.setdefault(key, value)
    if args.backend == "thread":
        for key, value in RUNTIME_ENV_THREAD["env_vars"].items():
            os.environ.setdefault(key, value)
    if args.debug:
        from coop_rl.runtime.training import RUNTIME_ENV_DEBUG

        for key, value in RUNTIME_ENV_DEBUG["env_vars"].items():
            os.environ.setdefault(key, value)
    run_training(args)


if __name__ == "__main__":
    main()
