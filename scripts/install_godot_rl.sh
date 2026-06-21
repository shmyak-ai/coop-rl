#!/usr/bin/env bash
# Install godot_rl (the Python side of the godot_rl_agents plugin) for this project.
#
# HandlerGodotEnv only needs godot_rl.core.godot_env.GodotEnv, which imports just
# numpy + gymnasium (already in this env). godot_rl's own pins are incompatible
# here (it requires gymnasium<=1.0.0 while coop-rl uses gymnasium 1.3.0 for
# AutoresetMode, and it pulls torch/stable-baselines3/onnx), so install it with
# --no-deps. The install lives in the venv and is lost on `uv sync` that
# reinstalls/cleans it — re-run this script afterwards.
#
# We install from a LOCAL clone rather than the git URL on purpose: the upstream
# repo declares its Godot-side GDScript plugin as an SSH git submodule
# (git@github.com:.../godot_rl_agents_plugin), which uv tries to fetch and which
# fails without SSH credentials. The local clone is already checked out at the
# target commit and the submodule is irrelevant to the Python package.
#
# Usage: scripts/install_godot_rl.sh
#   GODOT_RL_DIR overrides the clone path (default below).
#   GODOT_RL_REF is the commit this was written against (informational check).
set -euo pipefail

GODOT_RL_DIR="${GODOT_RL_DIR:-$HOME/Godot/godot_rl_agents}"
GODOT_RL_REF="${GODOT_RL_REF:-207b6f476f5846f33d08b92c7a350147e7b78bf5}"

if [ ! -d "$GODOT_RL_DIR/godot_rl" ]; then
    echo "godot_rl clone not found at $GODOT_RL_DIR" >&2
    echo "clone https://github.com/edbeeching/godot_rl_agents there, or set GODOT_RL_DIR" >&2
    exit 1
fi

HEAD_REF="$(git -C "$GODOT_RL_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "clone   : $GODOT_RL_DIR"
echo "HEAD    : $HEAD_REF"
echo "pinned  : $GODOT_RL_REF"
if [ "$HEAD_REF" != "$GODOT_RL_REF" ]; then
    echo "note: clone HEAD differs from the pinned commit; installing the clone as-is." >&2
fi

uv pip install --no-deps "$GODOT_RL_DIR"

uv run python -c "from godot_rl.core.godot_env import GodotEnv; print('godot_rl import OK')"
