# Cooperative reinforcement learning

Custom reinforcement learning agents, experience collectors, etc. in the ray cluster.

## Installation

Requires Python 3.11+. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

> JAX with CUDA 12 support is included as a dependency. Make sure the matching CUDA toolkit is available on your system.

### Godot (godot_rl_agents) environments

`HandlerGodotEnv` (used by the `dreamer_godot` config) drives a Godot game that embeds the [`godot_rl_agents`](https://github.com/edbeeching/godot_rl_agents) plugin. It only needs `godot_rl.core.godot_env.GodotEnv`, but `godot_rl`'s own dependency pins are incompatible here (it requires `gymnasium<=1.0.0` while this project uses `gymnasium` 1.3.0, and it pulls `torch`/`stable-baselines3`/`onnx`). Install it without its dependencies from the pinned commit:

```bash
scripts/install_godot_rl.sh
```

The script installs from a local clone of `godot_rl_agents` (default `~/Godot/godot_rl_agents`, override with `GODOT_RL_DIR`) — the upstream repo's Godot-side plugin is an SSH git submodule that a direct `git+https` install cannot fetch. The install lives in the virtualenv, so **re-run this script after any `uv sync`** that reinstalls or cleans it.

**Multi-env topology (K × N).** A single Bridge process hosts `N` in-process agents (`args_env.num_envs`, sent to the binary as `--n_envs`); `K` collectors (`config.num_collectors`) each launch their own headless process, for `K × N` parallel environments. Each collector gets a distinct TCP port (`args_env.port + i`) and env seed automatically. Use `--backend ray` for `K > 1`.

Build the Bridge binary first (see the Bridge project's README), then point `dreamer_godot` at it via `config.args_env.env_path` (default `/home/sia/Godot/Bridge/build/bridge`):

```bash
coop-rl-train --config dreamer_godot --backend ray --workdir ~/coop-rl_results
```

For single-process debugging set `args_env.env_path = None` to connect to a running editor instead: Python is the TCP **server**, so start the trainer first (it binds the port), then press **Play** in the Godot editor (`num_collectors` must be 1).

## Running training

Install the package, then use the `coop-rl-train` entry point with `--config` set to the config module name.

**DQN on Atari (thread backend):**

```bash
coop-rl-train --config dqn_atari --workdir ~/coop-rl_results
```

**With the Ray distributed backend:**

```bash
coop-rl-train --config dqn_atari --backend ray --workdir ~/coop-rl_results
```

Optional flags:
- `--orbax-checkpoint-dir PATH` — resume from an existing Orbax checkpoint
- `--debug-ray` — enable Ray's local debug mode
- `--debug-log` — enable debug-level logging
