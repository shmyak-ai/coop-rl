# Cooperative reinforcement learning

Custom reinforcement learning agents, experience collectors, etc. in the ray cluster.

## Installation

Requires Python 3.11+. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

> JAX with CUDA 12 support is included as a dependency. Make sure the matching CUDA toolkit is available on your system.

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
