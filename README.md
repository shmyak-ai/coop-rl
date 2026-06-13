# Cooperative reinforcement learning

Custom reinforcement learning agents, experience collectors, etc. in the ray cluster.

## Installation

Requires Python 3.11+. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

> JAX with CUDA 12 support is included as a dependency. Make sure the matching CUDA toolkit is available on your system.

### Unity (ML-Agents) environments

`HandlerUnityEnv` depends on `mlagents-envs`, which ships protobuf stubs built for `protobuf<3.21`. This project pins `protobuf` 6.x (required by ray/tensorflow/tensorboard/orbax), so a freshly installed `mlagents-envs` fails to `import mlagents_envs`. Regenerate its stubs for the installed protobuf/grpcio:

```bash
scripts/regen_mlagents_protos.sh
```

Set `MLAGENTS_PROTO_DIR` if your ml-agents `.proto` sources are not at the script's default path. The regenerated stubs live in the virtualenv, so **re-run this script after any `uv sync`** that reinstalls `mlagents-envs`.

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
