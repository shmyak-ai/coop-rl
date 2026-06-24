"""Evaluation / watch CLI for a trained Dreamer Godot checkpoint.

Restores a checkpoint and runs the policy against the live Godot Bridge env in a
single process (no Ray, no trainer/buffer). Reports per-episode and mean/std
returns; ``--watch`` switches to a real-time windowed view you can watch.

Usage
-----
Evaluate over 10 episodes (uses the env_path/speedup from the config, i.e. the
exported binary at ``~/Godot/Bridge/build/bridge``)::

    python -m coop_rl.cli.eval_godot_bridge --config dreamer_godot \
        --orbax-checkpoint-dir ~/coop-rl_results/run-77mfivc4/chkpt_train_step_0252000 \
        --episodes 10

Watch the agent play in real time (a Godot window opens and renders)::

    python -m coop_rl.cli.eval_godot_bridge --config dreamer_godot \
        --orbax-checkpoint-dir ~/coop-rl_results/run-77mfivc4/chkpt_train_step_0252000 \
        --watch

``--watch`` forces ``show_window=True``, ``speedup=1`` and ``num_envs=1`` (one
clean window). Pass ``--num-envs 4`` to watch the full 4-instance vector the
Bridge runs by default. ``--env-path`` overrides the binary (give the path
*without* the ``.x86_64`` suffix — ``godot_rl`` appends it); pass ``--env-path ""``
to connect to a running Godot editor instead (press Play).

Rebuilding the Bridge binary
----------------------------
The env launches the exported binary ``~/Godot/Bridge/build/bridge.x86_64``. To
rebuild it after changing the Godot project (see ``~/Godot/Bridge/README.md``):

1. One-time: in the Godot 4.6 editor, *Editor -> Manage Export Templates ->
   Download and Install* (must match your editor version, e.g. 4.6.3).
2. GUI: *Project -> Export...*, select the ``Linux`` preset (path already set to
   ``build/bridge.x86_64``), uncheck "Export With Debug", click *Export Project...*.
3. Or headless from a terminal (``godot`` is usually not on PATH, so call your
   Godot 4.6 binary directly)::

       cd ~/Godot/Bridge
       ~/Godot/<godot-4.6-binary> --headless --export-release "Linux" build/bridge.x86_64
"""

import argparse
import os

from coop_rl.configs import list_available_configs


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate / watch a trained Dreamer policy.")
    parser.add_argument("--config", required=True, type=str, choices=list_available_configs())
    parser.add_argument(
        "--orbax-checkpoint-dir",
        required=True,
        type=str,
        help="Absolute path to the orbax checkpoint dir (e.g. .../chkpt_train_step_0252000).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of completed episodes to collect before reporting.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Real-time single-env windowed view (sets show_window=True, speedup=1, num_envs=1).",
    )
    parser.add_argument(
        "--env-path",
        type=str,
        default=None,
        help="Override the Godot binary path. Use a non-headless export to watch; "
        "pass an empty string to connect to the editor (press Play).",
    )
    parser.add_argument("--speedup", type=int, default=None, help="Override env speedup.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of agents.")
    parser.add_argument("--port", type=int, default=None, help="Override env TCP port.")
    parser.add_argument("--seed", type=int, default=0, help="RNG / env reset seed.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100000,
        help="Safety cap on env steps before reporting whatever has completed.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args(argv)


def run_eval(args: argparse.Namespace) -> None:
    """Restore the checkpoint, roll out the policy, and report returns."""
    import jax
    import numpy as np

    from coop_rl.runtime.training import load_runtime_config

    conf = load_runtime_config(args.config, args.orbax_checkpoint_dir)

    if args.watch:
        conf.args_env.show_window = True
        conf.args_env.speedup = 1
        conf.args_env.num_envs = 1
    if args.env_path is not None:
        conf.args_env.env_path = args.env_path or None
    if args.speedup is not None:
        conf.args_env.speedup = args.speedup
    if args.num_envs is not None:
        conf.args_env.num_envs = args.num_envs
    if args.port is not None:
        conf.args_env.port = args.port

    conf.args_state_recover.rng = jax.random.PRNGKey(args.seed)
    flax_state = conf.state_recover(**conf.args_state_recover)
    select_action = conf.args_collector.get_select_action_fn(flax_state)

    env = conf.env(**conf.args_env)
    num_envs = env.num_envs
    try:
        img, _ = env.reset(seed=args.seed)
        obs = {
            "image": img,
            "is_first": np.ones(num_envs, bool),
            "is_last": np.zeros(num_envs, bool),
            "is_terminal": np.zeros(num_envs, bool),
            "reward": np.zeros(num_envs, np.float32),
        }
        prev_done = np.zeros(num_envs, bool)
        ep_reward = np.zeros(num_envs, np.float32)
        returns: list[float] = []

        steps = 0
        while len(returns) < args.episodes and steps < args.max_steps:
            flax_state, action, _ = select_action(flax_state, obs)
            act = np.asarray(action["action"], dtype=np.int32)

            # NEXT_STEP autoreset (mirrors CollectorDreamerUniform.run_rollout):
            # the step finishing an episode returns the terminal obs (done=True);
            # the following step returns the fresh reset obs, so is_first lags done.
            next_img, reward, terminated, truncated, _ = env.step(act)
            done = np.logical_or(terminated, truncated)
            is_first = prev_done
            reward = np.where(is_first, 0.0, reward).astype(np.float32)
            obs = {
                "image": next_img,
                "is_first": is_first,
                "is_last": done,
                "is_terminal": done,
                "reward": reward,
            }
            prev_done = done

            ep_reward += reward
            for i in np.where(done)[0]:
                returns.append(float(ep_reward[i]))
                print(f"episode {len(returns):>4d}: return = {ep_reward[i]:.2f}")
                ep_reward[i] = 0.0
            steps += 1
    finally:
        env.close()

    if returns:
        arr = np.asarray(returns)
        print(
            f"\n{len(returns)} episodes | mean {arr.mean():.2f} | std {arr.std():.2f} "
            f"| min {arr.min():.2f} | max {arr.max():.2f}"
        )
    else:
        print(f"\nNo episode completed within {args.max_steps} steps.")


def main(argv: list[str] | None = None) -> None:
    """Run checkpoint evaluation from the command line."""
    from coop_rl.runtime.training import RUNTIME_ENV_THREAD, TF_LOG_SUPPRESS_ENV_VARS

    args = parse_args(argv)
    for key, value in TF_LOG_SUPPRESS_ENV_VARS.items():
        os.environ.setdefault(key, value)
    for key, value in RUNTIME_ENV_THREAD["env_vars"].items():
        os.environ.setdefault(key, value)
    run_eval(args)


if __name__ == "__main__":
    main()
