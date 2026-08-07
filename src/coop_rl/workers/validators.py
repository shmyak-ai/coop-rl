# Copyright 2026 The Coop RL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from collections import deque

import jax
import numpy as np

from coop_rl.workers.auxiliary import CommandExecutor, _TBWriter


class ValidatorDQNWindowedUniform:
    """Async worker that measures algorithm performance and reports it, never
    training or feeding the replay buffer. Uses the same sliding
    context_length window as CollectorDQNWindowedUniform to act with a
    causal-transformer network. Skippable: a training run simply omits
    config.validator/config.args_validator.

    Two independent cadence modes, selected by `cadence`:
      - "interval": wait interval_seconds (wall clock) between validation
        runs, acting with the controller's latest published params.
      - "checkpoint": wait for the trainer to publish a new checkpoint (via
        Controller.set_checkpoint) and validate with that checkpoint's
        restored params. Polls the controller every poll_period seconds.
    """

    def __init__(
        self,
        *,
        controller,
        validator_seed,
        log_level,
        state_recover,
        args_state_recover,
        env,
        args_env,
        time_step_dtypes,
        steps_per_validation,
        get_select_action_fn,
        args_get_select_action_fn,
        context_length,
        cadence,
        interval_seconds,
        poll_period,
        workdir: str | None = None,
    ):
        if cadence not in ("interval", "checkpoint"):
            raise ValueError(f"cadence must be 'interval' or 'checkpoint', got {cadence!r}")

        self.logger = logging.getLogger(f"{__name__}.seed{validator_seed}")
        self.logger.setLevel(log_level)

        self.controller = controller
        self.command_executor = CommandExecutor(max_workers=1)

        self.env = env(**args_env)
        self.num_envs = self.env.num_envs

        self.dtypes = time_step_dtypes()
        self.steps_per_validation = steps_per_validation
        self.context_length = context_length

        self.cadence = cadence
        self.interval_seconds = interval_seconds
        self.poll_period = poll_period
        self._last_checkpoint_step = None

        self.validator_seed = validator_seed
        self._rng = jax.random.PRNGKey(validator_seed)
        self._rng, rng = jax.random.split(self._rng)
        self.state_recover = state_recover
        self.args_state_recover = args_state_recover
        self.args_state_recover.rng = rng
        flax_state = self.state_recover(**self.args_state_recover)

        # Validation always acts with the newest params/checkpoint -- no
        # staleness tolerance needed since there's no freezing concern like
        # the training-policy Collectors have.
        self.online_params = flax_state.params

        args_get_select_action_fn.apply_fn = flax_state.apply_fn
        self.select_action = get_select_action_fn(**args_get_select_action_fn)
        self.obs = None
        self._win_obs = None
        self._win_action = None
        self._win_reward = None
        self._win_terminated = None
        self._win_truncated = None
        self.episode_reward_now = np.zeros(self.num_envs)
        self.completed_returns: deque[float] = deque(maxlen=100)
        self.episode_steps_now = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_switches_now = np.zeros(self.num_envs, dtype=np.int64)
        self._prev_action = np.zeros(self.num_envs, dtype=self.dtypes.action)
        self.completed_lengths: deque[float] = deque(maxlen=100)
        self.completed_switch_rates: deque[float] = deque(maxlen=100)
        self.completed_terminated: deque[float] = deque(maxlen=100)
        self.completed_metrics: dict[str, deque[float]] = {}
        self._env_steps = 0
        self._writer: _TBWriter | None = (
            _TBWriter(os.path.join(workdir, "tb")) if workdir is not None else None
        )
        self._closed = False
        self.logger.info(
            "ValidatorDQNWindowedUniform initialized (seed=%d, num_envs=%d, "
            "context_length=%d, cadence=%s).",
            validator_seed,
            self.num_envs,
            self.context_length,
            self.cadence,
        )

    def _reset_obs(self) -> None:
        self.obs, _ = self.env.reset()
        win_shape = (self.num_envs, self.context_length, *self.obs.shape[1:])
        self._win_obs = np.zeros(win_shape, dtype=self.dtypes.obs)
        self._win_action = np.zeros((self.num_envs, self.context_length), dtype=self.dtypes.action)
        self._win_reward = np.zeros((self.num_envs, self.context_length), dtype=self.dtypes.reward)
        self._win_terminated = np.zeros(
            (self.num_envs, self.context_length), dtype=self.dtypes.terminated
        )
        # See CollectorDQNWindowedUniform._reset_obs: every slot but the newest
        # is seeded as "truncated" so build_causal_boundary_mask treats each
        # not-yet-real placeholder as its own isolated singleton episode.
        self._win_truncated = np.ones(
            (self.num_envs, self.context_length), dtype=self.dtypes.truncated
        )
        self._win_obs[:, -1] = self.obs
        self._win_truncated[:, -1] = 0

    def _push_window(self, next_obs: np.ndarray) -> None:
        for arr in (
            self._win_obs,
            self._win_action,
            self._win_reward,
            self._win_terminated,
            self._win_truncated,
        ):
            arr[:, :-1] = arr[:, 1:]
        self._win_obs[:, -1] = next_obs
        self._win_action[:, -1] = 0
        self._win_reward[:, -1] = 0
        self._win_terminated[:, -1] = 0
        self._win_truncated[:, -1] = 0

    def _track_step(self, actions, rewards, infos):
        """Advance the per-episode counters for one env step; returns the env's
        per-episode metrics dict from infos (or None) for _finish_episode_stats."""
        self.episode_reward_now += rewards
        self.episode_steps_now += 1
        self.episode_switches_now += (actions != self._prev_action) & (self.episode_steps_now > 1)
        self._prev_action = actions
        return infos.get("metrics") if isinstance(infos, dict) else None

    def _finish_episode_stats(self, i, terminated_i, metrics) -> None:
        """Close out env i's episode statistics and reset its counters."""
        steps = max(int(self.episode_steps_now[i]), 1)
        self.completed_returns.append(float(self.episode_reward_now[i]))
        self.completed_lengths.append(float(steps))
        self.completed_switch_rates.append(1000.0 * float(self.episode_switches_now[i]) / steps)
        self.completed_terminated.append(float(terminated_i))
        if metrics is not None:
            for name, values in metrics.items():
                self.completed_metrics.setdefault(name, deque(maxlen=100)).append(
                    float(np.asarray(values)[i])
                )
        self.episode_reward_now[i] = 0.0
        self.episode_steps_now[i] = 0
        self.episode_switches_now[i] = 0

    def episode_scalars(self) -> dict[str, float]:
        """Per-episode metrics since the last call, cleared on read ({} if none)."""
        if not self.completed_returns:
            return {}
        lengths = np.asarray(self.completed_lengths)
        scalars = {
            "validator/mean_return": float(np.mean(self.completed_returns)),
            "validator/median_return": float(np.median(self.completed_returns)),
            "validator/reward_per_step": float(np.sum(self.completed_returns) / np.sum(lengths)),
            "validator/episode_length": float(np.mean(lengths)),
            "validator/action_switch_rate_per_1k": float(np.mean(self.completed_switch_rates)),
            "validator/terminated_fraction": float(np.mean(self.completed_terminated)),
        }
        for name, values in self.completed_metrics.items():
            if values:
                scalars[f"validator/mean_{name}"] = float(np.mean(values))
                scalars[f"validator/median_{name}"] = float(np.median(values))
        self.completed_returns.clear()
        self.completed_lengths.clear()
        self.completed_switch_rates.clear()
        self.completed_terminated.clear()
        for values in self.completed_metrics.values():
            values.clear()
        return scalars

    def warmup(self) -> None:
        """Trigger JIT compilation of select_action in the calling thread."""
        self._reset_obs()
        self.select_action(
            self._rng,
            self.online_params,
            self._win_obs,
            self._win_action,
            self._win_reward,
            self._win_terminated,
            self._win_truncated,
        )

    def run_validation(self) -> None:
        """Step the env steps_per_validation times, feeding the episode-stat
        accumulators consumed by episode_scalars(). No trajectory is built or
        stored -- validation never touches the replay buffer."""
        for _ in range(self.steps_per_validation):
            self._rng, action_jnp = self.select_action(
                self._rng,
                self.online_params,
                self._win_obs,
                self._win_action,
                self._win_reward,
                self._win_terminated,
                self._win_truncated,
            )
            actions = np.asarray(action_jnp, dtype=self.dtypes.action)  # (num_envs,)
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

            self._win_action[:, -1] = actions
            self._win_reward[:, -1] = rewards.astype(self.dtypes.reward)
            self._win_terminated[:, -1] = terminated.astype(self.dtypes.terminated)
            self._win_truncated[:, -1] = truncated.astype(self.dtypes.truncated)

            metrics = self._track_step(actions, rewards, infos)
            done = np.logical_or(terminated, truncated)
            for i in np.where(done)[0]:
                self._finish_episode_stats(i, terminated[i], metrics)

            # AutoresetMode.DISABLED: env.step() returns the terminal obs but
            # never resets sub-environments internally. Reset done envs here so
            # self.obs always holds a valid initial observation for the next step.
            if done.any():
                reset_obs, _ = self.env.reset(options={"reset_mask": done})
                next_obs = next_obs.copy()
                next_obs[done] = reset_obs[done]

            self._push_window(next_obs)
            self.obs = next_obs

    def _wait_for_new_checkpoint(self) -> tuple[int, str] | tuple[None, None]:
        """Poll the controller until a not-yet-validated checkpoint appears or
        training finishes. Returns (step, path), or (None, None) if done."""
        while True:
            if self.command_executor.call(self.controller, "is_done"):
                return None, None
            step, path = self.command_executor.call(self.controller, "get_checkpoint")
            if path is not None and step != self._last_checkpoint_step:
                return int(step), path
            time.sleep(self.poll_period)

    def _sleep_or_done(self, seconds: float) -> bool:
        """Sleep in small chunks, checking is_done so shutdown stays responsive.
        Returns True if a done signal arrived during the sleep."""
        remaining = seconds
        chunk = min(1.0, seconds) if seconds > 0 else 0.0
        while remaining > 0:
            if self.command_executor.call(self.controller, "is_done"):
                return True
            time.sleep(min(chunk, remaining))
            remaining -= chunk
        return self.command_executor.call(self.controller, "is_done")

    def validating(self):
        try:
            self._validating()
        finally:
            self.close()

    def _validating(self):
        if self.obs is None:
            self._reset_obs()
        while True:
            if self.cadence == "checkpoint":
                step, path = self._wait_for_new_checkpoint()
                if path is None or step is None:
                    self.logger.info("Done signal received; finishing.")
                    return
                self._last_checkpoint_step = step
                self._rng, rng = jax.random.split(self._rng)
                self.args_state_recover.rng = rng
                self.args_state_recover.checkpointdir = path
                flax_state = self.state_recover(**self.args_state_recover)
                self.online_params = flax_state.params
                eval_step = step
            else:
                if self._sleep_or_done(self.interval_seconds):
                    self.logger.info("Done signal received; finishing.")
                    return
                params = self.command_executor.call(self.controller, "get_latest_parameters")
                if params is not None:
                    self.online_params = params
                eval_step = self._env_steps

            self.run_validation()
            self._env_steps += self.steps_per_validation * self.num_envs

            scalars = self.episode_scalars()
            self.logger.info(
                "Validation at step %d: %s",
                eval_step,
                {k: round(v, 3) for k, v in scalars.items()},
            )
            if self._writer is not None and scalars:
                self._writer.write_scalars(eval_step, scalars)
                self._writer.flush()

    def close(self) -> None:
        """Release local helper resources after validation stops."""
        if self._closed:
            return
        self._closed = True
        self.command_executor.shutdown()
        self.env.close()
        if self._writer is not None:
            self._writer.close()
        self.logger.info("ValidatorDQNWindowedUniform closed (seed=%d).", self.validator_seed)
