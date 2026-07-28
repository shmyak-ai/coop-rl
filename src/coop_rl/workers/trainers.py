# Copyright 2025 The Coop RL Authors.
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

import contextlib
import itertools
import logging
import os
import threading
import time
from collections.abc import Generator
from queue import Empty, Full, Queue

import flax
import jax
import numpy as np
import orbax.checkpoint as ocp
import psutil

from coop_rl.workers.auxiliary import CommandExecutor, Controller, _TBWriter


class _RWLock:
    """Readers-writer lock: concurrent reads, exclusive writes."""

    def __init__(self):
        self._mutex = threading.Lock()  # protects _readers count
        self._write_lock = threading.Lock()  # held exclusively by a writer
        self._readers = 0

    @contextlib.contextmanager
    def read(self):
        with self._mutex:
            self._readers += 1
            if self._readers == 1:
                self._write_lock.acquire()
        try:
            yield
        finally:
            with self._mutex:
                self._readers -= 1
                if self._readers == 0:
                    self._write_lock.release()

    @contextlib.contextmanager
    def write(self):
        with self._write_lock:
            yield


class BufferKeeper:
    def __init__(self, buffer, args_buffer, num_samples_on_gpu_cache):
        self.buffer = buffer(**args_buffer)
        self._rw_lock = _RWLock()
        self._samples_on_gpu = Queue(maxsize=num_samples_on_gpu_cache)
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            raise RuntimeError("No GPU devices found. BufferKeeper requires at least one GPU.")
        self.gpu_device = gpu_devices[0]
        self.logger = logging.getLogger(__name__)
        self._sample_steps = args_buffer.sample_batch_size * args_buffer.sample_sequence_length
        self._steps_added = 0
        self._steps_sampled = 0
        self.is_done = False

    def add_traj_seq(self, data):
        # a trick to start a ray thread, is it necessary?
        if data == 1:
            return
        _collector_seed, batch = data
        with self._rw_lock.write():
            self.buffer.add(batch)
        leaves = jax.tree_util.tree_leaves(batch)
        n, traj_len = leaves[0].shape[0], leaves[0].shape[1]
        self._steps_added += n * traj_len
        return True

    def buffer_sampling(self):
        while True:
            if self.is_done:
                self.logger.info("Done signal received; finishing buffer sampling.")
                return

            with self._rw_lock.read():
                can_sample = self.buffer.can_sample()
            if can_sample:
                self.logger.info("Sampler: start buffer sampling.")
                break
            else:
                time.sleep(1)

        while True:
            if self.is_done:
                self.logger.info("Done signal received; finishing buffer sampling.")
                return

            with self._rw_lock.read():
                sample = self.buffer.sample()
            self._steps_sampled += self._sample_steps
            sample_on_gpu = jax.device_put(sample, device=self.gpu_device)
            while True:
                try:
                    self._samples_on_gpu.put(sample_on_gpu, timeout=0.1)
                    break
                except Full:
                    if self.is_done:
                        return

    def get_samples(self, train_iterations_num: int = 10) -> Generator:
        while True:
            batch = []
            while len(batch) < train_iterations_num:
                try:
                    batch.append(self._samples_on_gpu.get(timeout=0.05))
                except Empty:
                    if self.is_done:
                        self.logger.info("Done signal received; finishing sample generator.")
                        return
            yield batch


class Trainer(BufferKeeper):
    def __init__(
        self,
        *,
        controller,
        trainer_seed,
        log_level,
        workdir,
        steps,
        training_iterations_per_step,
        summary_writing_period,
        save_period,
        synchronization_period,
        state_recover,
        args_state_recover,
        get_update_step,
        args_get_update_step,
        get_update_epoch,
        args_get_update_epoch,
        buffer,
        args_buffer,
        num_samples_on_gpu_cache,
    ):
        super().__init__(buffer, args_buffer, num_samples_on_gpu_cache)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.workdir = workdir

        self.steps = steps
        self.training_iterations_per_step = training_iterations_per_step
        self.batch_size = args_buffer.sample_batch_size
        self.batch_length = args_buffer.sample_sequence_length
        self.buffer_size = args_buffer.max_size

        self.synchronization_period = synchronization_period
        self.summary_writing_period = summary_writing_period
        self.save_period = save_period
        self.orbax_checkpointer = ocp.StandardCheckpointer()

        self._rng = jax.random.PRNGKey(trainer_seed)
        self._rng, rng = jax.random.split(self._rng)
        args_state_recover.rng = rng
        self.flax_state = state_recover(**args_state_recover)
        args_get_update_step.apply_fn = self.flax_state.apply_fn
        update_step_fn = get_update_step(**args_get_update_step)
        args_get_update_epoch.update_step_fn = update_step_fn
        if hasattr(args_get_update_epoch, "buffer_lock"):
            args_get_update_epoch.buffer_lock = self._rw_lock
        if hasattr(args_get_update_epoch, "buffer"):
            args_get_update_epoch.buffer = self.buffer
        self.update_epoch_fn = get_update_epoch(**args_get_update_epoch)

        self.controller = controller
        self.command_executor = CommandExecutor(max_workers=1)
        self.futures = self.command_executor.submit(
            self.controller,
            "set_parameters",
            jax.device_get(self.flax_state.params),
        )
        self.is_done = False
        self._closed = False
        self._proc = psutil.Process()
        psutil.cpu_percent()  # prime baseline; first call always returns 0.0
        self._writer = _TBWriter(os.path.join(workdir, "tb"))
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(self.flax_state.params))
        flat_params = flax.traverse_util.flatten_dict(self.flax_state.params, sep="/")
        shape_str = "\n".join(f"  {k}: {v.shape}" for k, v in flat_params.items())
        self.logger.info(
            "Trainer initialized (steps=%d, buffer_size=%d).", self.steps, self.buffer_size
        )
        self.logger.info("Network parameter shapes:\n%s", shape_str)
        self.logger.info("Total trainable parameters: %d.", total_params)

    def training(self):
        try:
            self._training()
        finally:
            self.close()

    def _training(self):
        self.command_executor.resolve(self.futures)
        samples_generator = self.get_samples(self.training_iterations_per_step)
        step_start = time.monotonic()
        for step in itertools.count(start=1, step=1):
            try:
                samples = next(samples_generator)
            except StopIteration:
                self.logger.info("Done signal received; finishing training.")
                return
            self.flax_state, info = self.update_epoch_fn(self.flax_state, samples)

            if step == self.steps:
                self.is_done = True
                self.command_executor.call(self.controller, "set_done")
                self.logger.info(f"Final training step {step} reached; finishing.")
                break

            if step % self.summary_writing_period == 0:
                elapsed = time.monotonic() - step_start
                queue_fill = self._samples_on_gpu.qsize() / max(self._samples_on_gpu.maxsize, 1)
                rss_main = self._proc.memory_info().rss / 2**30
                cpu_pct = psutil.cpu_percent()
                _gpu_stats = jax.devices("gpu")[0].memory_stats()
                gpu_peak_gib = _gpu_stats.get("peak_bytes_in_use", 0) / 2**30
                self.logger.info(f"Training step: {self.flax_state.step}.")
                self.logger.info(f"Steps added to buffer: {self._steps_added}.")
                self.logger.info(f"Steps sampled from buffer: {self._steps_sampled}.")
                self.logger.info(
                    f"Last {self.summary_writing_period} steps: {elapsed:.1f}s  "
                    f"GPU queue fill: {queue_fill:.2f}"
                )
                self.logger.info(
                    f"CPU: {cpu_pct:.1f}%  RSS: {rss_main:.2f} GiB  "
                    f"GPU peak: {gpu_peak_gib:.2f} GiB."
                )
                scalars = {
                    "trainer/steps_per_second": self.summary_writing_period / elapsed,
                    "perf/gpu_queue_fill": queue_fill,
                    "system/cpu_percent": cpu_pct,
                    "system/cpu_rss_gib": rss_main,
                    "system/gpu_peak_bytes_in_use_gib": gpu_peak_gib,
                }
                # All scalar diagnostics from the update step (skips per-sample
                # arrays such as the PER priorities), like TrainerSequential.
                for name, value in info.items():
                    if np.ndim(value) == 0:
                        scalars[f"trainer/{name}"] = float(value)
                self._writer.write_scalars(int(self.flax_state.step), scalars)
                self._writer.flush()
                step_start = time.monotonic()

            if step % self.synchronization_period == 0:
                self.command_executor.resolve(self.futures)
                self.futures = self.command_executor.submit(
                    self.controller,
                    "set_parameters",
                    jax.device_get(self.flax_state.params),
                )

            if step % self.save_period == 0:
                orbax_checkpoint_path = os.path.join(
                    self.workdir, f"chkpt_train_step_{self.flax_state.step:07}"
                )
                self.orbax_checkpointer.save(orbax_checkpoint_path, self.flax_state)
                self.logger.info(f"Orbax checkpoint is in: {orbax_checkpoint_path}")

    def close(self) -> None:
        """Release local helper resources after training threads have stopped."""
        if self._closed:
            return
        self._closed = True
        self.command_executor.shutdown()
        self._writer.close()
        self.logger.info("Trainer closed.")


class TrainerSequential:
    """Synchronous single-thread trainer: collect a rollout, then train, repeat.

    Restores the classic DQN coupling between environment frames and gradient
    updates: after every rollout of n transitions, performs n * replay_ratio
    updates (replay_ratio=1.0 is the BY571/IQN-and-Extensions regime, 0.25 is
    classic DQN's train-every-4-frames). n counts transitions summed across all
    vectorized envs (steps_per_rollout * num_envs), so num_envs changes the
    rollout granularity but not the updates-per-frame coupling: e.g. 32 envs at
    replay_ratio 1/64 accrue one update per two rollouts. Training stops once
    env_frames total transitions have been collected. No Controller relay, sampler threads, or
    GPU-sample queue: the collector runs in the same thread and always acts
    with the latest parameters.
    """

    def __init__(
        self,
        *,
        trainer_seed,
        log_level,
        workdir,
        env_frames,
        replay_ratio,
        summary_writing_period,
        save_period,
        state_recover,
        args_state_recover,
        get_update_step,
        args_get_update_step,
        get_update_epoch,
        args_get_update_epoch,
        buffer,
        args_buffer,
        collector,
        args_collector,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.workdir = workdir

        self.env_frames = env_frames
        self.replay_ratio = replay_ratio
        self.summary_writing_period = summary_writing_period
        self.save_period = save_period
        self.orbax_checkpointer = ocp.StandardCheckpointer()

        self.buffer = buffer(**args_buffer)
        self.sample_batch_size = args_buffer.sample_batch_size
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            raise RuntimeError("No GPU devices found. TrainerSequential requires at least one GPU.")
        self.gpu_device = gpu_devices[0]

        self._rng = jax.random.PRNGKey(trainer_seed)
        self._rng, rng = jax.random.split(self._rng)
        args_state_recover.rng = rng
        self.flax_state = state_recover(**args_state_recover)
        args_get_update_step.apply_fn = self.flax_state.apply_fn
        update_step_fn = get_update_step(**args_get_update_step)
        args_get_update_epoch.update_step_fn = update_step_fn
        if hasattr(args_get_update_epoch, "buffer_lock"):
            # Single thread, so the lock is never contended; it only satisfies the
            # shared update-epoch signature used for PER priority write-backs.
            args_get_update_epoch.buffer_lock = _RWLock()
        if hasattr(args_get_update_epoch, "buffer"):
            args_get_update_epoch.buffer = self.buffer
        self.update_epoch_fn = get_update_epoch(**args_get_update_epoch)

        # The collector submits a constructor-time get_parameters request, so it
        # needs a real controller object; this local one is never consulted again
        # because only run_rollout() is used here, not _collecting().
        args_collector.controller = Controller(log_level=log_level)
        args_collector.trainer = None
        self.collector = collector(**args_collector)

        self._closed = False
        self._writer = _TBWriter(os.path.join(workdir, "tb"))
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(self.flax_state.params))
        self.logger.info(
            "TrainerSequential initialized (env_frames=%d, replay_ratio=%s, buffer_size=%d).",
            self.env_frames,
            self.replay_ratio,
            args_buffer.max_size,
        )
        self.logger.info("Total trainable parameters: %d.", total_params)

    def training(self):
        try:
            self._training()
        finally:
            self.close()

    def _training(self):
        if self.collector.obs is None:
            self.collector.warmup()
        frames = 0
        episodes_done = 0
        update_debt = 0.0
        rollout_frames = self.collector.steps_per_rollout * self.collector.num_envs
        last_summary_updates = 0
        last_save_updates = 0
        window_start = time.monotonic()
        info = None
        while frames < self.env_frames:
            # The collector must act with the latest params only.
            self.collector.set_online_params(self.flax_state.params)
            trajectories = self.collector.run_rollout()
            self.buffer.add(trajectories)
            frames += rollout_frames
            episodes_done += int((trajectories.terminated | trajectories.truncated).sum())

            if not self.buffer.can_sample():
                continue  # warm-up frames earn no update debt, as in BY571

            update_debt += rollout_frames * self.replay_ratio
            while update_debt >= 1.0:
                sample = jax.device_put(self.buffer.sample(), device=self.gpu_device)
                self.flax_state, info = self.update_epoch_fn(self.flax_state, [sample])
                update_debt -= 1.0

            updates = int(self.flax_state.step)
            if info is not None and updates - last_summary_updates >= self.summary_writing_period:
                elapsed = time.monotonic() - window_start
                scalars = {
                    "trainer/env_frames": frames,
                    "trainer/frames_sampled": updates * self.sample_batch_size,
                    "trainer/updates": updates,
                    "trainer/updates_per_second": (updates - last_summary_updates) / elapsed,
                    "collector/episodes_done": episodes_done,
                }
                # All scalar diagnostics from the update step (skips per-sample
                # arrays such as the PER priorities).
                for name, value in info.items():
                    if np.ndim(value) == 0:
                        scalars[f"trainer/{name}"] = float(value)
                returns = self.collector.completed_returns
                if returns:
                    scalars["collector/mean_return"] = sum(returns) / len(returns)
                # Last epsilon computed by the collector's action-selection closure
                # (set only when an epsilon_scheduler_fn is configured).
                epsilon = getattr(self.collector.select_action, "epsilon", None)
                if epsilon is not None:
                    scalars["collector/epsilon"] = float(epsilon)
                self._writer.write_scalars(updates, scalars)
                self._writer.flush()
                self.logger.info(
                    "  ".join(
                        f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v:g}"
                        for k, v in scalars.items()
                    )
                )
                last_summary_updates = updates
                window_start = time.monotonic()

            if updates - last_save_updates >= self.save_period:
                orbax_checkpoint_path = os.path.join(
                    self.workdir, f"chkpt_train_step_{self.flax_state.step:07}"
                )
                self.orbax_checkpointer.save(orbax_checkpoint_path, self.flax_state)
                self.logger.info(f"Orbax checkpoint is in: {orbax_checkpoint_path}")
                last_save_updates = updates

        orbax_checkpoint_path = os.path.join(
            self.workdir, f"chkpt_train_step_{self.flax_state.step:07}"
        )
        self.orbax_checkpointer.save(orbax_checkpoint_path, self.flax_state)
        self.logger.info(
            "Frame budget reached (%d frames, %d updates); final checkpoint in: %s",
            frames,
            int(self.flax_state.step),
            orbax_checkpoint_path,
        )

    def close(self) -> None:
        """Release the collector's env/executor and the writer."""
        if self._closed:
            return
        self._closed = True
        # Checkpoint saves commit in a background thread; wait for them here,
        # otherwise interpreter shutdown kills the commit and leaves the final
        # checkpoint as an uncommitted *.orbax-checkpoint-tmp directory.
        self.orbax_checkpointer.close()
        self.collector.close()
        self._writer.close()
        self.logger.info("TrainerSequential closed.")
