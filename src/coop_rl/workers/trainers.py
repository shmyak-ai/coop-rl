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

import collections
import itertools
import logging
import os
import threading
import time
from collections.abc import Generator
from queue import Empty, Queue

import jax
import numpy as np
import orbax.checkpoint as ocp

from coop_rl.workers.auxiliary import CommandExecutor


class BufferKeeper:
    def __init__(
        self, buffer, args_buffer, num_samples_on_gpu_cache, num_samples_to_gpu
    ):
        self.buffer = buffer(**args_buffer)
        self.add_batch_size = args_buffer.add_batch_size
        # Flat deque of individual trajectories; bounded to 8 rounds of all collectors.
        self.traj_queue = collections.deque(maxlen=self.add_batch_size * 8)
        self.buffer_lock = threading.Lock()
        self.store_lock = threading.Lock()
        self.device_lock = threading.Lock()
        self._samples_on_gpu = Queue(maxsize=num_samples_on_gpu_cache)
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            raise RuntimeError("No GPU devices found. BufferKeeper requires at least one GPU.")
        self.gpu_device = gpu_devices[0]
        self.num_samples_to_gpu = num_samples_to_gpu
        self.logger = logging.getLogger(__name__)
        self.is_done = False

    def add_traj_seq(self, data):
        # a trick to start a ray thread, is it necessary?
        if data == 1:
            return
        _collector_seed, _data = data
        with self.store_lock:
            self.traj_queue.append(_data)
        return True

    def buffer_updating(self):
        while True:
            if self.is_done:
                self.logger.info("Done signal received; finishing buffer updating.")
                break
            time.sleep(0.015)  # 15ms cadence; consume any ready chunk
            with self.store_lock:
                if len(self.traj_queue) < self.add_batch_size:
                    continue
                trajectories = [self.traj_queue.popleft() for _ in range(self.add_batch_size)]
            batched = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *trajectories)
            with self.buffer_lock:
                self.buffer.add(batched)

    def buffer_sampling(self):
        while True:
            if self.is_done:
                self.logger.info("Done signal received; finishing buffer sampling.")
                return

            with self.buffer_lock:
                can_sample = self.buffer.can_sample()
            if can_sample:
                break
            else:
                time.sleep(1)

        while True:
            samples = []
            for _ in range(self.num_samples_to_gpu):
                if self.is_done:
                    self.logger.info("Done signal received; finishing buffer sampling.")
                    return

                with self.buffer_lock:
                    sample = self.buffer.sample()
                samples.append(sample)
            with self.device_lock:
                samples_on_gpu = jax.device_put(samples, device=self.gpu_device)
            for sample_on_gpu in samples_on_gpu:
                # Queue.put blocks when full
                self._samples_on_gpu.put(sample_on_gpu)

    def get_samples(self, batch_size: int = 10) -> Generator:
        while True:
            batch = []
            while len(batch) < batch_size:
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
        num_samples_to_gpu,
    ):
        super().__init__(
            buffer, args_buffer, num_samples_on_gpu_cache, num_samples_to_gpu
        )

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
        self.logger.info("Trainer initialized (steps=%d, buffer_size=%d).", self.steps, self.buffer_size)

    def training(self):
        try:
            self._training()
        finally:
            self.close()

    def _training(self):
        self.command_executor.resolve(self.futures)
        samples_generator = self.get_samples(self.training_iterations_per_step)
        transitions_processed = 0
        step_start = time.monotonic()
        for step in itertools.count(start=1, step=1):
            try:
                samples = next(samples_generator)
            except StopIteration:
                self.logger.info("Done signal received; finishing training.")
                return
            self.flax_state, info = self.update_epoch_fn(self.flax_state, samples)
            transitions_processed += (
                self.batch_size * self.batch_length * self.training_iterations_per_step
            )

            if step == self.steps:
                self.is_done = True
                self.command_executor.call(self.controller, "set_done")
                self.logger.info(f"Final training step {step} reached; finishing.")
                break

            if step % self.summary_writing_period == 0:
                elapsed = time.monotonic() - step_start
                queue_fill = self._samples_on_gpu.qsize() / max(self._samples_on_gpu.maxsize, 1)
                self.logger.info(f"Training step: {self.flax_state.step}.")
                self.logger.info(f"Transitions sampled from restart: {transitions_processed}.")
                self.logger.info(
                    f"Last {self.summary_writing_period} steps: {elapsed:.1f}s  "
                    f"GPU queue fill: {queue_fill:.2f}"
                )
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
        self.logger.info("Trainer closed.")
