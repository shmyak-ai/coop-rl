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
from queue import Queue

import jax
import numpy as np
import orbax.checkpoint as ocp
import ray
from flashbax.buffers.sum_tree import get_tree_index


class BufferKeeper:
    def __init__(self, buffer, args_buffer, num_samples_on_gpu_cache, num_samples_to_gpu, num_semaphor):
        self.buffer = buffer(**args_buffer)
        self.traj_store = {}
        self.add_batch_size = args_buffer.add_batch_size
        self.buffer_lock = threading.Lock()
        self.store_lock = threading.Lock()
        self.device_lock = threading.Lock()
        self.semaphore = threading.Semaphore(num_semaphor)
        self._samples_on_gpu = Queue(maxsize=num_samples_on_gpu_cache)
        self.gpu_device = jax.devices("gpu")[0]
        self.num_samples_to_gpu = num_samples_to_gpu

    def add_traj_seq(self, data):
        # a trick to start a ray thread, is it necessary?
        if data == 1:
            return
        # save data from a collector
        collector_seed, _data = data
        with self.store_lock:
            # for collectors synchronization, put only if there is no data already
            if self.traj_store.get(collector_seed) is not None:
                return False
            self.traj_store[collector_seed] = _data
            return True

    def buffer_updating(self):
        while True:
            if self.is_done:
                self.logger.info("Done signal received; finishing buffer updating.")
                break

            with self.store_lock:
                trajectories = [self.traj_store[key] for key in self.traj_store if self.traj_store[key]]

            if len(trajectories) != self.add_batch_size:
                time.sleep(0.1)
                continue

            with self.store_lock:
                self.traj_store = {}

            batched = {k: np.stack([x[k] for x in trajectories]) for k in trajectories[0]}
            with self.buffer_lock:
                self.buffer.add(batched)

    def buffer_sampling(self):
        while True:
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
            with self.semaphore:
                for sample_on_gpu in samples_on_gpu:
                    self._samples_on_gpu.put(sample_on_gpu)

    def get_samples(self, batch_size: int = 10) -> Generator:
        while True:
            batch = []
            while True:
                if not self._samples_on_gpu.empty():
                    batch.append(self._samples_on_gpu.get())
                else:
                    self.logger.info("Not enough data; sampling generator sleeps for a second.")
                    time.sleep(1)
                if len(batch) == batch_size:
                    break
            yield batch


class Trainer(BufferKeeper):
    def __init__(
        self,
        *,
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
        get_update_epoch,
        agent_params,
        buffer,
        args_buffer,
        neptune_run,
        args_neptune_run,
        num_samples_on_gpu_cache,
        num_samples_to_gpu,
        num_semaphor,
        controller,
    ):
        super().__init__(buffer, args_buffer, num_samples_on_gpu_cache, num_samples_to_gpu, num_semaphor)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.workdir = workdir
        self.neptune_run = neptune_run(**args_neptune_run)
        self.neptune_run["workdir"] = workdir

        self.steps = steps
        self.training_iterations_per_step = training_iterations_per_step
        self.batch_size = args_buffer.sample_batch_size
        self.buffer_size = args_buffer.max_size

        self.synchronization_period = synchronization_period
        self.summary_writing_period = summary_writing_period
        self.save_period = save_period
        self.orbax_checkpointer = ocp.StandardCheckpointer()

        self._rng = jax.random.PRNGKey(trainer_seed)
        self._rng, rng = jax.random.split(self._rng)
        self.flax_state = state_recover(rng, **args_state_recover)
        self.update_epoch_fn = get_update_epoch(
            get_update_step(self.flax_state.apply_fn, agent_params), self.buffer_lock, self.buffer
        )

        self.controller = controller
        self.futures = self.controller.set_parameters.remote(jax.device_get(self.flax_state.params))
        self.is_done = False

    def training(self):
        samples_generator = self.get_samples(self.training_iterations_per_step)
        transitions_processed = 0
        for step in itertools.count(start=1, step=1):
            samples = next(samples_generator)
            self.flax_state, info = self.update_epoch_fn(self.flax_state, samples)
            transitions_processed += self.batch_size * self.training_iterations_per_step

            if step == self.steps:
                self.is_done = True
                ray.get(self.controller.set_done.remote())
                self.logger.info(f"Final training step {step} reached; finishing.")
                break

            if step % self.summary_writing_period == 0:
                self.logger.info(f"Training step: {self.flax_state.step}.")
                self.logger.info(f"Transitions sampled from restart: {transitions_processed}.")
                self.neptune_log(info, transitions_processed, samples)

            if step % self.synchronization_period == 0:
                ray.get(self.futures)
                self.futures = self.controller.set_parameters.remote(jax.device_get(self.flax_state.params))

            if step % self.save_period == 0:
                orbax_checkpoint_path = os.path.join(self.workdir, f"chkpt_train_step_{self.flax_state.step:07}")
                # self.orbax_checkpointer.save(orbax_checkpoint_path, self.flax_state)
                self.logger.info(f"Orbax checkpoint is in: {orbax_checkpoint_path}")
    
    def priority_buffer_log(self, info, samples):
        with self.buffer_lock:
            start_index = get_tree_index(self.buffer.state.priority_state.tree_depth, 0)
            all_priorities = jax.device_get(
                self.buffer.state.priority_state.nodes[start_index : start_index + self.buffer_size]
            )
        hist, bin_edges = np.histogram(all_priorities, bins=10)
        hist_str = np.array2string(hist, precision=2, separator=",", suppress_small=True)
        bin_edges_str = np.array2string(bin_edges, precision=2, separator=",", suppress_small=True)
        self.neptune_run["buffer_priorities_hist"].append(hist_str)
        self.neptune_run["buffer_priorities_bin_edges"].append(bin_edges_str)
        self.neptune_run["importance_sampling_exponent"].append(info["importance_sampling_exponent"])
        sample_cpu = jax.device_get(samples)[-1]
        priorities_str = np.array2string(sample_cpu.priorities, precision=2, separator=",", suppress_small=True)
        self.neptune_run["sample_priorities"].append(priorities_str)

    def neptune_log(self, info, transitions_processed, samples):
        self.neptune_run["step"].append(self.flax_state.step)
        with contextlib.suppress(KeyError):
            self.neptune_run["loss"].append(info["loss"])
        self.neptune_run["transitions_sampled_from_restart"].append(transitions_processed)
        with self.buffer_lock:
            self.neptune_run["buffer_current_index"].append(self.buffer.state.current_index)
        
        if "importance_sampling_exponent" in info:
            self.priority_buffer_log(info, samples)
