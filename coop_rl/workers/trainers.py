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

import jax
import numpy as np
import orbax.checkpoint as ocp
import ray
from flashbax.buffers.sum_tree import get_tree_index

from coop_rl.workers.auxiliary import BufferKeeper


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
        self.futures = self.controller.set_parameters.remote(self.flax_state.params)
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
                self.futures = self.controller.set_parameters.remote(self.flax_state.params)

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
