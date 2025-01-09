# Copyright 2024 The Coop RL Authors.
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
import logging
import threading
import time
from collections.abc import Generator
from queue import Queue

import jax
import numpy as np


class Controller:
    def __init__(self):
        self.done = False
        self.params_store = collections.deque(maxlen=10)  # FIFO

    def set_done(self):
        self.done = True

    def is_done(self) -> bool:
        return self.done

    def set_parameters(self, w):
        self.params_store.append(w)

    def get_parameters(self):
        try:
            w = self.params_store.popleft()
        except IndexError:
            return None
        return w

    def get_parameters_done(self):
        return self.get_parameters(), self.done

    def store_size(self):
        return len(self.params_store)


class BufferKeeper:
    def __init__(self, buffer, args_buffer, num_samples_on_gpu_cache, num_samples_to_gpu):
        self.buffer = buffer(**args_buffer)
        self.traj_store = {}
        self.add_batch_size = args_buffer.add_batch_size
        self.buffer_lock = threading.Lock()
        self.store_lock = threading.Lock()
        self._samples_on_gpu = Queue(maxsize=num_samples_on_gpu_cache)
        self.gpu_device = jax.devices("gpu")[0]
        self.num_samples_to_gpu = num_samples_to_gpu

    def add_traj_seq(self, data):
        # a trick to start a ray thread, is it necessary?
        if data == 1:
            return
        # save data from a collector
        collector_seed, *_data = data
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

            transposed = list(zip(*trajectories, strict=True))
            merged = [np.stack(arrays, axis=0) for arrays in transposed]

            with self.buffer_lock:
                self.buffer.add(*merged)

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
            samples_on_gpu = jax.device_put(samples, device=self.gpu_device)
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
            if self.logger.getEffectiveLevel() == logging.DEBUG:
                self.logger.debug(f"Samples queue size: {self._samples_on_gpu.qsize()}")
                with self.buffer_lock:
                    self.logger.debug(f"Buffer current index: {self.buffer.state.current_index}")
                    self.logger.debug(f"Buffer is full: {self.buffer.state.is_full}")
            yield batch
