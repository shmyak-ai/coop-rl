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
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


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


class CommandExecutor:
    """Execute worker commands through Ray actors or local thread workers."""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Submit a method call and return a pending handle."""
        method = getattr(target, method_name)
        if hasattr(method, "remote"):
            return method.remote(*args, **kwargs)
        return self._executor.submit(method, *args, **kwargs)

    def call(self, target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a method call and return the resolved result."""
        method = getattr(target, method_name)
        if hasattr(method, "remote"):
            return self.resolve(method.remote(*args, **kwargs))
        return method(*args, **kwargs)

    def resolve(self, handle: Any) -> Any:
        """Resolve a pending handle from Ray or local thread executor."""
        if isinstance(handle, Future):
            return handle.result()

        try:
            import ray

            if isinstance(handle, ray.ObjectRef):
                return ray.get(handle)
        except (ImportError, AttributeError):
            pass

        return handle
