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


class Controller:
    def __init__(self):
        self.done = False
        self.params_store = collections.deque(maxlen=10)

    def set_done(self):
        self.done = True

    def is_done(self) -> bool:
        return self.done

    def set_parameters(self, w):
        # add to the right side of the deque
        self.params_store.append(w)

    def get_parameters(self):
        try:
            # pop from the right side
            w = self.params_store.pop()
        except IndexError:
            return None
        return {"params": w}

    def get_parameters_done(self):
        return self.get_parameters(), self.done

    def store_size(self):
        return len(self.params_store)
