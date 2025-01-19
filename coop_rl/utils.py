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

import functools
import time
from collections.abc import Callable

import optax


def make_learning_rate_schedule(init_lr: float, num_updates: int, num_epochs: int, num_minibatches: int) -> Callable:
    """
    We use a simple linear learning rate scheduler based on the suggestions from a blog on PPO
    implementation details which can be viewed at http://tinyurl.com/mr3chs4p
    relevant arguments to the system config and then parsing them accordingly here.
    """

    def linear_scedule(count: int) -> float:
        frac: float = 1.0 - (count // (num_epochs * num_minibatches)) / num_updates
        return init_lr * frac

    return linear_scedule


def make_learning_rate(
    *,
    init_lr: float,
    decay_learning_rates: bool,
    num_updates: int | None = None,
    num_epochs: int | None = None,
    num_minibatches: int | None = None,
) -> float | Callable:
    if num_minibatches is None:
        num_minibatches = 1

    if decay_learning_rates:
        return make_learning_rate_schedule(init_lr, num_updates, num_epochs, num_minibatches)
    else:
        return init_lr


def make_optimizer(*, max_grad_norm: float, **kwargs):
    lr = make_learning_rate(**kwargs)
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )


def timeit(func):
    """Decorator to measure and report the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        print(f"Function '{func.__name__}' took {execution_time:.4f} seconds to complete.")
        return result

    return wrapper
