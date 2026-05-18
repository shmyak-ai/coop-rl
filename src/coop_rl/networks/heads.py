#
# from Stoix https://github.com/EdanToledo/Stoix
#

from typing import Any

import chex
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from coop_rl.networks.epsilon_greedy import EpsilonGreedy


class ScalarCriticHead(nn.Module):
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        return nn.Dense(1, kernel_init=self.kernel_init)(embedding).squeeze(axis=-1)


class DiscreteQNetworkHead(nn.Module):
    action_dim: int
    epsilon: float = 0.1
    kernel_init: Initializer = orthogonal(1.0)
    dtype: Any = None

    @nn.compact
    def __call__(self, embedding: chex.Array) -> EpsilonGreedy:
        q_values = nn.Dense(self.action_dim, kernel_init=self.kernel_init, dtype=self.dtype)(
            embedding
        )
        return EpsilonGreedy(preferences=q_values, epsilon=self.epsilon)


class LinearHead(nn.Module):
    output_dim: int
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        return nn.Dense(self.output_dim, kernel_init=self.kernel_init)(embedding)
