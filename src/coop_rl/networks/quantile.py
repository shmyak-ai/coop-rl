#
# IQN-style implicit quantile head, following Dabney et al. 2018 (arXiv:1806.06923).
#

from collections.abc import Sequence
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from coop_rl.networks.epsilon_greedy import EpsilonGreedy
from coop_rl.networks.torso import MLPTorso


class DuelingQuantileQNetworkHead(nn.Module):
    action_dim: int
    epsilon: float
    layer_sizes: Sequence[int]
    activation: str = "relu"
    n_cos: int = 64
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    dtype: Any = None

    @nn.compact
    def __call__(
        self, embedding: chex.Array, num_quantiles: int, epsilon: float | None = None
    ) -> tuple[EpsilonGreedy, chex.Array, chex.Array]:
        """Returns (epsilon-greedy policy over mean-z, z (..., N, A), quantiles (..., N)).

        `epsilon` overrides the module's static `self.epsilon` when given (e.g. for a
        schedule computed by the caller); defaults to `self.epsilon` otherwise.
        """
        quantiles = jax.random.uniform(
            self.make_rng("quantiles"), (*embedding.shape[:-1], num_quantiles)
        )
        # Cosine embedding of the quantile fractions: cos(pi * i * sigma), i = 1..n_cos.
        cos_embedding = jnp.cos(
            jnp.pi * quantiles[..., jnp.newaxis] * jnp.arange(1, self.n_cos + 1)
        )
        phi = nn.relu(
            nn.Dense(embedding.shape[-1], kernel_init=self.kernel_init, dtype=self.dtype)(
                cos_embedding.astype(embedding.dtype)
            )
        )
        # Hadamard product of the state embedding with each quantile embedding.
        x = embedding[..., jnp.newaxis, :] * phi

        value_torso = MLPTorso(
            self.layer_sizes,
            self.activation,
            self.use_layer_norm,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        advantages_torso = MLPTorso(
            self.layer_sizes,
            self.activation,
            self.use_layer_norm,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), dtype=self.dtype)(value_torso)
        advantages = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), dtype=self.dtype)(
            advantages_torso
        )
        z_values = value + advantages - advantages.mean(axis=-1, keepdims=True)

        q_values = z_values.mean(axis=-2)
        eps = self.epsilon if epsilon is None else epsilon
        return EpsilonGreedy(preferences=q_values, epsilon=eps), z_values, quantiles
