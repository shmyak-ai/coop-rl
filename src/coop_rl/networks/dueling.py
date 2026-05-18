#
# from Stoix https://github.com/EdanToledo/Stoix
#

from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from coop_rl.networks.epsilon_greedy import EpsilonGreedy
from coop_rl.networks.layers import NoisyLinear
from coop_rl.networks.torso import NoisyMLPTorso


class NoisyDistributionalDuelingQNetwork(nn.Module):
    num_atoms: int
    vmax: float
    vmin: float
    action_dim: int
    epsilon: float
    layer_sizes: Sequence[int]
    sigma_zero: float
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, embeddings: chex.Array) -> chex.Array:
        value_torso = NoisyMLPTorso(
            self.layer_sizes, self.activation, self.use_layer_norm, self.sigma_zero
        )(embeddings)
        advantages_torso = NoisyMLPTorso(
            self.layer_sizes, self.activation, self.use_layer_norm, self.sigma_zero
        )(embeddings)

        value_logits = NoisyLinear(self.num_atoms, sigma_zero=self.sigma_zero)(value_torso)
        value_logits = jnp.reshape(value_logits, (-1, 1, self.num_atoms))
        adv_logits = NoisyLinear(self.action_dim * self.num_atoms, sigma_zero=self.sigma_zero)(
            advantages_torso
        )
        adv_logits = jnp.reshape(adv_logits, (-1, self.action_dim, self.num_atoms))
        q_logits = value_logits + adv_logits - adv_logits.mean(axis=1, keepdims=True)

        atoms = jnp.linspace(self.vmin, self.vmax, self.num_atoms)
        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * atoms, axis=2)
        q_values = jax.lax.stop_gradient(q_values)
        atoms = jnp.broadcast_to(atoms, (q_values.shape[0], self.num_atoms))
        return EpsilonGreedy(preferences=q_values, epsilon=self.epsilon), q_logits, atoms
