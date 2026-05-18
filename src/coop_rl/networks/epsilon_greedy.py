import chex
import jax
import jax.numpy as jnp


class EpsilonGreedy:
    """Epsilon-greedy action selection over Q-value preferences."""

    def __init__(self, preferences: chex.Array, epsilon: float):
        self.preferences = preferences
        self.epsilon = epsilon

    def sample(self, seed: chex.PRNGKey) -> chex.Array:
        key1, key2 = jax.random.split(seed)
        greedy = jnp.argmax(self.preferences, axis=-1)
        random = jax.random.randint(key2, greedy.shape, 0, self.preferences.shape[-1])
        return jnp.where(jax.random.uniform(key1) < self.epsilon, random, greedy)
