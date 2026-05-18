#
# from Stoix https://github.com/EdanToledo/Stoix
#

from typing import Any

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np


class AffineTanhTransformedDistribution(distrax.Transformed):
    """Distribution followed by tanh and then affine transformations."""

    def __init__(
        self,
        distribution: distrax.DistributionLike,
        minimum: float,
        maximum: float,
        epsilon: float = 1e-3,
    ) -> None:
        """Initialize the distribution with a tanh and affine bijector.

        Args:
          distribution: The distribution to transform.
          minimum: Lower bound of the target range.
          maximum: Upper bound of the target range.
          epsilon: epsilon value for numerical stability.
            epsilon is used to compute the log of the average probability distribution
            outside the clipping range, i.e. on the interval
            [-inf, atanh(inverse_affine(minimum))] for log_prob_left and
            [atanh(inverse_affine(maximum)), inf] for log_prob_right.
        """
        # Calculate scale and shift for the affine transformation to achieve the range
        # [minimum, maximum] after the tanh.
        scale = (maximum - minimum) / 2.0
        shift = (minimum + maximum) / 2.0

        # Chain the bijectors: distrax.Chain applies last-to-first, so Tanh runs first.
        joint_bijector = distrax.Chain([distrax.ScalarAffine(shift=shift, scale=scale), distrax.Tanh()])

        super().__init__(distribution=distribution, bijector=joint_bijector)

        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, atanh(inverse_affine(minimum))] for
        # log_prob_left and [atanh(inverse_affine(maximum)), inf] for log_prob_right.
        self._min_threshold = minimum + epsilon
        self._max_threshold = maximum - epsilon
        min_inverse_threshold = self.bijector.inverse(jnp.asarray(self._min_threshold))
        max_inverse_threshold = self.bijector.inverse(jnp.asarray(self._max_threshold))
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(epsilon)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = self.distribution.log_cdf(min_inverse_threshold) - log_epsilon
        self._log_prob_right = (
            self.distribution.log_survival_function(max_inverse_threshold) - log_epsilon
        )

    def log_prob(self, event: chex.Array) -> chex.Array:
        # Without this clip there would be NaNs in the inner jnp.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, self._min_threshold, self._max_threshold)
        return jnp.where(
            event <= self._min_threshold,
            self._log_prob_left,
            jnp.where(event >= self._max_threshold, self._log_prob_right, super().log_prob(event)),
        )

    def mode(self) -> chex.Array:
        return self.bijector.forward(self.distribution.mode())

    def entropy(self, seed: chex.PRNGKey = None) -> chex.Array:
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed)
        )


class ClippedBeta(distrax.Beta):
    """Beta distribution with clipped samples."""

    def _sample_n(self, key: chex.PRNGKey, n: int) -> chex.Array:
        _epsilon = 1e-7
        samples = super()._sample_n(key, n)
        return jnp.clip(samples, _epsilon, 1 - _epsilon)


class DiscreteValuedDistribution:
    """Categorical distribution over a real-valued support.

    The support can be any real valued range, whereas a standard Categorical
    distribution has integer support [0, n_categories - 1]. This generalization
    allows computing the mean and variance of the distribution over its support.
    """

    def __init__(
        self,
        values: chex.Array,
        logits: chex.Array | None = None,
        probs: chex.Array | None = None,
        name: str = "DiscreteValuedDistribution",
    ):
        """Initialization.

        Args:
          values: Values making up support of the distribution. Should have a shape
            compatible with logits.
          logits: An N-D Tensor, N >= 1, representing the log probabilities of a set
            of Categorical distributions. The first N - 1 dimensions index into a
            batch of independent distributions and the last dimension indexes into
            the classes.
          probs: An N-D Tensor, N >= 1, representing the probabilities of a set of
            Categorical distributions. Only one of logits or probs should be passed in.
          name: Name of the distribution object.
        """
        self._values = np.asarray(values)
        if logits is not None:
            self._logits = jnp.asarray(logits)
        elif probs is not None:
            self._logits = jnp.log(jnp.asarray(probs))
        else:
            raise ValueError("One of logits or probs must be provided.")

    @property
    def values(self) -> chex.Array:
        return self._values

    @property
    def logits(self) -> chex.Array:
        return self._logits

    @property
    def probs(self) -> chex.Array:
        return jax.nn.softmax(self._logits)

    def sample(self, seed: chex.PRNGKey, sample_shape: tuple[int, ...] = ()) -> chex.Array:
        indices = jax.random.categorical(seed, self._logits)
        return jnp.take(self._values, indices, axis=-1)

    def mean(self) -> chex.Array:
        return jnp.sum(self.probs * self._values, axis=-1)

    def variance(self) -> chex.Array:
        dist_squared = jnp.square(jnp.expand_dims(self.mean(), -1) - self._values)
        return jnp.sum(self.probs * dist_squared, axis=-1)


# Preserve old name as alias so any external code referencing it still works.
DiscreteValuedTfpDistribution = DiscreteValuedDistribution
