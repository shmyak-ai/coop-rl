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
#
# Distribution / output heads for DreamerV3, ported verbatim (math unchanged)
# from the original ninjax implementation (embodied.jax.outs). These are pure
# JAX and framework-agnostic.

import jax
import jax.numpy as jnp

i32 = jnp.int32
f32 = jnp.float32
sg = jax.lax.stop_gradient


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


class Output:
    def __repr__(self):
        name = type(self).__name__
        pred = self.pred()
        return f"{name}({pred.dtype}, shape={pred.shape})"

    def pred(self):
        raise NotImplementedError

    def loss(self, target):
        return -self.logp(sg(target))

    def sample(self, seed, shape=()):
        raise NotImplementedError

    def logp(self, event):
        raise NotImplementedError

    def prob(self, event):
        return jnp.exp(self.logp(event))

    def entropy(self):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError


class Agg(Output):
    def __init__(self, output, dims, agg=jnp.sum):
        self.output = output
        self.axes = [-i for i in range(1, dims + 1)]
        self.agg = agg

    def __repr__(self):
        name = type(self.output).__name__
        pred = self.pred()
        dims = len(self.axes)
        return f"{name}({pred.dtype}, shape={pred.shape}, agg={dims})"

    def pred(self):
        return self.output.pred()

    def loss(self, target):
        loss = self.output.loss(target)
        return self.agg(loss, self.axes)

    def sample(self, seed, shape=()):
        return self.output.sample(seed, shape)

    def logp(self, event):
        return self.output.logp(event).sum(self.axes)

    def prob(self, event):
        return self.output.prob(event).sum(self.axes)

    def entropy(self):
        entropy = self.output.entropy()
        return self.agg(entropy, self.axes)

    def kl(self, other):
        assert isinstance(other, Agg), other
        kl = self.output.kl(other.output)
        return self.agg(kl, self.axes)


class MSE(Output):
    def __init__(self, mean, squash=None):
        self.mean = f32(mean)
        self.squash = squash or (lambda x: x)

    def pred(self):
        return self.mean

    def loss(self, target):
        assert jnp.issubdtype(target.dtype, jnp.floating), target.dtype
        assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
        return jnp.square(self.mean - sg(self.squash(f32(target))))


class Huber(Output):
    def __init__(self, mean, eps=1.0):
        # Soft Huber loss or Charbonnier loss.
        self.mean = f32(mean)
        self.eps = eps

    def pred(self):
        return self.mean

    def loss(self, target):
        assert jnp.issubdtype(target.dtype, jnp.floating), target.dtype
        assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
        dist = self.mean - sg(f32(target))
        return jnp.sqrt(jnp.square(dist) + jnp.square(self.eps)) - self.eps


class Normal(Output):
    def __init__(self, mean, stddev=1.0):
        self.mean = f32(mean)
        self.stddev = jnp.broadcast_to(f32(stddev), self.mean.shape)

    def pred(self):
        return self.mean

    def sample(self, seed, shape=()):
        sample = jax.random.normal(seed, shape + self.mean.shape, f32)
        return sample * self.stddev + self.mean

    def logp(self, event):
        assert jnp.issubdtype(event.dtype, jnp.floating), event.dtype
        return jax.scipy.stats.norm.logpdf(f32(event), self.mean, self.stddev)

    def entropy(self):
        return 0.5 * jnp.log(2 * jnp.pi * jnp.square(self.stddev)) + 0.5

    def kl(self, other):
        assert isinstance(other, type(self)), (self, other)
        return 0.5 * (
            jnp.square(self.stddev / other.stddev)
            + jnp.square(other.mean - self.mean) / jnp.square(other.stddev)
            + 2 * jnp.log(other.stddev)
            - 2 * jnp.log(self.stddev)
            - 1
        )


class Binary(Output):
    def __init__(self, logit):
        self.logit = f32(logit)

    def pred(self):
        return self.logit > 0

    def logp(self, event):
        event = f32(event)
        logp = jax.nn.log_sigmoid(self.logit)
        lognotp = jax.nn.log_sigmoid(-self.logit)
        return event * logp + (1 - event) * lognotp

    def sample(self, seed, shape=()):
        prob = jax.nn.sigmoid(self.logit)
        return jax.random.bernoulli(seed, prob, shape + self.logit.shape)


class Categorical(Output):
    def __init__(self, logits, unimix=0.0):
        logits = f32(logits)
        if unimix:
            probs = jax.nn.softmax(logits, -1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - unimix) * probs + unimix * uniform
            logits = jnp.log(probs)
        self.logits = logits

    def pred(self):
        return jnp.argmax(self.logits, -1)

    def sample(self, seed, shape=()):
        return jax.random.categorical(seed, self.logits, -1, shape + self.logits.shape[:-1])

    def logp(self, event):
        onehot = jax.nn.one_hot(event, self.logits.shape[-1])
        return (jax.nn.log_softmax(self.logits, -1) * onehot).sum(-1)

    def entropy(self):
        logprob = jax.nn.log_softmax(self.logits, -1)
        prob = jax.nn.softmax(self.logits, -1)
        entropy = -(prob * logprob).sum(-1)
        return entropy

    def kl(self, other):
        logprob = jax.nn.log_softmax(self.logits, -1)
        logother = jax.nn.log_softmax(other.logits, -1)
        prob = jax.nn.softmax(self.logits, -1)
        return (prob * (logprob - logother)).sum(-1)


class OneHot(Output):
    def __init__(self, logits, unimix=0.0):
        self.dist = Categorical(logits, unimix)

    def pred(self):
        index = self.dist.pred()
        return self._onehot_with_grad(index)

    def sample(self, seed, shape=()):
        index = self.dist.sample(seed, shape)
        return self._onehot_with_grad(index)

    def logp(self, event):
        return (jax.nn.log_softmax(self.dist.logits, -1) * event).sum(-1)

    def entropy(self):
        return self.dist.entropy()

    def kl(self, other):
        return self.dist.kl(other.dist)

    def _onehot_with_grad(self, index):
        # Straight through gradients.
        value = jax.nn.one_hot(index, self.dist.logits.shape[-1], dtype=f32)
        probs = jax.nn.softmax(self.dist.logits, -1)
        value = sg(value) + (probs - sg(probs))
        return value


class TwoHot(Output):
    def __init__(self, logits, bins, squash=None, unsquash=None):
        logits = f32(logits)
        assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
        assert bins.dtype == f32, bins.dtype
        self.logits = logits
        self.probs = jax.nn.softmax(logits)
        self.bins = jnp.array(bins)
        self.squash = squash or (lambda x: x)
        self.unsquash = unsquash or (lambda x: x)

    def pred(self):
        # The naive implementation results in a non-zero result even if the bins
        # are symmetric and the probabilities uniform, because the sum operation
        # goes left to right, accumulating numerical errors. Instead, we use a
        # symmetric sum to ensure that the predicted rewards and values are
        # actually zero at initialization.
        # return self.unsquash((self.probs * self.bins).sum(-1))
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
            return self.unsquash(wavg)
        else:
            p1 = self.probs[..., : n // 2]
            p2 = self.probs[..., n // 2 :]
            b1 = self.bins[..., : n // 2]
            b2 = self.bins[..., n // 2 :]
            wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)
            return self.unsquash(wavg)

    def loss(self, target):
        assert target.dtype == f32, target.dtype
        target = sg(self.squash(target))
        below = (self.bins <= target[..., None]).astype(i32).sum(-1) - 1
        above = len(self.bins) - (self.bins > target[..., None]).astype(i32).sum(-1)
        below = jnp.clip(below, 0, len(self.bins) - 1)
        above = jnp.clip(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - target))
        dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - target))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None]
            + jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None]
        )
        log_pred = self.logits - jax.scipy.special.logsumexp(self.logits, -1, keepdims=True)
        return -(target * log_pred).sum(-1)
