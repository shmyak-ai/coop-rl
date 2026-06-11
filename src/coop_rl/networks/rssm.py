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
# DreamerV3 world model: RSSM, image Encoder, image Decoder. Flax port of the
# original ninjax implementation (dreamerv3.rssm). The recurrence math (block
# GRU core, prior/posterior, OneHot KL) is preserved unchanged. Actions are
# embedded by the caller (the agent owns the action space); the RSSM consumes
# pre-embedded float action vectors, which is mathematically identical to the
# original (the reset mask zeros the embedded action on episode boundaries).

import math

import chex
import einops
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from coop_rl.base import distributions as outs
from coop_rl.networks.dreamer_nn import (
    BlockLinear,
    Conv2D,
    Linear,
    Norm,
    cast,
    get_act,
    mask,
)

f32 = jnp.float32
sg = jax.lax.stop_gradient


class RSSM(nn.Module):
    deter: int = 4096
    hidden: int = 2048
    stoch: int = 32
    classes: int = 32
    norm: str = "rms"
    act: str = "gelu"
    unimix: float = 0.01
    outscale: float = 1.0
    imglayers: int = 2
    obslayers: int = 1
    dynlayers: int = 1
    blocks: int = 8
    free_nats: float = 1.0

    def setup(self):
        assert self.deter % self.blocks == 0
        h = self.hidden
        # Posterior (observation) layers.
        self.obs_lin = [Linear(h, name=f"obs{i}") for i in range(self.obslayers)]
        self.obs_norm = [Norm(self.norm, name=f"obs{i}norm") for i in range(self.obslayers)]
        self.obslogit_lin = Linear(
            self.stoch * self.classes, outscale=self.outscale, name="obslogit"
        )
        # Core (block GRU) layers.
        self.dynin0 = Linear(h, name="dynin0")
        self.dynin0norm = Norm(self.norm, name="dynin0norm")
        self.dynin1 = Linear(h, name="dynin1")
        self.dynin1norm = Norm(self.norm, name="dynin1norm")
        self.dynin2 = Linear(h, name="dynin2")
        self.dynin2norm = Norm(self.norm, name="dynin2norm")
        self.dynhid = [
            BlockLinear(self.deter, self.blocks, name=f"dynhid{i}") for i in range(self.dynlayers)
        ]
        self.dynhidnorm = [Norm(self.norm, name=f"dynhid{i}norm") for i in range(self.dynlayers)]
        self.dyngru = BlockLinear(3 * self.deter, self.blocks, name="dyngru")
        # Prior (imagination) layers.
        self.prior_lin = [Linear(h, name=f"prior{i}") for i in range(self.imglayers)]
        self.prior_norm = [Norm(self.norm, name=f"prior{i}norm") for i in range(self.imglayers)]
        self.priorlogit_lin = Linear(
            self.stoch * self.classes, outscale=self.outscale, name="priorlogit"
        )

    def initial(self, bsize):
        return cast(
            dict(
                deter=jnp.zeros([bsize, self.deter], f32),
                stoch=jnp.zeros([bsize, self.stoch, self.classes], f32),
            )
        )

    def truncate(self, entries, carry=None):
        assert entries["deter"].ndim == 3, entries["deter"].shape
        return jax.tree.map(lambda x: x[:, -1], entries)

    def starts(self, entries, carry, nlast):
        b = len(jax.tree.leaves(carry)[0])
        return jax.tree.map(lambda x: x[:, -nlast:].reshape((b * nlast, *x.shape[2:])), entries)

    # -- single-step transitions -------------------------------------------

    def observe_step(self, carry, tokens, action, reset):
        deter, stoch, action = mask((carry["deter"], carry["stoch"], action), ~reset)
        deter = self._core(deter, stoch, action)
        tokens = tokens.reshape((*deter.shape[:-1], -1))
        x = jnp.concatenate([deter, tokens], -1)
        for lin, norm in zip(self.obs_lin, self.obs_norm, strict=True):
            x = get_act(self.act)(norm(lin(x)))
        logit = self._logit(self.obslogit_lin, x)
        stoch = cast(self._dist(logit).sample(self.make_rng("sample")))
        carry = dict(deter=deter, stoch=stoch)
        feat = dict(deter=deter, stoch=stoch, logit=logit)
        entry = dict(deter=deter, stoch=stoch)
        return carry, (entry, feat)

    def imagine_step(self, carry, action):
        deter = self._core(carry["deter"], carry["stoch"], action)
        logit = self._prior(deter)
        stoch = cast(self._dist(logit).sample(self.make_rng("sample")))
        carry = cast(dict(deter=deter, stoch=stoch))
        feat = cast(dict(deter=deter, stoch=stoch, logit=logit))
        return carry, feat

    # -- sequence wrappers --------------------------------------------------

    def observe(self, carry, tokens, action, reset):
        carry, tokens, action = cast((carry, tokens, action))

        def step(mdl, c, inputs):
            tok, act, res = inputs
            return mdl.observe_step(c, tok, act, res)

        scan = nn.scan(
            step,
            variable_broadcast="params",
            split_rngs={"sample": True, "params": False},
            in_axes=1,
            out_axes=1,
        )
        carry, (entries, feat) = scan(self, carry, (tokens, action, reset))
        return carry, entries, feat

    def loss(self, carry, tokens, action, reset):
        carry, entries, feat = self.observe(carry, tokens, action, reset)
        prior = self._prior(feat["deter"])
        post = feat["logit"]
        dyn = self._dist(sg(post)).kl(self._dist(prior))
        rep = self._dist(post).kl(self._dist(sg(prior)))
        if self.free_nats:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)
        losses = {"dyn": dyn, "rep": rep}
        metrics = {
            "dyn_ent": self._dist(prior).entropy().mean(),
            "rep_ent": self._dist(post).entropy().mean(),
        }
        return carry, entries, losses, feat, metrics

    # -- internals ----------------------------------------------------------

    def _core(self, deter, stoch, action):
        stoch = stoch.reshape((stoch.shape[0], -1))
        action = action / sg(jnp.maximum(1, jnp.abs(action)))
        g = self.blocks
        flat2group = lambda x: einops.rearrange(x, "... (g h) -> ... g h", g=g)  # noqa: E731
        group2flat = lambda x: einops.rearrange(x, "... g h -> ... (g h)", g=g)  # noqa: E731
        x0 = get_act(self.act)(self.dynin0norm(self.dynin0(deter)))
        x1 = get_act(self.act)(self.dynin1norm(self.dynin1(stoch)))
        x2 = get_act(self.act)(self.dynin2norm(self.dynin2(action)))
        x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
        x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
        for hid, norm in zip(self.dynhid, self.dynhidnorm, strict=True):
            x = get_act(self.act)(norm(hid(x)))
        x = self.dyngru(x)
        gates = jnp.split(flat2group(x), 3, -1)
        reset, cand, update = [group2flat(y) for y in gates]
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter

    def _prior(self, feat):
        x = feat
        for lin, norm in zip(self.prior_lin, self.prior_norm, strict=True):
            x = get_act(self.act)(norm(lin(x)))
        return self._logit(self.priorlogit_lin, x)

    def _logit(self, layer, x):
        x = layer(x)
        return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

    def _dist(self, logits):
        return outs.Agg(outs.OneHot(logits, self.unimix), 1, jnp.sum)


class Encoder(nn.Module):
    """Image-only encoder (DreamerV3 'simple' encoder, Atari path)."""

    img_keys: tuple[str, ...]
    norm: str = "rms"
    act: str = "gelu"
    depth: int = 64
    mults: tuple[int, ...] = (2, 3, 4, 4)
    kernel: int = 5

    @property
    def depths(self):
        return tuple(self.depth * m for m in self.mults)

    @nn.compact
    def __call__(self, obs, reset) -> chex.Array:
        bshape = reset.shape
        bdims = len(bshape)
        imgs = [obs[k] for k in sorted(self.img_keys)]
        assert all(x.dtype == jnp.uint8 for x in imgs)
        x = cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
        x = x.reshape((-1, *x.shape[bdims:]))
        for i, depth in enumerate(self.depths):
            x = Conv2D(depth, self.kernel, name=f"cnn{i}")(x)
            b, h, w, c = x.shape
            x = x.reshape((b, h // 2, 2, w // 2, 2, c)).max((2, 4))
            x = get_act(self.act)(Norm(self.norm, name=f"cnn{i}norm")(x))
        assert 3 <= x.shape[-3] <= 16, x.shape
        assert 3 <= x.shape[-2] <= 16, x.shape
        x = x.reshape((x.shape[0], -1))
        tokens = x.reshape((*bshape, *x.shape[1:]))
        return tokens


class Decoder(nn.Module):
    """Image-only decoder (DreamerV3 'simple' decoder, Atari path)."""

    img_keys: tuple[str, ...]
    img_res: tuple[int, int]
    img_channels: tuple[int, ...]
    units: int = 1024
    norm: str = "rms"
    act: str = "gelu"
    outscale: float = 1.0
    depth: int = 64
    mults: tuple[int, ...] = (2, 3, 4, 4)
    kernel: int = 5
    bspace: int = 8

    @property
    def depths(self):
        return tuple(self.depth * m for m in self.mults)

    @nn.compact
    def __call__(self, feat, reset):
        depths = self.depths
        imgdep = sum(self.img_channels)
        k = self.kernel
        bshape = reset.shape
        factor = 2 ** len(depths)
        minres = [int(x // factor) for x in self.img_res]
        assert 3 <= minres[0] <= 16, minres
        assert 3 <= minres[1] <= 16, minres
        shape = (*minres, depths[-1])
        u, g = math.prod(shape), self.bspace
        x0, x1 = cast((feat["deter"], feat["stoch"]))
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        x0 = BlockLinear(u, g, name="sp0")(x0)
        x0 = einops.rearrange(x0, "... (g h w c) -> ... h w (g c)", h=minres[0], w=minres[1], g=g)
        x1 = get_act(self.act)(
            Norm(self.norm, name="sp1norm")(Linear(2 * self.units, name="sp1")(x1))
        )
        x1 = Linear(shape, name="sp2")(x1)
        x = get_act(self.act)(Norm(self.norm, name="spnorm")(x0 + x1))
        for i, depth in reversed(list(enumerate(depths[:-1]))):
            x = x.repeat(2, -2).repeat(2, -3)
            x = Conv2D(depth, k, name=f"conv{i}")(x)
            x = get_act(self.act)(Norm(self.norm, name=f"conv{i}norm")(x))
        x = x.repeat(2, -2).repeat(2, -3)
        x = Conv2D(imgdep, k, outscale=self.outscale, name="imgout")(x)
        x = jax.nn.sigmoid(x)
        x = x.reshape((*bshape, *x.shape[1:]))
        recons = {}
        split = np.cumsum(list(self.img_channels)[:-1])
        for key, out in zip(sorted(self.img_keys), jnp.split(x, split, -1), strict=True):
            out = outs.MSE(out)
            out = outs.Agg(out, 3, jnp.sum)
            recons[key] = out
        return recons
