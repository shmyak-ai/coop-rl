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
# DreamerV3 agent on flax/optax (no ninjax). The world model (RSSM, encoder,
# decoder), heads and distributions live under coop_rl.networks / coop_rl.base;
# this module wires them into one composite flax module and exposes the coop_rl
# agent contract (TrainState, create/restore, get_select_action_fn,
# get_update_step, get_update_epoch). The algorithm math (imag_loss, repl_loss,
# lambda_return, KL balancing) is a faithful port of the original.

from collections.abc import Callable
from typing import Any

import chex
import elements
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flashbax.buffers.trajectory_buffer import TrajectoryBufferSample
from flax import linen as nn
from flax import struct

from coop_rl.networks.dreamer_nn import COMPUTE_DTYPE, cast
from coop_rl.networks.dreamer_nn import where as nn_where
from coop_rl.networks.heads import DictMLPHead, HeadSpec, MLPHead
from coop_rl.networks.rssm import RSSM, Decoder, Encoder

f32 = jnp.float32
i32 = jnp.int32


def sg(xs, skip=False):
    return xs if skip else jax.lax.stop_gradient(xs)


def concat(xs, axis):
    return jax.tree.map(lambda *x: jnp.concatenate(x, axis), *xs)


def take_outs(outs_tree):
    outs_tree = jax.tree.map(lambda x: x.__array__(), outs_tree)
    outs_tree = jax.tree.map(lambda x: np.float32(x) if x.dtype == jnp.bfloat16 else x, outs_tree)
    return outs_tree


# ---------------------------------------------------------------------------
# Running normalizer (return / value / advantage). Stateful via the flax
# 'stats' mutable collection. Port of embodied.jax.utils.Normalize (single
# device, so the cross-host pmean is dropped).
# ---------------------------------------------------------------------------


class Normalize(nn.Module):
    impl: str
    rate: float = 0.01
    limit: float = 1e-8
    perclo: float = 5.0
    perchi: float = 95.0
    debias: bool = True

    def setup(self):
        if self.debias and self.impl != "none":
            self.corr = self.variable("stats", "corr", lambda: jnp.zeros((), f32))
        if self.impl == "meanstd":
            self.mean = self.variable("stats", "mean", lambda: jnp.zeros((), f32))
            self.sqrs = self.variable("stats", "sqrs", lambda: jnp.zeros((), f32))
        elif self.impl == "perc":
            self.lo = self.variable("stats", "lo", lambda: jnp.zeros((), f32))
            self.hi = self.variable("stats", "hi", lambda: jnp.zeros((), f32))
        elif self.impl != "none":
            raise NotImplementedError(self.impl)

    def __call__(self, x, update):
        if update:
            self.update(x)
        return self.stats()

    def update(self, x):
        x = sg(f32(x))
        if self.impl == "meanstd":
            self._ema(self.mean, x.mean())
            self._ema(self.sqrs, jnp.square(x).mean())
        elif self.impl == "perc":
            self._ema(self.lo, jnp.percentile(x, self.perclo))
            self._ema(self.hi, jnp.percentile(x, self.perchi))
        if self.debias and self.impl != "none":
            self._ema(self.corr, 1.0)

    def stats(self):
        corr = 1.0
        if self.debias and self.impl != "none":
            corr = corr / jnp.maximum(self.rate, self.corr.value)
        if self.impl == "none":
            return 0.0, 1.0
        elif self.impl == "meanstd":
            mean = self.mean.value * corr
            std = jnp.sqrt(jax.nn.relu(self.sqrs.value * corr - mean**2))
            std = jnp.maximum(self.limit, std)
            return mean, std
        elif self.impl == "perc":
            lo, hi = self.lo.value * corr, self.hi.value * corr
            return sg(lo), sg(jnp.maximum(self.limit, hi - lo))
        else:
            raise NotImplementedError(self.impl)

    def _ema(self, var, val):
        var.value = (1 - self.rate) * var.value + self.rate * sg(val)


# ---------------------------------------------------------------------------
# Composite world-model + actor-critic module.
# ---------------------------------------------------------------------------


class DreamerModel(nn.Module):
    enc: Encoder
    dyn: RSSM
    dec: Decoder
    rew: MLPHead
    con: MLPHead
    pol: DictMLPHead
    val: MLPHead
    slowval: MLPHead
    retnorm: Normalize
    valnorm: Normalize
    advnorm: Normalize

    act_keys: tuple[str, ...]
    act_classes: tuple[int, ...]
    img_keys: tuple[str, ...]
    contdisc: bool
    horizon: int
    imag_length: int
    imag_last: int
    ac_grads: bool
    reward_grad: bool
    repval_loss: bool
    repval_grad: bool
    replay_context: int
    loss_scales: tuple[tuple[str, float], ...]
    imag_lam: float
    imag_actent: float
    imag_slowreg: float
    imag_slowtar: bool
    repl_lam: float
    repl_slowreg: float
    repl_slowtar: bool

    def feat2tensor(self, x):
        stoch = x["stoch"].reshape((*x["stoch"].shape[:-2], -1))
        return jnp.concatenate([cast(x["deter"]), cast(stoch)], -1)

    def _embed_action(self, action):
        parts = []
        for k, cls in zip(self.act_keys, self.act_classes, strict=True):
            a = action[k].astype(jnp.int32)
            parts.append(jax.nn.one_hot(a, cls, dtype=COMPUTE_DTYPE))
        return jnp.concatenate(parts, -1)

    def initial_carry(self, batch_size):
        enc_carry = {}
        dyn_carry = self.dyn.initial(batch_size)
        dec_carry = {}
        prevact = {k: jnp.zeros((batch_size,), i32) for k in self.act_keys}
        return (enc_carry, dyn_carry, dec_carry, prevact)

    def policy(self, carry, obs):
        enc_carry, dyn_carry, dec_carry, prevact = carry
        reset = obs["is_first"]
        tokens = self.enc(obs, reset)
        action_emb = self._embed_action(prevact)
        dyn_carry, (dyn_entry, feat) = self.dyn.observe_step(dyn_carry, tokens, action_emb, reset)
        polout = self.pol(self.feat2tensor(feat), 1)
        act = {k: v.sample(self.make_rng("sample")) for k, v in polout.items()}
        out = elements.tree.flatdict({"dyn": dyn_entry})
        carry = (enc_carry, dyn_carry, dec_carry, act)
        return carry, act, out

    def _imagine(self, start_carry, length):
        def step(mdl, carry, _):
            feat = sg(carry)
            polout = mdl.pol(mdl.feat2tensor(feat), 1)
            act = {k: v.sample(mdl.make_rng("sample")) for k, v in polout.items()}
            act_emb = mdl._embed_action(act)
            new_carry, newfeat = mdl.dyn.imagine_step(carry, act_emb)
            return new_carry, (newfeat, act)

        scan = nn.scan(
            step,
            variable_broadcast="params",
            split_rngs={"sample": True, "params": False},
            in_axes=0,
            out_axes=1,
        )
        _, (feat, act) = scan(self, start_carry, jnp.arange(length))
        return feat, act

    def loss(self, carry, obs, prevact, training):
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs["is_first"]
        b, t = reset.shape
        losses = {}
        metrics = {}

        # World model.
        tokens = self.enc(obs, reset)
        action_emb = self._embed_action(prevact)
        dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
            dyn_carry, tokens, action_emb, reset
        )
        losses.update(los)
        metrics.update(mets)
        recons = self.dec(repfeat, reset)
        inp = sg(self.feat2tensor(repfeat), skip=self.reward_grad)
        losses["rew"] = self.rew(inp, 2).loss(obs["reward"].astype(f32))
        con = (~obs["is_terminal"]).astype(f32)
        if self.contdisc:
            con *= 1 - 1 / self.horizon
        losses["con"] = self.con(self.feat2tensor(repfeat), 2).loss(con)
        for key in self.img_keys:
            target = obs[key].astype(f32) / 255
            losses[key] = recons[key].loss(sg(target))

        # Imagination.
        k = min(self.imag_last or t, t)
        h = self.imag_length
        starts = self.dyn.starts(dyn_entries, dyn_carry, k)
        imgfeat_s, imgact_s = self._imagine(starts, h)
        first = jax.tree.map(lambda x: x[:, -k:].reshape((b * k, 1, *x.shape[2:])), repfeat)
        imgfeat = concat([sg(first, skip=self.ac_grads), sg(imgfeat_s)], 1)
        lastfeat = jax.tree.map(lambda x: x[:, -1], imgfeat)
        lastpol = self.pol(self.feat2tensor(lastfeat), 1)
        lastact = {key: v.sample(self.make_rng("sample")) for key, v in lastpol.items()}
        lastact = jax.tree.map(lambda x: x[:, None], lastact)
        imgact = concat([imgact_s, lastact], 1)
        inp = self.feat2tensor(imgfeat)
        los, imgloss_out, mets = imag_loss(
            imgact,
            self.rew(inp, 2).pred(),
            self.con(inp, 2).prob(1),
            self.pol(inp, 2),
            self.val(inp, 2),
            self.slowval(inp, 2),
            self.retnorm,
            self.valnorm,
            self.advnorm,
            update=training,
            contdisc=self.contdisc,
            horizon=self.horizon,
            slowtar=self.imag_slowtar,
            lam=self.imag_lam,
            actent=self.imag_actent,
            slowreg=self.imag_slowreg,
        )
        losses.update({key: v.mean(1).reshape((b, k)) for key, v in los.items()})
        metrics.update(mets)

        # Replay value loss.
        if self.repval_loss:
            feat = sg(repfeat, skip=self.repval_grad)
            last, term, rew = obs["is_last"], obs["is_terminal"], obs["reward"].astype(f32)
            boot = imgloss_out["ret"][:, 0].reshape(b, k)
            feat, last, term, rew, boot = jax.tree.map(
                lambda x: x[:, -k:], (feat, last, term, rew, boot)
            )
            inp = self.feat2tensor(feat)
            los, _, mets = repl_loss(
                last,
                term,
                rew,
                boot,
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.valnorm,
                update=training,
                horizon=self.horizon,
                slowtar=self.repl_slowtar,
                lam=self.repl_lam,
                slowreg=self.repl_slowreg,
            )
            losses.update(los)

        scales = dict(self.loss_scales)
        assert set(losses.keys()) == set(scales.keys()), (
            sorted(losses.keys()),
            sorted(scales.keys()),
        )
        metrics.update({f"loss/{key}": v.mean() for key, v in losses.items()})
        loss = sum([v.mean() * scales[key] for key, v in losses.items()])

        carry = (enc_carry, dyn_carry, dec_carry)
        entries = ({}, dyn_entries, {})
        out = {"losses": losses}
        return loss, (carry, entries, out, metrics)

    def train_loss(self, carry, data, training=True):
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        loss, (carry, entries, _, metrics) = self.loss(carry, obs, prevact, training)
        out = {}
        if self.replay_context:
            out["replay"] = elements.tree.flatdict(
                {"stepid": stepid, "enc": entries[0], "dyn": entries[1], "dec": entries[2]}
            )
        carry = (*carry, {key: data[key][:, -1] for key in self.act_keys})
        return loss, (carry, out, metrics)

    def _apply_replay_context(self, carry, data):
        enc_carry, dyn_carry, dec_carry, prevact = carry
        carry = (enc_carry, dyn_carry, dec_carry)
        stepid = data["stepid"]
        obs = {k: data[k] for k in (*self.img_keys, "is_first", "is_last", "is_terminal", "reward")}

        def prepend(x, y):
            return jnp.concatenate([x[:, None], y[:, :-1]], 1)

        prevact = {k: prepend(prevact[k], data[k]) for k in self.act_keys}
        if not self.replay_context:
            return carry, obs, prevact, stepid

        c = self.replay_context
        nested = elements.tree.nestdict(data)
        entries = [nested.get(k, {}) for k in ("enc", "dyn", "dec")]

        def lhs(xs):
            return jax.tree.map(lambda x: x[:, :c], xs)

        def rhs(xs):
            return jax.tree.map(lambda x: x[:, c:], xs)

        rep_carry = (
            {},
            self.dyn.truncate(lhs(entries[1]), dyn_carry),
            {},
        )
        rep_obs = {k: rhs(obs[k]) for k in obs}
        rep_prevact = {k: data[k][:, c - 1 : -1] for k in self.act_keys}
        rep_stepid = rhs(stepid)

        first_chunk = data["consec"][:, 0] == 0
        carry, obs, prevact, stepid = jax.tree.map(
            lambda normal, replay: nn_where(first_chunk, replay, normal),
            (carry, rhs(obs), rhs(prevact), rhs(stepid)),
            (rep_carry, rep_obs, rep_prevact, rep_stepid),
        )
        return carry, obs, prevact, stepid


# ---------------------------------------------------------------------------
# Actor-critic losses (verbatim pure port of dreamerv3.agent).
# ---------------------------------------------------------------------------


def imag_loss(
    act,
    rew,
    con,
    policy,
    value,
    slowvalue,
    retnorm,
    valnorm,
    advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
    losses = {}
    metrics = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    disc = 1 if contdisc else 1 - 1 / horizon
    weight = jnp.cumprod(disc * con, 1) / disc
    last = jnp.zeros_like(con)
    term = 1 - con
    ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

    roffset, rscale = retnorm(ret, update)
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale
    logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
    ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
    policy_loss = sg(weight[:, :-1]) * -(logpi * sg(adv_normed) + actent * sum(ents.values()))
    losses["policy"] = policy_loss

    voffset, vscale = valnorm(ret, update)
    tar_normed = (ret - voffset) / vscale
    tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
    losses["value"] = (
        sg(weight[:, :-1])
        * (value.loss(sg(tar_padded)) + slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]
    )

    ret_normed = (ret - roffset) / rscale
    metrics["adv"] = adv.mean()
    metrics["rew"] = rew.mean()
    metrics["con"] = con.mean()
    metrics["ret"] = ret_normed.mean()
    metrics["val"] = val.mean()
    metrics["weight"] = weight.mean()
    for k in act:
        metrics[f"ent/{k}"] = ents[k].mean()

    out = {"ret": ret}
    return losses, out, metrics


def repl_loss(
    last,
    term,
    rew,
    boot,
    value,
    slowvalue,
    valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
    losses = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    disc = 1 - 1 / horizon
    weight = f32(~last)
    ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

    voffset, vscale = valnorm(ret, update)
    ret_normed = (ret - voffset) / vscale
    ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
    losses["repval"] = (
        weight[:, :-1]
        * (value.loss(sg(ret_padded)) + slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]
    )

    out = {"ret": ret}
    metrics = {}
    return losses, out, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
    chex.assert_equal_shape((last, term, rew, val, boot))
    rets = [boot[:, -1]]
    live = (1 - f32(term))[:, 1:] * disc
    cont = (1 - f32(last))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return jnp.stack(list(reversed(rets))[:-1], 1)


# ---------------------------------------------------------------------------
# Model construction from a (dreamer) config + obs/act spaces.
# ---------------------------------------------------------------------------


def build_model(config, obs_space, act_space):
    exclude = ("is_first", "is_last", "is_terminal", "reward")
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    img_keys = tuple(sorted(k for k, s in enc_space.items() if len(s.shape) == 3))
    s0 = enc_space[img_keys[0]]
    img_res = (int(s0.shape[0]), int(s0.shape[1]))
    img_channels = tuple(int(enc_space[k].shape[-1]) for k in img_keys)

    r, e, d = config.dyn.rssm, config.enc.simple, config.dec.simple
    enc = Encoder(
        img_keys=img_keys,
        depth=int(e.depth),
        mults=tuple(e.mults),
        kernel=int(e.kernel),
        act=e.act,
        norm=e.norm,
    )
    dyn = RSSM(
        deter=int(r.deter),
        hidden=int(r.hidden),
        stoch=int(r.stoch),
        classes=int(r.classes),
        norm=r.norm,
        act=r.act,
        unimix=float(r.unimix),
        outscale=float(r.outscale),
        imglayers=int(r.imglayers),
        obslayers=int(r.obslayers),
        dynlayers=int(r.dynlayers),
        blocks=int(r.blocks),
        free_nats=float(r.free_nats),
    )
    dec = Decoder(
        img_keys=img_keys,
        img_res=img_res,
        img_channels=img_channels,
        units=int(d.units),
        depth=int(d.depth),
        mults=tuple(d.mults),
        kernel=int(d.kernel),
        bspace=int(d.bspace),
        act=d.act,
        norm=d.norm,
        outscale=float(d.outscale),
    )

    rh, ch, vh, ph = config.rewhead, config.conhead, config.value, config.policy
    rew = MLPHead(
        int(rh.layers),
        int(rh.units),
        rh.act,
        rh.norm,
        HeadSpec("symexp_twohot", (), bins=int(rh.bins), outscale=float(rh.outscale)),
    )
    con = MLPHead(
        int(ch.layers),
        int(ch.units),
        ch.act,
        ch.norm,
        HeadSpec("binary", (), outscale=float(ch.outscale)),
    )
    val = MLPHead(
        int(vh.layers),
        int(vh.units),
        vh.act,
        vh.norm,
        HeadSpec("symexp_twohot", (), bins=int(vh.bins), outscale=float(vh.outscale)),
    )
    slowval = MLPHead(
        int(vh.layers),
        int(vh.units),
        vh.act,
        vh.norm,
        HeadSpec("symexp_twohot", (), bins=int(vh.bins), outscale=float(vh.outscale)),
    )

    act_keys = tuple(sorted(act_space.keys()))
    act_classes = tuple(int(np.asarray(act_space[k].classes).max()) for k in act_keys)
    specs = tuple(
        HeadSpec(
            config.policy_dist_disc,
            (),
            classes=c,
            unimix=float(ph.unimix),
            minstd=float(ph.minstd),
            maxstd=float(ph.maxstd),
            outscale=float(ph.outscale),
        )
        for c in act_classes
    )
    pol = DictMLPHead(int(ph.layers), int(ph.units), ph.act, ph.norm, keys=act_keys, specs=specs)

    rn, vn, an = config.retnorm, config.valnorm, config.advnorm
    retnorm = Normalize(
        rn.impl,
        rate=float(rn.rate),
        limit=float(rn.limit),
        perclo=float(rn.perclo),
        perchi=float(rn.perchi),
        debias=bool(rn.debias),
    )
    valnorm = Normalize(vn.impl, rate=float(vn.rate), limit=float(vn.limit))
    advnorm = Normalize(an.impl, rate=float(an.rate), limit=float(an.limit))

    scales = dict(config.loss_scales)
    rec = scales.pop("rec")
    for k in img_keys:
        scales[k] = rec
    loss_scales = tuple(sorted((k, float(v)) for k, v in scales.items()))

    il, rl = config.imag_loss, config.repl_loss
    return DreamerModel(
        enc=enc,
        dyn=dyn,
        dec=dec,
        rew=rew,
        con=con,
        pol=pol,
        val=val,
        slowval=slowval,
        retnorm=retnorm,
        valnorm=valnorm,
        advnorm=advnorm,
        act_keys=act_keys,
        act_classes=act_classes,
        img_keys=img_keys,
        contdisc=bool(config.contdisc),
        horizon=int(config.horizon),
        imag_length=int(config.imag_length),
        imag_last=int(config.imag_last),
        ac_grads=bool(config.ac_grads),
        reward_grad=bool(config.reward_grad),
        repval_loss=bool(config.repval_loss),
        repval_grad=bool(config.repval_grad),
        replay_context=int(config.replay_context),
        loss_scales=loss_scales,
        imag_lam=float(il.lam),
        imag_actent=float(il.actent),
        imag_slowreg=float(il.slowreg),
        imag_slowtar=bool(il.slowtar),
        repl_lam=float(rl.lam),
        repl_slowreg=float(rl.slowreg),
        repl_slowtar=bool(rl.slowtar),
    )


def ext_space(config, obs_space, act_space):
    """Extra buffer columns (replay-context latents + ids). Mirrors the original
    Agent.ext_space used by BufferTrajectoryDreamer."""
    spaces = {}
    spaces["consec"] = elements.Space(np.int32)
    spaces["stepid"] = elements.Space(np.uint8, 20)
    if int(config.replay_context):
        r = config.dyn.rssm
        spaces["dyn/deter"] = elements.Space(np.float32, int(r.deter))
        spaces["dyn/stoch"] = elements.Space(np.float32, (int(r.stoch), int(r.classes)))
    return spaces


# ---------------------------------------------------------------------------
# Train state + agent contract.
# ---------------------------------------------------------------------------


class TrainState(struct.PyTreeNode):
    model: DreamerModel = struct.field(pytree_node=False)
    tx: Any = struct.field(pytree_node=False)
    policy_fn: Callable = struct.field(pytree_node=False)
    apply_fn: Callable = struct.field(pytree_node=False)
    key: jax.Array
    step: int | jax.Array
    params: Any = struct.field(pytree_node=True)
    stats: Any = struct.field(pytree_node=True)
    opt_state: Any = struct.field(pytree_node=True)
    carry: Any = struct.field(pytree_node=True)
    carry_train: Any = struct.field(pytree_node=True)

    def get_key(self):
        in_key, out_key = jax.random.split(self.key)
        return self.replace(key=in_key), out_key

    def update_state(self, params, carry, carry_train, **kwargs):
        # Kept for collector compatibility (params/carry sync from controller).
        return self.replace(params=params, carry=carry, carry_train=carry_train, **kwargs)

    def update_after_train(self, params, stats, opt_state, carry_train):
        return self.replace(
            step=self.step + 1,
            params=params,
            stats=stats,
            opt_state=opt_state,
            carry_train=carry_train,
        )

    @classmethod
    def create(
        cls, *, model, tx, policy_fn, apply_fn, key, params, stats, opt_state, carry, carry_train
    ):
        return cls(
            model=model,
            tx=tx,
            policy_fn=policy_fn,
            apply_fn=apply_fn,
            key=key,
            step=0,
            params=params,
            stats=stats,
            opt_state=opt_state,
            carry=carry,
            carry_train=carry_train,
        )


def _ema_slow(params, rate):
    slow = optax.incremental_update(params["val"], params["slowval"], rate)
    return {**params, "slowval": slow}


def _make_policy_fn(model):
    @jax.jit
    def policy_fn(params, carry, obs, rng):
        return model.apply(
            {"params": params}, carry, obs, rngs={"sample": rng}, method=DreamerModel.policy
        )

    return policy_fn


def _make_train_fn(model, tx, slow_rate):
    @jax.jit
    def train_fn(params, stats, opt_state, carry_train, data, rng):
        def loss_fn(p):
            (loss, (carry, out, mets)), new_vars = model.apply(
                {"params": p, "stats": stats},
                carry_train,
                data,
                rngs={"sample": rng},
                method=DreamerModel.train_loss,
                mutable=["stats"],
            )
            return loss, (carry, out, mets, new_vars.get("stats", stats))

        (loss, (carry, out, mets, new_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = _ema_slow(params, slow_rate)
        mets = {**mets, "loss": loss, "grad_norm": optax.global_norm(grads)}
        return params, new_stats, opt_state, carry, out, mets

    return train_fn


def create_train_state(
    rng,
    network,
    args_network,
    optimizer,
    args_optimizer,
    obs_space,
    act_space,
    batch_size,
    batch_length,
    slow_rate,
):
    obs_space = dict(obs_space)
    act_space = dict(act_space)
    model = network(args_network, obs_space, act_space)

    init_rng, state_rng = jax.random.split(rng)
    params_rng, sample_rng = jax.random.split(init_rng)
    batch = 1
    t = int(batch_length) + int(args_network.replay_context)
    carry = model.apply({}, batch, method=DreamerModel.initial_carry)
    data = _dummy_data(args_network, obs_space, act_space, batch, t)
    variables = model.init(
        {"params": params_rng, "sample": sample_rng},
        carry,
        data,
        method=DreamerModel.train_loss,
    )
    params = dict(variables["params"])
    stats = variables.get("stats", {})
    params["slowval"] = jax.tree.map(lambda x: x, params["val"])

    tx = optimizer(**dict(args_optimizer))
    opt_state = tx.init(params)

    policy_fn = _make_policy_fn(model)
    train_fn = _make_train_fn(model, tx, float(slow_rate))
    carry_train = model.apply({}, int(batch_size), method=DreamerModel.initial_carry)

    return TrainState.create(
        model=model,
        tx=tx,
        policy_fn=policy_fn,
        apply_fn=train_fn,
        key=state_rng,
        params=params,
        stats=stats,
        opt_state=opt_state,
        carry=carry,
        carry_train=carry_train,
    )


def _dummy_data(args_network, obs_space, act_space, batch, t):
    spaces = dict(**obs_space, **act_space, **ext_space(args_network, obs_space, act_space))
    data = {}
    for k, v in spaces.items():
        data[k] = jnp.zeros((batch, t, *v.shape), _np_dtype(v.dtype))
    return data


def _np_dtype(dtype):
    return jnp.dtype(dtype)


def restore_dreamer_flax_state(
    *,
    rng,
    network,
    args_network,
    optimizer,
    args_optimizer,
    observation_shape,
    actions_shape,
    batch_size,
    batch_length,
    slow_rate,
    checkpointdir,
):
    state = create_train_state(
        rng,
        network,
        args_network,
        optimizer,
        args_optimizer,
        observation_shape,
        actions_shape,
        batch_size,
        batch_length,
        slow_rate,
    )
    if checkpointdir is None:
        return state
    orbax_checkpointer = ocp.StandardCheckpointer()
    abstract = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    return orbax_checkpointer.restore(checkpointdir, abstract)


def get_select_action_fn(flax_state: TrainState):
    def select_action(flax_state: TrainState, observation: chex.ArrayTree):
        flax_state, seed = flax_state.get_key()
        carry, acts, out = flax_state.policy_fn(
            flax_state.params, flax_state.carry, observation, seed
        )
        flax_state = flax_state.update_state(flax_state.params, carry, flax_state.carry_train)
        acts, out = take_outs((acts, out))
        return flax_state, acts, out

    return select_action


def get_update_step(apply_fn=None, config=None) -> Callable:
    def _update_step(
        train_state: TrainState, buffer_sample: TrajectoryBufferSample
    ) -> tuple[TrainState, dict, dict | None]:
        data = buffer_sample.experience
        train_state, seed = train_state.get_key()
        params, stats, opt_state, carry_train, out, mets = train_state.apply_fn(
            train_state.params,
            train_state.stats,
            train_state.opt_state,
            train_state.carry_train,
            data,
            seed,
        )
        train_state = train_state.update_after_train(params, stats, opt_state, carry_train)
        return train_state, mets, out.get("replay")

    return _update_step


def get_update_epoch(update_step_fn: Callable, buffer_lock, buffer) -> Callable:
    def _update_epoch(train_state: TrainState, samples: list):
        info = {}
        for buffer_sample, batch_indices, time_indices in samples:
            train_state, info, replay_updates = update_step_fn(train_state, buffer_sample)
            if replay_updates is not None and buffer is not None:
                replay_updates, batch_indices, time_indices = jax.device_get(
                    (replay_updates, batch_indices, time_indices)
                )
                with buffer_lock.write():
                    buffer.update(batch_indices, time_indices, replay_updates)
        return train_state, info

    return _update_epoch
