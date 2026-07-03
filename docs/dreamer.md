# DreamerV3 in coop-rl

This document describes the DreamerV3 implementation in coop-rl and reports the results of
a line-by-line comparison of its logic and math against the reference implementation at
`~/src/dreamerv3` (Hafner's ninjax/JAX codebase), from which it was ported onto flax/optax
in June 2026.

**Verdict:** the port is mathematically faithful. No genuine differences in algorithm logic
or math were found in the exercised (Atari, image-observation, discrete-action) path. Two
**hyperparameter values** differ from the reference configuration (learning rate and actor
entropy scale — see [Differences found](#differences-found)), and one latent bug in the
reference was fixed in passing. Everything else is either an equivalent refactoring
(ninjax → flax state plumbing) or an intentional scope reduction.

## File mapping

| coop-rl (port) | original dreamerv3 |
|---|---|
| `src/coop_rl/networks/rssm.py` — RSSM, EncoderImage, DecoderImage | `dreamerv3/rssm.py` |
| `src/coop_rl/networks/dreamer_nn.py` — NN primitives | `embodied/jax/nets.py` |
| `src/coop_rl/networks/heads.py` — Head/MLPHead/DictMLPHead | `embodied/jax/heads.py` |
| `src/coop_rl/agents/dreamer.py` — model, losses, train step | `dreamerv3/agent.py` + `embodied/jax/agent.py`, `embodied/jax/utils.py` |
| `src/coop_rl/base/distributions.py` — output distributions | `embodied/jax/outs.py` |
| `src/coop_rl/base/utils.py` — `make_dreamer_optimizer` | `embodied/jax/opt.py` + `agent.py:_make_opt` |
| `src/coop_rl/configs/dreamer_atari.py` | `dreamerv3/configs.yaml` (defaults + `atari` + `size1m`) |

## The algorithm

DreamerV3 is a model-based RL agent with three parts trained jointly from replayed
experience: a **world model** that learns a compact latent dynamics model of the
environment, a **critic** that estimates returns, and an **actor** that is trained
entirely inside the world model's imagination — the policy never receives gradients
from real environment interaction, only from latent rollouts.

All numeric values below are the coop-rl Atari configuration
(`configs/dreamer_atari.py`, the `size1m` preset).

### World model (RSSM)

The Recurrent State-Space Model (`networks/rssm.py`) maintains a latent state with two
parts:

- a **deterministic** recurrent state `h_t` (`deter`, 512 units), and
- a **stochastic** state `z_t` (`stoch`): 32 independent categorical variables with 4
  classes each, stored as one-hot vectors.

One observation step (`RSSM.observe_step`) computes:

```
h_t     = f(h_{t-1}, z_{t-1}, a_{t-1})          # recurrent core, _core()
prior:    p(z_t | h_t)                           # _prior(): MLP over h_t → logits
posterior: q(z_t | h_t, x_t)                     # obs MLP over [h_t, enc(x_t)] → logits
z_t     ~ q                                      # one sample per step
feat_t  = (h_t, z_t)                             # features consumed by all heads
```

**Recurrent core** (`rssm.py:166`, `_core`). Not a stock GRU: a block-diagonal gated cell.
The action is first bounded, `a ← a / sg(max(1, |a|))`. Then `h`, `z`, `a` each pass
through their own Linear → Norm → SiLU branch; the concatenation is broadcast to each of
`blocks = 8` groups of the deterministic state and mixed by per-block hidden layers
(`BlockLinear`, block-diagonal weight matrices, so groups don't interact except through
the broadcast input). A final `BlockLinear` producing `3 × deter` units is split into
GRU-style gates:

```
reset  = sigmoid(r)
cand   = tanh(reset * c)
update = sigmoid(u - 1)          # -1 bias initializes the gate mostly closed
h_t    = update * cand + (1 - update) * h_{t-1}
```

**Stochastic state.** Prior and posterior logits are reshaped to `(32, 4)` and wrapped in
a `OneHot` distribution with **1% unimix** (`probs = 0.99 · softmax(logits) + 0.01 / K`),
which keeps every class probability strictly positive so log-probs and KLs stay bounded.
Sampling uses the **straight-through** estimator (`distributions.py:237`): the forward
value is the sampled one-hot, the gradient flows through the softmax probabilities.

**Episode boundaries.** `is_first` masks the carry and previous action to zeros before the
step, so each episode starts from the learned initial state. Sequences are processed with
`nn.scan` over time (per-step sampling RNG split, shared params), matching the original's
`nj.scan`.

**KL loss** (`rssm.py:148`). The KL between posterior and prior is split into two terms
with separate stop-gradients — this is DreamerV3's "KL balancing":

```
dyn = KL( sg(q)  ‖  p )      # trains the prior (dynamics) to predict the posterior
rep = KL(  q  ‖  sg(p) )     # trains the posterior (representation) toward the prior
dyn, rep ← max(dyn, 1.0)     # free bits: no gradient once below 1 nat
```

`dyn` has loss scale 1.0, `rep` scale 0.1: dynamics adapt to representations much more
than representations bend toward what the dynamics can predict. Free bits prevent the
posterior from collapsing into the prior when reconstruction is easy.

### Encoder and decoder

The **encoder** (`rssm.py`, image path) normalizes frames to `x/255 − 0.5` and applies a
stack of Conv2D (kernel 5, stride 1) + space-to-depth max-pooling (each stage halves the
resolution), with RMSNorm + SiLU after each stage; channel widths are
`depth · mults = 4 · (2, 3, 4, 4)`. The flattened output tokens feed the posterior.

The **decoder** maps `(h, z)` back to an image. The deterministic state goes through a
`BlockLinear` (block space `bspace = 8`) reshaped into a spatial map; the stochastic state
through a two-layer MLP; their sum (after Norm + SiLU) is upsampled by nearest-neighbor ×2
+ Conv2D stages back to the frame resolution, ending in a sigmoid. The reconstruction
loss is a plain MSE against `obs/255`, summed over pixels (`Agg(MSE(·), 3, sum)`).

### symlog, symexp, and twohot returns

Rewards and returns in different environments span orders of magnitude. DreamerV3 handles
this without per-task tuning through two devices:

```
symlog(x) = sign(x) · log(1 + |x|)        symexp(x) = sign(x) · (e^|x| − 1)
```

Reward and value heads are **symexp-twohot** heads: a Linear layer produces logits over
255 bins whose positions are `symexp(linspace(−20, 20))` — exponentially spaced, covering
roughly `±4.8 × 10^8` with fine resolution near zero. A scalar target `y` is encoded as a
**twohot** vector: all mass on the two neighboring bins, split in proportion to proximity,
and the loss is the cross-entropy against that vector (`distributions.py:278`,
`TwoHot.loss` — computed with `logsumexp` for stability). The predicted scalar is the
bin-weighted sum, evaluated as a symmetric sum from the outside in
(`TwoHot.pred`, `distributions.py:252`) so the large outer bins cancel numerically.

This makes reward/value learning a well-conditioned classification problem regardless of
the return scale. Both heads use `outscale = 0.0` — zero-initialized output layers, so
predictions start at exactly 0.

### Imagination

Actor and critic train on latent rollouts (`agents/dreamer.py:192`, `_imagine`). Start
states are the posterior latents of replayed steps (`K = batch_length` starts per
sequence). From each start, the model rolls forward `imag_length = 15` steps:

```
a_t   ~ π( sg(feat_t) )        # policy acts on stop-gradient features
h,z   = imagine_step(carry, a_t)   # prior only — no observations in imagination
```

The stop-gradient on the policy input means actor gradients flow through the REINFORCE
term only, not through the dynamics. The imagined features themselves are also
stop-gradient for the actor/critic losses (`ac_grads = False`): world model and
actor-critic are decoupled through `sg`.

The continuation head supplies per-step discounting: with `contdisc = True`, the
continuation target is `(1 − is_terminal) · (1 − 1/horizon)`, i.e. the discount
`γ = 1 − 1/333 ≈ 0.997` is folded into the continuation probability, and the imagination
weight is `weight_t = ∏_{s≤t} con_s` (`imag_loss`, `dreamer.py:392`).

### Critic

The critic (`value` head, 3-layer MLP, symexp-twohot, 255 bins) is trained on
**λ-returns** computed over the imagined trajectory (`lambda_return`,
`dreamer.py:465`):

```
R_T = v(feat_T)                                       # bootstrap
R_t = r_{t+1} + γ·con_{t+1} · [ (1−λ)·v_{t+1} + λ·R_{t+1} ],    λ = 0.95
```

Value predictions are un-normalized through `valnorm` before use (`impl = "none"` in this
config, so valnorm is inert but the plumbing matches the original). The critic loss
(`dreamer.py:409`) is the twohot loss against `sg(R_t)` **plus** a slow-critic
regularizer:

```
L_value = value.loss(sg(R)) + slowreg · value.loss(sg(slowvalue.pred())),   slowreg = 1.0
```

The **slow critic** (`slowval`) is an EMA copy of the critic, updated after every
optimizer step with rate 0.02 (`optax.incremental_update`, `dreamer.py:694` — equivalent
to the original's `SlowModel.update`). With `slowtar = False` the λ-return targets use
the online critic; the slow copy only anchors the regularizer.

### Actor

The actor loss (`imag_loss`, `dreamer.py:366`) is REINFORCE with normalized advantages
and an entropy bonus:

```
(roffset, rscale) = retnorm(R)                    # percentile return normalization
adv_t  = (R_t − v_t) / rscale
L_policy = weight_t · −( log π(a_t) · sg(adv_t) + actent · H[π_t] )
```

`retnorm` (`Normalize`, `dreamer.py:66`) tracks EMA estimates (rate 0.01) of the 5th and
95th percentiles of the return batch and normalizes by
`rscale = max(limit, perc95 − perc5)` with `limit = 1.0` — returns are scaled down when
they span a large range but never scaled up, which is what makes a single `actent` work
across environments. `advnorm` is `impl = "none"` in this config (advantages pass
through unchanged; the offset/scale plumbing exists to match the original). The policy
head is categorical with 1% unimix and `outscale = 0.01`.

### Replay-value loss

In addition to the imagination critic loss, the critic is trained on **replayed** (real)
trajectories (`repl_loss`, `dreamer.py:428`, loss scale `repval = 0.3`): λ-returns over
the replay sequence using real rewards and real terminals, bootstrapped at the sequence
edge by the imagination return of the corresponding start state
(`boot = imgloss_out["ret"][:, 0]`, `dreamer.py:275`). This grounds the value function in
real data while imagination provides lookahead.

### World-model loss (total)

Per replayed batch (16 sequences × 64 steps), the total loss is the scale-weighted sum:

| term | loss | scale |
|---|---|---|
| `rec` (image) | MSE(decoder, obs/255), sum over pixels | 1.0 |
| `rew` | symexp-twohot cross-entropy | 1.0 |
| `con` | Bernoulli log-likelihood | 1.0 |
| `dyn` | `max(KL(sg(q)‖p), 1)` | 1.0 |
| `rep` | `max(KL(q‖sg(p)), 1)` | 0.1 |
| `policy` | REINFORCE + entropy (imagination) | 1.0 |
| `value` | twohot + slow regularizer (imagination) | 1.0 |
| `repval` | twohot + slow regularizer (replay) | 0.3 |

Everything trains in a single joint gradient step; the `sg` boundaries above define which
loss reaches which parameters.

### Replay-latent warm start

Replay stores, alongside observations, the RSSM latents (`dyn/deter`, `dyn/stoch`) and a
`consec` counter per step (`ext_space`, `dreamer.py:627`). When a sampled sequence is a
continuation (`consec[:, 0] != 0`), the model warm-starts from the stored latents via
`dyn.truncate` instead of the learned initial state; the one-step `replay_context = 1`
prefix supplies the previous action (`_apply_replay_context`, `dreamer.py:320`). Fresh
latents are written back to the buffer after each train step.

### Optimizer

`make_dreamer_optimizer` (`base/utils.py:127`) chains, in order:

1. **AGC** — adaptive gradient clipping: per-parameter, scale gradients so
   `‖g‖ ≤ 0.3 · max(10⁻³, ‖θ‖)`;
2. **RMS** — second-moment normalization, `β₂ = 0.999`, `ε = 10⁻²⁰`, bias-corrected;
3. **momentum** — first moment, `β₁ = 0.9`, bias-corrected;
4. **learning rate** — constant schedule with 1000-step linear warmup.

Weight decay is 0. Compute dtype is bfloat16; parameters and normalizer statistics are
float32. All weight matrices use truncated-normal fan-in initialization
(`Initializer("trunc_normal", fan="in")`, the original's `trunc_normal_in`).

### Acting and data flow

The policy path (`DreamerModel.policy`) runs one `observe_step` per environment step and
**samples** from the policy head (the original's unused `mode` argument had no effect —
it also always sampled). Discrete actions are one-hot embedded by the agent
(`_embed_action`, `dreamer.py:166`) before entering the RSSM, so the RSSM never sees the
action space — an intentional structural change (keeps RSSM fields hashable for jit) that
is net-equivalent to the original's in-RSSM `DictConcat` embedding plus reset masking.

The original's `embodied` driver/replay is replaced by coop-rl infrastructure: collector
workers step gymnasium Atari environments (`HandlerEnvDreamerAtari`, 96×96 grayscale,
sticky actions) and push transitions to a flashbax trajectory buffer; sampler threads
draw `16 × 64`-step batches; a single trainer runs the jitted train step and applies the
slow-critic EMA. This is orchestration, not algorithm — the per-batch math is identical.

## Differences found

### Genuine differences

1. **Learning rate — 4× higher than the reference config.**
   coop-rl `configs/dreamer_atari.py:160` sets `lr = 4e-5`; the original's effective
   Atari value is `1e-5` (`configs.yaml:90`, `opt.lr`, not overridden by the `atari` or
   `size1m` blocks). The port kept the code-level default of `_make_opt` (4e-5, the
   DreamerV3 paper value) instead of the repo's YAML override. Load-bearing for training
   parity.

2. **Actor entropy scale — ~3.3× lower than the reference config.**
   coop-rl `configs/dreamer_atari.py:117` sets `actent = 3e-4`; the original's effective
   value is `1e-3` (`configs.yaml:111`). Same pattern: the Python function default
   (`imag_loss`, 3e-4 on both sides) was kept where the original's YAML overrides it.
   Weaker exploration pressure.

   Both mismatches are config values, not math. To reproduce the reference checkout's
   behavior exactly, set `lr = 1e-5` and `actent = 1e-3`.

3. **`Binary.sample` — latent bug fixed in the port (coop-rl is correct).**
   Original `embodied/jax/outs.py:205` calls
   `jax.random.bernoulli(seed, prob, -1, shape + ...)` — a spurious `-1` positional
   (copied from `Categorical.sample`'s axis argument) that would misbind `shape`.
   coop-rl `base/distributions.py:175` drops it. The path is never exercised in training
   (only `pred`/`logp` of the continuation head are used), so this changes nothing in
   practice.

### Verified equivalent (math identical)

- **RSSM**: `_core` block-GRU arithmetic, prior/posterior MLPs, unimix `OneHot`
  distribution, KL split + free bits, `initial`/`truncate`/`starts` — byte-for-byte.
- **Losses**: `imag_loss`, `repl_loss`, `lambda_return` are verbatim ports, including
  the `valnorm.stats()` un-normalization, `cumprod` imagination weights, twohot target
  padding, and the slow-critic regularizer.
- **Imagination restructuring**: the rollout loop moved from `RSSM.imagine` into
  `DreamerModel._imagine` (flax `nn.scan`), but `sg(carry)` before the policy, start
  selection, horizon, and feature/action stitching are all preserved — traced
  step-by-step to produce identical tensors.
- **Slow critic**: `SlowModel` replaced by `optax.incremental_update` after the optimizer
  step; identical formula, rate, init-copy, and update position. The slow copy lives in
  `params` but only appears under `sg` in losses, so its gradient (and update) is exactly
  zero before the EMA overwrite.
- **Normalizers**: `Normalize` (percentile/meanstd EMA with debias) identical; the only
  removal is the cross-host `pmean`/`all_gather`, which is a no-op on a single device.
- **Distributions** (`MSE`, `Huber`, `Normal`, `Binary`, `Categorical`, `OneHot`,
  `TwoHot`, `Agg`) and **optimizer** (AGC → RMS → momentum → lr, all coefficients):
  character-identical apart from the `Binary.sample` fix above.
- **Initialization**: both default to truncated-normal fan-in with the 1.1368 correction
  factor; conv/linear/norm math (including the manual transposed conv and RMSNorm
  `eps = 1e-4`) identical.
- **Replay context / is_first handling / `ext_space` columns**: identical, including the
  first-chunk selection and previous-action prepend.
- **Config values**: everything else matches the `defaults + atari + size1m` merge —
  network sizes, all loss scales, `horizon = 333`, `imag_length = 15`, `λ = 0.95`,
  unimix 0.01, free nats 1.0, retnorm/valnorm/advnorm settings, slow rate 0.02, all
  optimizer coefficients, batch 16 × 64, `replay_context = 1`.

### Intentional scope and scale choices (not bugs)

- **Model size**: `size1m` (deter 512, units 64, depth 4) instead of the original repo's
  default ~200M preset (deter 8192, units 1024). Architecture identical, capacity smaller.
- **Image + discrete only**: the vector-observation encoder/decoder branch, continuous
  actions (`bounded_normal` is ported but unused), the `huber`/`normal_logstd` head
  impls, and the `swiglu` activation were not ported.
- **Unused library modules not ported**: `Attention`, `Transformer`, `GRU`, `Embed`,
  `DictEmbed` exist in the original `embodied/jax/nets.py` but have **zero references
  anywhere in the original repo** — `embodied/jax` is a general-purpose NN library, and
  DreamerV3 itself never uses them (the RSSM's recurrence is its own inline block-GRU in
  `_core`, which the port reproduces). Omitting them loses nothing.
- **Training scale**: replay capacity 300k transitions vs 1e6; 3M vs 51M total steps.
- **Environment**: gymnasium `ALE/Pong-v5` (6 actions) vs the original `elements` Atari
  wrapper with the full legal set (18); 96×96 grayscale and sticky actions match.
- **Reporting**: the original's `report()` pass (open-loop video predictions, gradient
  norms) and several logging-only metrics were dropped — no training effect.
- **Infrastructure**: the `embodied.jax.Agent` wrapper (device meshes, FSDP sharding,
  prefetch, param sync) is replaced by coop-rl's single-device `TrainState` + jitted
  `policy_fn`/`train_fn` + Ray/thread workers.

### Edge-case caveats (benign under the current config)

- **Decoder image-key order**: the port sorts image keys in both encoder and decoder;
  the original decoder splits output channels in dict-insertion order. Identical for the
  single-image-key Atari path; could mis-assign channels only with multiple image keys
  in non-sorted order.
- **Slow-critic cadence**: the original has `slowvalue.every` (update every N steps);
  the port hardcodes every-step updates. The reference config uses `every: 1`, so inert.
- **Init/bias passthrough**: the original threads a `**kw` (winit/binit/bias) from config
  into every sublayer; the port uses the defaults directly. Equivalent because the
  reference config's values equal those defaults (`trunc_normal_in`, zeros, bias on);
  would diverge only if a config customized them.
- **`Frozen`/`Concat`** helper wrappers from `outs.py` were not ported (unused by the
  ported heads).
