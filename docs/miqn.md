# Munchausen IQN (M-IQN) in coop-rl

M-IQN is the distributional variant of Munchausen RL from the same paper as M-DQN
(Vieillard, Pietquin, Geist, *Munchausen Reinforcement Learning*, NeurIPS 2020,
Appendix B.1). It applies the Munchausen recipe — add the scaled, clipped log-policy
to the reward and soften the bootstrap — to IQN (Dabney et al. 2018) instead of DQN.
In the paper it outperformed Rainbow and set the then state of the art on Atari.
Read `docs/mdqn.md` first for the Munchausen idea; this document covers what changes
when the q-function becomes a quantile distribution, and which DQN improvements this
implementation layers on top.

## From M-DQN to M-IQN

IQN does not estimate `q(s, a)` directly. It estimates the quantile function
`z_σ(s, a)` of the return distribution at fractions `σ ∈ [0, 1]`, sampled anew each
call; the q-value is recovered as the Monte-Carlo mean `q̃(s, a) = E_σ[z_σ(s, a)]`.
The paper's M-IQN TD error, for an online sample `σ` and a target sample `σ'`:

```
TD = r + α·clip(τ ln π(a|s), l₀, 0)
       + γ·Σ_a' π(a'|s')·(z_σ'(s', a') − τ ln π(a'|s'))
       − z_σ(s, a)

π(·|s) = softmax(q̃(s, ·)/τ),  q̃ from the target network's quantile mean
```

Two things differ from M-DQN:

1. **No logsumexp shortcut.** In M-DQN the soft bootstrap collapses to
   `τ·logsumexp(q/τ)`. Here the bootstrap is per-target-quantile, so the explicit
   policy expectation `Σ_a' π·(z − τ ln π)` is kept. When all quantiles are equal the
   two forms coincide (this identity is checked in the numerical tests).
2. **Quantile Huber regression instead of a scalar TD loss.** With N online samples
   `σ_i` and N' target samples `σ'_j`, the loss is the IQN pairwise quantile Huber:

```
ρ_σ(δ) = |σ − 1{δ<0}| · huber_κ(δ) / κ
L = Σ_i (1/N') Σ_j ρ_{σ_i}(TD_ij)
```

Acting is ε-greedy over `q̃` estimated from K fresh quantile samples, exactly as in
the paper (M-IQN keeps ε-greedy despite the naturally stochastic softmax policy).

## Improvements included (and excluded)

| Extension | In this implementation | Why |
|---|---|---|
| n-step returns (n = 3) | ✓ | the paper's headline M-IQN uses 3-step returns; the Munchausen term is applied only at the anchor step s₀, matching BTR and BY571 (an earlier version shaped every window reward and collapsed a healthy policy — see [Known caveats](#known-caveats)) |
| Dueling | ✓ | per-quantile value/advantage streams, `z = v + a − mean(a)`; validated for Munchausen by DM-DQN (Gu et al. 2022) |
| Prioritized replay | ✓ (feedforward only) | `BufferPrioritised` + importance-sampling weights, priorities = per-sequence quantile-Huber loss (same machinery as rainbow); the recurrent variant deliberately uses uniform replay (see below) |
| Double Q | ✗ | inapplicable — the bootstrap is a softmax-policy expectation, there is no argmax to decouple |
| Noisy nets | ✗ (by571, rec); ✓ (`miqn_atari`, btr) | BTR's published agent (Clark et al. 2025, arXiv:2411.03820) combines Munchausen RL, IQN, and NoisyNets, validated across the full Atari-60 suite (IQM 7.4, Table 1 / Appendix C.2); `miqn_atari.py` and `miqn_btr_atari.py` match that combination — see the BTR variant section. The paper's own vanilla M-IQN uses plain ε-greedy |

## The network

`QuantileFeedForwardNetwork` (`networks/base.py`) is `FeedForwardNetwork` plus a
`num_quantiles` call argument forwarded to the head. `DuelingQuantileQNetworkHead`
(`networks/quantile.py`) implements the IQN parametrization:

1. sample `σ ~ U[0,1]` per batch element (`num_quantiles` of them) via the
   `"quantiles"` RNG stream (mirrors rainbow's `"noise"` stream);
2. cosine embedding `cos(π·i·σ), i = 1..n_cos` → Dense → ReLU;
3. Hadamard product with the torso's state embedding;
4. dueling streams → `z (…, N, A)`; `q̃ = mean_N(z)` → `EpsilonGreedy`.

`QuantileQNetworkHead` (same file) is the plain non-dueling variant — steps 1–3
identical, then MLP → Dense(A) directly — used by `miqn_by571_atari.py` for BY571
parity.

The head returns `(EpsilonGreedy, z, σ)` and is shape-agnostic over leading batch
dims, so the same apply serves acting `(B, …)` and the window pass `(B, L, …)`.

## The update step

`get_update_step` in `agents/miqn.py` follows the corrected n-step window assembly
shared with `mdqn.py`/`dqn.py`/`rainbow.py` (first-done cut, cut-reward exclusion,
`γⁿ` bootstrap discount, truncation still bootstraps). Munchausen-specific parts:

- the shaping policy comes from the **target network's** `q̃` at the anchor
  observation `s₀` only (K = `num_quantile_samples` fractions, quantile mean taken in
  f32 so bf16 quantization does not distort the τ = 0.03 softmax); the Munchausen term
  is added inside the loss from `q̃_target(s₀)`, and intermediate n-step rewards are
  **not** shaped — matching BTR and BY571, whose n-step targets carry the term only
  at s_t (see [Known caveats](#known-caveats) for the collapse this convention
  prevents, and for the deliberate target-net-vs-online-net difference from BTR);
- the loss `munchausen_quantile_q_learning` (`base/loss.py`) draws N =
  `num_tau_samples` online fractions at `s₀` and N' = `num_tau_prime_samples` target
  fractions at `s_n`, builds the per-target-quantile Munchausen target, and returns
  the per-sequence quantile-Huber loss vector;
- PER: importance weights `(1/p)^β / max` with β linearly annealed 0.5 → 1.0;
  new priorities are the per-sequence loss (+1e-5), written back through
  `get_update_epoch`'s `buffer.set_priorities`.

## The BTR variant (`miqn_atari.py`, `miqn_btr_atari.py`)

Both `miqn_atari.py` and `miqn_btr_atari.py` re-wire the same feedforward agent along
the lines of **Beyond The Rainbow** (BTR, Clark et al.,
[arXiv:2411.03820](https://arxiv.org/abs/2411.03820)), whose ablations attribute the
single largest gain (+142% IQM) to the encoder. BTR's own published agent combines
Munchausen RL, IQN, and NoisyNets together (Table 1 / Appendix C.2), validated across
the full Atari-60 suite (IQM 7.4) — exactly the combination these two configs use.
Almost all of `agents/miqn.py` is shared — the additions are an opt-in hard-copy target
update (see below) and an always-available `"noise"` rng stream for NoisyNets, both
inert for `miqn_by571_atari.py`/`miqn_rec_atari.py` (Polyak stays the default there;
plain heads never call `self.make_rng("noise")`, so the extra stream is simply unused):

- **Impala ResNet encoder** (`VisualResNetTorso`, `networks/resnet.py`): BTR's 2× width
  (32-64-64 channels, 2 residual blocks per group, conv+maxpool downsampling), ReLU;
- **LayerNorm instead of BTR's spectral norm** — the BTR authors themselves note
  (footnote) that LayerNorm, found after completion, is the better normalizer; it also
  avoids threading a mutable power-iteration state through every `apply_fn` call;
- **6×6 adaptive maxpool** (`adaptive_max_pool`, PyTorch semantics) before the flatten
  → a 2304-dim IQN embedding, then 512-unit **noisy** dueling streams (see below).
  Total 5,265,834 parameters — hand-computing BTR's own `networks.py` with its actual
  default (`--noisy 1`, `model_size=2`, `linear_size=512`, `n_cos=64`, 4 actions for
  Breakout; conv bias=True everywhere; spectral norm adds no trainable parameters;
  `FactorizedNoisyLinear` stores a full sigma matrix per weight/bias exactly like this
  repo's `NoisyLinear`) gives 5,264,554 — a 1,280-parameter difference, fully explained
  by this repo's LayerNorm γ/β terms replacing BTR's spectral norm (which adds none);
- **NoisyNets** (`NoisyDuelingQuantileQNetworkHead`, `networks/quantile.py`), matching
  BTR's actual default recipe (`--noisy 1`) rather than pure ε-greedy: the dueling
  value/advantage streams use `NoisyMLPTorso`/`NoisyLinear` (Fortunato et al. 2018,
  already used by `configs/rainbow_atari.py`) instead of `MLPTorso`/`nn.Dense`; the IQN
  cosine-embedding fusion Dense stays plain, matching BTR's own `networks.py` (its
  `cos_embedding` is a bare `nn.Linear`, never wrapped in its noisy/plain switch). A
  `"noise"` rng stream is threaded through `create_train_state`,
  `get_select_action_batch_fn`, and `get_update_step` in `agents/miqn.py`, mirroring
  `agents/rainbow.py`'s existing convention exactly. `sigma_zero = 0.5` (the
  Fortunato-paper/class default, not `rainbow_atari.py`'s tuned `0.25`).
  `miqn_by571_atari.py`/`miqn_rec_atari.py` exclude NoisyNets, matching the M-IQN
  paper's own plain ε-greedy;
- **Exploration schedule**: BTR combines NoisyNets with an **annealed ε-greedy for the
  first half of training**, then disables it (`ε = 0` from `env_frames // 2` onward),
  leaving noisy weights as the only exploration source. BTR's `EpsilonGreedy` subtracts
  `(ε − 0.01) / eps_steps` per stored transition (`eps_steps = 2M`) — an exponential-gap
  decay `ε(k) = 0.01 + 0.99·exp(−k/2e6)`, ~0.37 at 2M, ~0.14 at 4M, ~0.03 at 8M frames.
  `miqn_btr_atari.py` and `miqn_atari.py` both reproduce it with the same plain Python
  schedule in `args_get_select_action_fn.epsilon_scheduler_fn`; in `miqn_atari.py`'s
  async setup there is no single global frame counter (each of the `num_collectors`
  parallel `CollectorDQNUniform` Ray actors anneals independently over its own local
  frame count), so the boundaries are per-collector approximations there;
- **Trainer/collector wiring is the one deliberate difference between the two configs**:
  `miqn_btr_atari.py` uses `TrainerSequential` (run with `--backend sequential`), a
  synchronous single-thread loop reproducing BTR's own exact interleaving — one
  `learn()` call per 64 collected transitions (`replay_ratio = 1/64`, matching BTR's
  one `learn()` per 64-env step batch; with 32 envs and `steps_per_rollout = 1` that
  is one update per two rollouts, paid through the trainer's fractional
  `update_debt`). `miqn_atari.py` instead keeps coop-rl's async architecture:
  a `Trainer` actor and `num_collectors` separate `CollectorDQNUniform` Ray actors
  running independently, decoupled via a `Controller`, exactly like every other
  non-BTR feedforward config — only the network/hyperparameters/exploration recipe
  changed, not the training-loop coupling;
- **Hard target-network copy every 500 steps**, matching BTR exactly. `TrainState` in
  `agents/miqn.py` gained an opt-in `target_update_period` field
  (`args_state_recover.target_update_period = 500`): when `> 0` it hard-copies the
  online params into the target every `target_update_period` steps via
  `optax.periodic_update`, instead of Polyak-blending via `tau` every step. Every
  other miqn config leaves it at the default `0` and keeps the original Polyak path;
- **BTR hyperparameters**: γ = 0.997, PER α = 0.2, grad-clip 10, batch 256, lr 1e-4
  (matches BTR — same batch size, so there's no reason left to keep the repo's
  lower Dopamine-convention lr here), Adam ε = `0.005 / batch_size` ≈ 1.953e-5 (BTR
  hardcodes this formula, not a flag; the repo's other miqn configs keep the default
  1e-5). PER importance-sampling β is a **constant 0.45**, not annealed — BTR's own
  `--per_beta_anneal` flag defaults to off, so `self.per_beta` never leaves its
  hardcoded 0.45 initial value. Learning starts only after **200,000 stored
  transitions** (BTR's hardcoded `min_sampling_size`; flashbax `min_length = 6250`
  per time axis × 32 add rows). BTR's own N = N' = K = 8 quantile-sample counts are
  **not** used — an earlier attempt at matching them exactly reproduced a flat,
  state-independent-Q failure mode on Breakout (confirmed by an actual training run:
  mean return rose to ~3 then flatlined near 0 for 100k+ updates), so this config uses
  paper-parity N = N' = 64, K = 32 instead, like `miqn_rec_atari.py`. This combination
  (paper-parity quantile counts + NoisyNets + hard-copy targets) hasn't itself been
  validated by BTR's own ablations — BTR's Table 1 / Appendix C.2 result used N=8;
- **32 parallel envs** instead of BTR's default 64 — halved alongside the batch size
  to fit the Impala encoder's 84×84 activations on 8GB GPUs, same rationale as the
  batch-256 note above.

## The recurrent variant

`miqn_rec_atari.py` is the R2D2-style counterpart, structured exactly like the recurrent
MDQN (GRU instead of frame stacking, per-step hidden states in the buffer, 10-step
stop-gradient burn-in + 20-step learn window, configurable `n_steps = 3`). It uses the
same BTR-style Impala encoder and hyperparameter alignment as `miqn_btr_atari.py`
(γ = 0.997, grad-clip 10) and the same paper-parity quantile sample counts
(N = N' = 64, K = 32 — BTR's own N = N' = K = 8 trains flat for M-IQN in both the
recurrent and feedforward variants, see the BTR variant section), with
`hidden_sizes = (512,)` as a Dense
bridge into the GRU — feeding the raw 2304-dim flatten into GRU(512) would ~3× the RNN
input parameters for no BTR-grounded reason. The GRU output then passes through a
post-torso `DeepResidualTorso` (width 256, depth 8, Swish — Wang et al. 2025) before the
dueling quantile head, same as `mdqn_rec_atari.py`. Per learn
window, `get_recurrent_rollout` (`agents/miqn.py`) unrolls the online network with N
quantile samples and the target network with N' samples; the loss
(`munchausen_quantile_q_learning_n_step` in `base/loss.py`) builds an n-step Munchausen
target **per target quantile** with the same backward recursion as the scalar
`munchausen_q_learning_n_step` — the bootstrap is the per-quantile soft value
`soft_z_j = Σ_a π(a|s)·(z_j(s,a) − τ ln π(a|s))` instead of `τ·logsumexp` — and applies
the pairwise quantile Huber at every anchor (`learn_length − n_steps` per sequence,
truncated anchors masked).

Two deliberate simplifications versus the feedforward variant:

- **Uniform replay, no PER.** There is no prioritized *sequence* buffer in the repo
  (`BufferTrajectoryDQNRecurrent` is uniform), so PER here would mean building one plus
  R2D2's mixed max/mean priority scheme, and it would force the update epoch out of jit
  for host-side priority write-backs. Uniform replay reuses the existing recurrent
  buffer/collector unchanged and keeps the whole epoch jitted (the config imports the
  plain jitted `get_update_epoch` from `agents/mdqn`).
- **The shaping policy comes free.** The rollout already computes target-network z for
  every learn step, so the Munchausen policy is just its quantile mean — no separate
  K-sample policy pass like the feedforward window assembly needs.

## The BY571 variant (sequential backend)

`miqn_by571_atari.py` replicates, as closely as the repo allows, the M-IQN Breakout
demo from [BY571/IQN-and-Extensions](https://github.com/BY571/IQN-and-Extensions)
(`-frames 500000 -eps_frames 75000 -min_eps 0.025 -lr 1e-4 -t 5e-3 -m 15000 -N 32`).
Run it with the **sequential backend**:

```
coop-rl-train --config miqn_by571_atari --backend sequential
```

`TrainerSequential` (`workers/trainers.py`) is a single-thread synchronous loop —
collect a rollout, then do `n_transitions × replay_ratio` gradient updates, repeat —
restoring the classic DQN coupling between environment frames and updates that the
async `ray`/`thread` backends deliberately decouple. Config knobs: `env_frames`
(stop condition, in collected transitions) and `replay_ratio` (1.0 = BY571's one
update per transition; 0.25 would express classic DQN's train-every-4-frames).
The network reproduces BY571's `-agent iqn` architecture exactly (parameter-count
identical: 1,890,020): Nature-DQN convs (ReLU, `padding = "VALID"` — flax's SAME
default would yield 11×11×64 = 7744 features instead of PyTorch's 7×7×64 = 3136)
→ flatten (3136) with **no residual tail** (`CNNTorso` with `depth = 0`) →
quantile cosine fusion at the raw conv features → one 512 FC → plain **non-dueling**
`QuantileQNetworkHead`, all float32. Other exact matches: Adam eps 1e-8
(`args_optimizer.adam_eps`; other configs keep coop-rl's 1e-5 default), learning
starts at 33 stored transitions (`min_length = 33`), single env,
N = N' = K = 32 (BY571 collapses all three), 1-step returns
(`sample_sequence_length = 2`), uniform replay via `priority_exponent = 0.0`,
unclipped rewards, batch 32, buffer 15000, lr 1e-4, grad-clip 1.0, soft target
updates τ = 0.005 every step, and a linearly annealed epsilon (1.0 → 0.025 over
75k frames) via `epsilon_scheduler_fn` in the collector's action selection.
Weight init matches PyTorch's layer default, `kaiming_uniform(a=√5)` =
`U(±1/√fan_in)`, expressed as `variance_scaling(1/3, "fan_in", "uniform")` and
passed as `kernel_init` to both torso and head (`CNNTorso` now forwards
`kernel_init` to its convs; its default stays flax's `lecun_normal`, so other
configs are unaffected). The env matches BY571's wrapper stack too:
`noop_max = 0` (no noop starts) and `fire_on_reset = True` (a `FireOnReset`
wrapper in `base/environment.py` presses FIRE then action 2 after every reset,
like the classic `FireResetEnv`). Remaining (minor) deviations: flax zero-init
biases vs PyTorch's `U(±1/√fan_in)` biases, and "evaluation" being the
training-policy rolling mean return rather than a separate near-greedy eval.

## Known caveats

**Intermediate-shaping collapse (fixed).** An earlier version of `get_update_step`
added the Munchausen term to every intermediate reward of the 3-step window on top of
the anchor-step application. Checkpoint forensics on a Breakout run showed the failure
directly: the greedy policy (ε = 0.01 eval) climbed to a mean return of 9.3 by 20k
updates, then collapsed to 0.9 by 30k while mean Q dove −2.2 → −9.0 → −12.9 —
impossible under clipped rewards in [0, 1] and only explicable by the extra negative
shaping (each intermediate term is `α·clip(τ·ln π(a|s), −1, 0)`, and buffer actions
are mostly random early, so the penalties compound without the fixed-point
compensation the anchor-step term gets through the bootstrap). BTR and BY571 shape
only s_t; the implementation now matches — the feedforward window assembly, the
recurrent `munchausen_quantile_q_learning_n_step` recursion (only the final unroll
level, the anchor's own reward, carries the term), and the mdqn counterparts (see
`docs/mdqn.md`, Known caveats).

**Resuming from a checkpoint.** `--orbax-checkpoint-dir` restores the `TrainState`
only (params, target params, optimizer state, update-step counter, rng). The replay
buffer and PER priorities, the local frames counter, and the ε-schedule counter (a
Python closure in `get_select_action_batch_fn`) all restart from zero — a resumed run
re-anneals ε from 1.0 and re-warms the buffer before its first update. Judge training
health by a near-greedy eval on checkpoints, not the ε-mixed training return.

**Remaining divergences from BTR** (deliberate; roughly in ablation-priority order if
a run misbehaves):

1. **Munchausen term from the target net.** BTR computes `τ·ln π` from the *online*
   net; this repo uses the target net, matching the M-DQN paper and BY571. A one-line
   change in `get_update_step` — the first ablation to try.
2. **bf16 network compute** (BTR is fp32 end-to-end). Only the shaping-sensitive path
   — the quantile mean feeding the τ = 0.03 softmax — is forced to f32; TD errors and
   priorities still come from bf16 activations. A full-fp32 run (batch 128 on 8GB
   GPUs) is the fallback if the flat-Q mode reappears.
3. **Unvalidated combination.** Paper-parity quantile counts (N = N' = 64, K = 32)
   + NoisyNets + hard-copy targets + Impala encoder is a combination no reference has
   run: BTR's own result used N = 8 (which trains flat here), and the paper's M-IQN
   used no NoisyNets.
4. **Sticky actions off** (`repeat_action_probability = 0.0`; BTR defaults to 0.25).
   The env is easier, so returns are not comparable to BTR's published numbers.
5. **PER priority metric**: priorities are the quantile-Huber loss value; BTR uses
   |TD| summed over online quantiles, averaged over target quantiles. A different
   sampling distribution, judged low-impact.
6. **Update budget**: `replay_ratio = 1/64` at `env_frames = 32M` yields ~500k
   gradient steps vs BTR's ~781k over 200M frames. Raise `env_frames` if a healthy
   run is still improving at the end.

**Other caveats:**

- **Truncation at the window's first step degenerates** (feedforward path, shared
  with mdqn — see `docs/mdqn.md`, Known caveats): the target reduces to a
  self-regression on the soft value of `o_0`. At most one window per truncation,
  rare on Atari, left as is.
- **Noise leaks into the shaping policy**: the target-net pass at s₀ runs with
  sampled NoisyNet noise, so `π = softmax(q̃/τ)` — and hence the Munchausen term —
  is stochastic. BTR combines NoisyNets and Munchausen the same way, so this is
  validated in practice.
- **Host RAM**: the 1M-transition buffer grows to ~28GB of uint8 observations;
  swapping silently craters updates/s.

## Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| τ (entropy temperature) | 0.03 | paper Table 2 |
| α (Munchausen coefficient) | 0.9 | paper Table 2 |
| l₀ (log-policy clip) | −1 | paper Table 2 |
| κ (quantile Huber) | 1.0 | IQN |
| N, N' (loss quantile samples) | 64, 64 (`miqn_atari`, btr, rec); 32, 32 (by571) | IQN / Dopamine; BY571 `-N 32`; BTR's own N=N'=8 trains flat for M-IQN, see the BTR variant section |
| K (acting / shaping-policy samples) | 32 (all configs) | IQN / Dopamine; BTR's own K=8 trains flat for M-IQN |
| n_cos (cosine embedding size) | 64 | IQN |
| n-step | 3 (`sample_sequence_length = 4`) | paper's M-IQN |
| γ | 0.99 (by571); 0.997 (`miqn_atari`, btr, rec) | paper; BTR |
| PER priority exponent / IS β | 0.6 / 0.5→1.0 (by571, rec); 0.2 / constant 0.45 (`miqn_atari`, btr) | rainbow config; BTR (`per_beta_anneal` defaults off, so β never anneals) |
| Rewards | clipped to [−1, 1] (`max_abs_reward = 1.0`) | paper Table 2 — matches `mdqn`'s Atari configs |
| Batch size | 256 (`miqn_atari`, btr) | BTR value; also fits 8GB GPUs — the Impala encoder's 84×84 activations OOM at 512 |
| Target update | Polyak τ = 0.005 (by571, rec); hard copy every 500 steps (`miqn_atari`, btr) | repo convention (paper: hard copy every 8000); BTR |
| Optimizer | Adam 1e-4 eps 0.005/batch_size≈1.953e-5, grad-clip 10 (`miqn_atari`, btr); grad-clip 10 (rec) | repo convention (paper: Adam 5e-5); BTR |
| Exploration | ε-greedy, fixed (rec); ε-greedy, annealed (by571); NoisyNets + ε-greedy annealed then disabled at `env_frames // 2` — exponential-gap `ε(k) = 0.01 + 0.99·exp(−k/2e6)`, BTR's exact recurrence (`miqn_atari`, btr) | paper's M-IQN uses ε-greedy; BTR's own default (`--noisy 1`) |
| Replay warm-up / ratio | 200k transitions, 1 update per 64 transitions (btr) | BTR `min_sampling_size`; BTR one `learn()` per 64-env step |

## File map

| What | Where |
|---|---|
| M-IQN quantile-Huber losses | `src/coop_rl/base/loss.py` — `munchausen_quantile_q_learning`, `munchausen_quantile_q_learning_n_step` |
| Update steps / epoch, action selection, TrainState | `src/coop_rl/agents/miqn.py` |
| Implicit quantile dueling head (plain and `Noisy*`, `miqn_atari`/btr) | `src/coop_rl/networks/quantile.py` |
| Network wrappers with `num_quantiles` arg | `src/coop_rl/networks/base.py` — `QuantileFeedForwardNetwork`, `QuantileRecurrentNetwork` |
| Impala encoder + adaptive maxpool (BTR) | `src/coop_rl/networks/resnet.py` — `VisualResNetTorso`, `adaptive_max_pool` |
| Post-GRU deep residual torso (rec only) | `src/coop_rl/networks/torso.py` — `DeepResidualTorso` |
| Atari configs | `src/coop_rl/configs/miqn_atari.py`, `src/coop_rl/configs/miqn_btr_atari.py`, `src/coop_rl/configs/miqn_rec_atari.py`, `src/coop_rl/configs/miqn_by571_atari.py` |
| Sequential (synchronous) trainer | `src/coop_rl/workers/trainers.py` — `TrainerSequential` |

## References

- Vieillard, Pietquin, Geist. *Munchausen Reinforcement Learning.* NeurIPS 2020.
  [arXiv:2007.14430](https://arxiv.org/abs/2007.14430) — M-IQN definition in Appx. B.1.
- Dabney, Ostrovski, Silver, Munos. *Implicit Quantile Networks for Distributional
  Reinforcement Learning.* ICML 2018. [arXiv:1806.06923](https://arxiv.org/abs/1806.06923)
- Clark, Towers, Evers, Hare. *Beyond The Rainbow: High Performance Deep Reinforcement
  Learning on a Desktop PC.* ICML 2025. [arXiv:2411.03820](https://arxiv.org/abs/2411.03820)
  — Impala encoder, adaptive maxpooling, and hyperparameters used by the btr and
  recurrent configs.
- Gu, Zhu, Lv, Shi, Hou, Xu. *DM-DQN: Dueling Munchausen deep Q network for robot path
  planning.* Complex & Intelligent Systems, 2022.
