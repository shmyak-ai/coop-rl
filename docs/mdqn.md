# Munchausen DQN (M-DQN) in coop-rl

This document explains the Munchausen DQN algorithm and reports the result of comparing
the coop-rl implementation (`src/coop_rl/agents/mdqn.py`, `src/coop_rl/base/loss.py`, and
the configs `mdqn_atari.py` / `mdqn_rec_atari.py`) against the original paper:
**"Munchausen Reinforcement Learning"**, Vieillard, Pietquin, Geist, NeurIPS 2020
([arXiv:2007.14430](https://arxiv.org/abs/2007.14430)).

**Verdict:** the core Munchausen loss (`base/loss.py:munchausen_q_learning`) is a faithful
implementation of the paper's Eq. (7), and all three Munchausen-specific hyperparameters
match the paper exactly (τ = 0.03, α = 0.9, l₀ = −1). The configs deviate from the paper
in several deliberate ways (n-step returns in the feedforward config, soft target updates,
L2 instead of Huber, no reward clipping, deep residual torsos) — catalogued in
[Differences from the paper](#differences-from-the-paper). Three correctness issues found
in the original audit (n-step target misalignment, Munchausen bonus only at the window's
first step, one TD error per recurrent sequence) have since been fixed; see
[Known caveats](#known-caveats) for what remains.

## The idea in one paragraph

Every TD-based agent bootstraps on its value estimate. Munchausen RL asks: the agent also
maintains a second estimate of "what's good" — its **policy** — so why not bootstrap on
that too? The recipe is a one-line change: **add the scaled log-policy of the action taken
to the immediate reward**,

```
r_t  →  r_t + α · τ · ln π(a_t | s_t)
```

Since `ln π ≤ 0`, this is a *penalty* everywhere except where the policy is confident —
effectively a bonus for actions the current policy already believes in. The agent "pulls
itself up by its own hair", like Baron Munchausen out of the swamp — hence the name.
Applied to DQN, this tiny modification (M-DQN) was the first non-distributional agent to
beat C51 on Atari; applied to IQN (M-IQN) it beat Rainbow — with no n-step returns, no
prioritized replay, no distributional machinery.

## From DQN to M-DQN in three steps

**1. DQN.** The classic regression target for a transition `(s_t, a_t, r_t, s_{t+1})`:

```
q̂_dqn = r_t + γ · max_a' q_target(s_{t+1}, a')
```

**2. Soft-DQN.** DQN's policy is deterministic, so `ln π` is ill-defined. First make it
stochastic via maximum-entropy RL: the policy becomes a softmax over Q-values with
temperature τ, `π = softmax(q_target / τ)`, and the hard max becomes a *soft* max:

```
q̂_s-dqn = r_t + γ · Σ_a' π(a'|s_{t+1}) · [ q_target(s_{t+1}, a') − τ ln π(a'|s_{t+1}) ]
```

The bracket is the soft state value: expected Q under the policy *plus* its entropy. A
convenient identity collapses the whole sum into one numerically-friendly expression:

```
Σ_a π(a|s) [q(s,a) − τ ln π(a|s)]  =  τ · logsumexp(q(s,·) / τ)
```

As τ → 0 this reduces back to `max_a q`, i.e. plain DQN.

**3. M-DQN.** Now add the Munchausen term — the scaled log-policy of the action actually
taken, weighted by α ∈ [0, 1] (paper Eq. 2):

```
q̂_m-dqn = r_t + α·τ·ln π(a_t|s_t) + γ · τ · logsumexp(q_target(s_{t+1}, ·) / τ)
```

with `π = softmax(q_target / τ)` computed from the **target** network in both places.
The loss is the (Huber, in the paper) regression of `q_online(s_t, a_t)` onto this target.

**Log-policy clipping.** `ln π(a|s)` is unbounded below — if the policy gets near-greedy,
the term for a rarely-taken action can explode. The paper clips it to `[l₀, 0]` with
l₀ = −1, i.e. the Munchausen term is `α · clip(τ ln π(a_t|s_t), l₀, 0)`.

**Numerical stability.** `τ ln π = q − τ·ln Σ exp(q/τ)` is unstable for small τ, so the
paper computes it with a temperature-aware log-sum-exp trick (subtract `max_a q` first).
`jax.nn.log_softmax(q / τ)` and `jax.nn.logsumexp(q / τ)` do exactly this internally, so
the coop-rl implementation gets the same stability for free.

## Why it works

The paper's Section 3 rewrites M-DQN as an abstract value-iteration scheme, M-VI(α, τ),
and shows the log-policy reward bonus is **implicit KL regularization** between successive
policies: each greedy step actually solves

```
π_{k+1} = argmax_π  ⟨π, q_k⟩ − α·τ·KL(π ‖ π_k) + (1−α)·τ·H(π)
```

Two consequences:

- **Errors average instead of accumulating.** With α = 1, the performance bound depends on
  `‖(1/k) Σ_j ε_j‖` — the norm of the *average* approximation error — instead of the
  discounted *sum of norms* that standard approximate VI suffers. Successive errors can
  cancel out, and the bound scales as `(1−γ)⁻¹` rather than `(1−γ)⁻²`. The paper argues
  this is the strongest known bound of its kind for a practical deep-RL agent.
- **Bigger action gap.** In the τ → 0 limit, M-DQN becomes Advantage Learning:
  `r + α(q(s,a) − max q(s,·)) + γ max q(s',·)`. Both increase the gap between the best and
  second-best action values by a quantifiable amount, which makes the greedy policy more
  robust to estimation errors.

So the "one-line trick" is secretly a principled trust-region method (in the spirit of
Conservative Value Iteration / Dynamic Policy Programming), delivered without any explicit
extra machinery.

## Hyperparameters (paper values)

| Parameter | Value | Notes |
|---|---|---|
| τ (entropy temperature) | 0.03 | scales the softmax policy and soft value |
| α (Munchausen coefficient) | 0.9 | α = 0 recovers Soft-DQN |
| l₀ (log-policy clip) | −1 | Munchausen term clipped to [l₀, 0] |
| γ | 0.99 | |
| n-step | **1** | the paper uses no n-step returns for M-DQN |
| loss | Huber | |
| optimizer | Adam, lr 5e-5 | changed from DQN's RMSProp — not anodyne, ablated in the paper |
| target update | hard copy every 8000 grad steps | |
| ε-greedy | 0.01, linear decay over 250k steps | |
| rewards | clipped to [−1, 1] | standard Dopamine/Machado ALE protocol |

**Why ε-greedy, not sampling the softmax policy?** M-DQN produces a genuinely stochastic
policy, but the paper still acts ε-greedily (Appendix B.2): at initialization Q-values are
near zero, so `softmax(q/0.03)` is close to uniform, and on sparse-reward games (Venture,
Enduro) a near-uniform policy never finds reward to learn from. ε-greedy exploration is
invariant to the scale of Q-values and does not have this failure mode. coop-rl follows
the paper: `DiscreteQNetworkHead` returns an `EpsilonGreedy` distribution which the
collectors sample.

## How coop-rl implements it

### File map

| What | Where |
|---|---|
| Munchausen loss (paper Eq. 7) | `src/coop_rl/base/loss.py` — `munchausen_q_learning` |
| Feedforward update step (n-step assembly + grad) | `src/coop_rl/agents/mdqn.py` — `get_update_step` |
| Recurrent (R2D2-style) update step | `mdqn.py` — `get_update_step_recurrent`, `get_recurrent_rollout` |
| Action selection (ε-greedy, batched / recurrent) | `mdqn.py` — `get_select_action_batch_fn`, `get_select_action_recurrent_batch_fn` |
| Soft target-network update | `mdqn.py` — `TrainState.apply_gradients` (`optax.incremental_update`) |
| Feedforward Atari config | `src/coop_rl/configs/mdqn_atari.py` |
| Recurrent Atari config | `src/coop_rl/configs/mdqn_rec_atari.py` |

### The loss

`munchausen_q_learning` maps line-for-line onto the math above:

```python
# Munchausen term: α · clip(τ ln π(a_t|s_t), l0, 0), π = softmax(q_target/τ)
munchausen_term = entropy_temperature * jax.nn.log_softmax(q_tm1_target / entropy_temperature)
munchausen_term_a = jnp.clip(sum(action_one_hot * munchausen_term), clip_value_min, 0.0)

# Soft bootstrap: τ · logsumexp(q_target(s_{t+1})/τ) — the soft state value
next_v = entropy_temperature * jax.nn.logsumexp(q_t_target / entropy_temperature)

target_q = stop_gradient(r_t + munchausen_coefficient * munchausen_term_a + d_t * next_v)
```

`q_tm1_target` and `q_t_target` both come from the target network, matching the paper's
use of `π_θ̄` and `q_θ̄`. The TD error is squared (L2) by default; setting
`huber_loss_parameter > 0` switches to Huber as in the paper.

### The update step (feedforward)

`get_update_step` receives a sampled trajectory window of length
`sample_sequence_length` (5 in `mdqn_atari.py`, i.e. a 4-step return) and builds one
n-step Munchausen target per window. With the storage convention "index t holds
`(o_t, a_t, r_t, term_t)` where `r_t` was produced by `a_t` at `o_t`", the window is cut
at the first done transition, index n (or the last transition if none):

- **Reward sum**: `Σ_{i<n} γ^i r̃_i`, where each intermediate reward carries its own
  Munchausen bonus, `r̃_i = r_i + α·clip(τ ln π(a_i|o_i), l₀, 0)` — M-RL is reward
  shaping, so with n-step returns every reward in the window is shaped, not just the
  first (step 0's bonus is added inside `munchausen_q_learning`).
- **Bootstrap**: `γⁿ · τ·logsumexp(q_target(o_n)/τ)`, discounted by γⁿ, not γ. The cut
  transition's own reward `r_n` is *excluded* from the sum because it is part of the
  bootstrap value.
- **Termination**: if the cut transition terminated the episode, `r̃_n` *is* included
  (it is the final reward) and the bootstrap is dropped. A truncated episode still
  bootstraps from its last stored observation.

Targets are handed to `munchausen_q_learning`, and `TrainState.apply_gradients` performs
the Adam step plus a Polyak (τ = 0.005) target update. The same alignment fix is applied
to the window assembly in `agents/dqn.py` and `agents/rainbow.py`.

### The recurrent variant

`mdqn_rec_atari.py` trades DQN's 4-frame stacking for a GRU (`stack_size = 1`,
`hidden_state_dim = 512`). Its visual encoder is a BTR-style Impala ResNet
(`VisualResNetTorso`: 2× width, LayerNorm, 6×6 adaptive maxpool — per *Beyond The
Rainbow*, Clark et al. 2025) with a 512-unit Dense bridge into the GRU; see
`docs/miqn.md` for the encoder details. R2D2-style: the buffer stores the per-step hidden state, and
each sampled 30-step sequence is split into a 10-step **burn-in** (hidden state warmed up
from the stored value under stop-gradient, separately for online and target networks) and
a 20-step **learn** window (`get_recurrent_rollout`). The loss
(`munchausen_q_learning_n_step` in `base/loss.py`) computes an **n-step Munchausen TD
error at every learn step** that has n successor steps inside the window (R2D2-style
dense targets; `n_steps` is configurable, default 3): `q_online[:, t]` regresses onto

```
Σ_{i<n} γ^i·r̃_{t+i} + γ^n·τ·logsumexp(q_target[:, t+n]/τ)
```

where `r̃ = r + α·clip(τ ln π(a|s), l₀, 0)` is the Munchausen-shaped reward, applied to
every step. An interior termination stops the sum after that step's reward (no
bootstrap); an interior truncation at offset k bootstraps `γᵏ·softV` from the truncated
step; truncated anchors are masked out of the mean. Each 30-step sequence yields
`learn_length − n_steps` TD errors (17 at n = 3). Note n = 1 matches the paper's 1-step
operator exactly; n > 1 is a practical R2D2-style extension — the paper's
KL-regularization theory is derived for 1-step, and n-step replay targets carry the
usual uncorrected off-policy bias (benign for small n).

### Differences from the paper

Deliberate or benign deviations in the coop-rl configs:

| coop-rl | Paper | Impact |
|---|---|---|
| n-step returns (4-step ff; configurable `n_steps` = 3 rec) | 1-step only | the paper's headline result avoids n-step; here every reward in the window gets its own Munchausen bonus (reward shaping), keeping the n-step target consistent; recurrent `n_steps = 1` recovers the paper's operator |
| L2 loss (`huber_loss_parameter = 0.0`) | Huber | heavier tails on TD errors get more gradient weight |
| `max_abs_reward = 1000` → effectively unclipped | rewards clipped to [−1, 1] | τ = 0.03 and l₀ = −1 were tuned against clipped rewards; with raw-scale rewards the Munchausen bonus (∈ [−0.9, 0]) is relatively much smaller |
| Polyak target update, τ = 0.005 per step | hard copy every 8000 steps | modern smooth-target choice, similar effective timescale |
| Adam lr 6.25e-5, eps 1e-5, grad-clip 0.5 (ff) / 10 (rec, per BTR) | Adam lr 5e-5, no clipping | minor |
| γ = 0.99 (ff) / 0.997 (rec, per BTR) | γ = 0.99 | longer effective horizon in the recurrent config |
| ε = 0.01 constant | ε = 0.01 with 250k-step linear decay | slightly less early exploration |
| ff: CNN + deep residual torso (Wang et al. 2025), SiLU/Swish; rec: Impala ResNet + LayerNorm + 6×6 adaptive maxpool (BTR, Clark et al. 2025); both bfloat16 | Conv×3 → FC 512, ReLU, fp32 | deliberate modernization; loss math runs in fp32 (`astype(jnp.float32)` on Q-values) |

### Known caveats

Three issues found in the original audit — misaligned n-step targets (last reward
double-counted, bootstrap discounted γ instead of γⁿ), the Munchausen bonus applied only
to the window's first step, and the recurrent variant emitting a single TD error per
sequence — have been fixed as described above (the n-step alignment fix is also ported
to `agents/dqn.py` and `agents/rainbow.py`). What remains:

1. **Truncation at the window's first step** (feedforward path) degenerates: with no
   rewards to accumulate the target is just `softV(o_0)` plus the Munchausen term — a
   self-regression rather than a Bellman backup. This affects at most one window per
   truncation (windows overlap with `period = 1`) and truncations are rare on Atari, so
   it is left as is.

### Composing with other DQN extensions

M-DQN changes the *target*, not the architecture or the replay scheme, so most classic
DQN improvements compose with it — but not all of them are worth it:

| Extension | Compatible? | Status in coop-rl |
|---|---|---|
| **Dueling** (Q = V + A − mean A) | Yes — pure head change, the loss never looks inside; DM-DQN (Gu et al. 2022) found it converges faster than both M-DQN and Dueling DQN | **Enabled**: both mdqn configs use `DuelingQNetworkHead` (`networks/dueling.py`) |
| **n-step returns** | Yes — Munchausen is reward shaping, so every reward in the window gets its own bonus | Enabled (see above) |
| **Double Q** | Not applicable — the bootstrap is `τ·logsumexp(q_target/τ)` (a soft max, no argmax to decouple), and the Munchausen penalty is already conservative | — |
| **Prioritized replay** | Yes, but needs a prioritized buffer + importance-sampling weights in the loss; the paper beat C51 with uniform replay | **Enabled in M-IQN** (`BufferPrioritised`); mdqn stays uniform |
| **Noisy nets** | Possible (`NoisyLinear` exists for Rainbow), but the noise leaks into the shaping policy `π = softmax(q_target/τ)`, and M-DQN's entropy term already aids exploration | Not recommended |
| **Distributional (C51/IQN)** | Yes — the paper's own M-IQN | **Implemented**: `agents/miqn.py`, see `docs/miqn.md` |

The dueling head keeps the `EpsilonGreedy`/`.preferences` interface of
`DiscreteQNetworkHead`, so it is a drop-in config swap: no change to the loss or update
steps, and it works for both the feedforward `(batch, emb)` and recurrent
`(time, batch, emb)` paths (no leading-dim reshapes).

## References

- Vieillard, Pietquin, Geist. *Munchausen Reinforcement Learning.* NeurIPS 2020.
  [arXiv:2007.14430](https://arxiv.org/abs/2007.14430)
- Kapturowski et al. *Recurrent Experience Replay in Distributed Reinforcement Learning*
  (R2D2). ICLR 2019 — burn-in scheme used by the recurrent variant.
- Wang et al. *1000 Layer Networks for Self-Supervised RL.* NeurIPS 2025 — deep residual
  torso used by the feedforward config.
- Clark, Towers, Evers, Hare. *Beyond The Rainbow: High Performance Deep Reinforcement
  Learning on a Desktop PC.* ICML 2025. [arXiv:2411.03820](https://arxiv.org/abs/2411.03820)
  — Impala encoder and hyperparameters used by the recurrent config.
- Gu, Zhu, Lv, Shi, Hou, Xu. *DM-DQN: Dueling Munchausen deep Q network for robot path
  planning.* Complex & Intelligent Systems, 2022 — dueling + Munchausen combination.
