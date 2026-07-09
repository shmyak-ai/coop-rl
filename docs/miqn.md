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
| n-step returns (n = 3) | ✓ | the paper's headline M-IQN uses 3-step returns; every window reward gets its own Munchausen bonus (reward shaping), as in the mdqn feedforward path |
| Dueling | ✓ | per-quantile value/advantage streams, `z = v + a − mean(a)`; validated for Munchausen by DM-DQN (Gu et al. 2022) |
| Prioritized replay | ✓ (feedforward only) | `BufferPrioritised` + importance-sampling weights, priorities = per-sequence quantile-Huber loss (same machinery as rainbow); the recurrent variant deliberately uses uniform replay (see below) |
| Double Q | ✗ | inapplicable — the bootstrap is a softmax-policy expectation, there is no argmax to decouple |
| Noisy nets | ✗ | the noise would leak into the shaping policy `π = softmax(q̃_target/τ)`; the paper's M-IQN uses ε-greedy |

## The network

`QuantileFeedForwardNetwork` (`networks/base.py`) is `FeedForwardNetwork` plus a
`num_quantiles` call argument forwarded to the head. `DuelingQuantileQNetworkHead`
(`networks/quantile.py`) implements the IQN parametrization:

1. sample `σ ~ U[0,1]` per batch element (`num_quantiles` of them) via the
   `"quantiles"` RNG stream (mirrors rainbow's `"noise"` stream);
2. cosine embedding `cos(π·i·σ), i = 0..n_cos−1` → Dense → ReLU;
3. Hadamard product with the torso's state embedding;
4. dueling streams → `z (…, N, A)`; `q̃ = mean_N(z)` → `EpsilonGreedy`.

The head returns `(EpsilonGreedy, z, σ)` and is shape-agnostic over leading batch
dims, so the same apply serves acting `(B, …)` and the window pass `(B, L, …)`.

## The update step

`get_update_step` in `agents/miqn.py` follows the corrected n-step window assembly
shared with `mdqn.py`/`dqn.py`/`rainbow.py` (first-done cut, cut-reward exclusion,
`γⁿ` bootstrap discount, truncation still bootstraps). Munchausen-specific parts:

- the shaping policy comes from the **target network's** `q̃` over the whole window
  (K = `num_quantile_samples` fractions); intermediate rewards (steps ≥ 1) get their
  bonus in the assembly, step 0's bonus is added inside the loss from `q̃_target(s₀)`;
- the loss `munchausen_quantile_q_learning` (`base/loss.py`) draws N =
  `num_tau_samples` online fractions at `s₀` and N' = `num_tau_prime_samples` target
  fractions at `s_n`, builds the per-target-quantile Munchausen target, and returns
  the per-sequence quantile-Huber loss vector;
- PER: importance weights `(1/p)^β / max` with β linearly annealed 0.5 → 1.0;
  new priorities are the per-sequence loss (+1e-5), written back through
  `get_update_epoch`'s `buffer.set_priorities`.

## The BTR variant

`miqn_btr_atari.py` re-wires the same feedforward agent along the lines of **Beyond The
Rainbow** (BTR, Clark et al., [arXiv:2411.03820](https://arxiv.org/abs/2411.03820)),
whose ablations attribute the single largest gain (+142% IQM) to the encoder. Only the
network and hyperparameters change — `agents/miqn.py` is shared:

- **Impala ResNet encoder** (`VisualResNetTorso`, `networks/resnet.py`): BTR's 2× width
  (32-64-64 channels, 2 residual blocks per group, conv+maxpool downsampling), ReLU;
- **LayerNorm instead of BTR's spectral norm** — the BTR authors themselves note
  (footnote) that LayerNorm, found after completion, is the better normalizer; it also
  avoids threading a mutable power-iteration state through every `apply_fn` call;
- **6×6 adaptive maxpool** (`adaptive_max_pool`, PyTorch semantics) before the flatten
  → a 2304-dim IQN embedding, then 512-unit dueling streams. Total 2.90M parameters
  vs BTR's reported 2.91M;
- **BTR hyperparameters**: γ = 0.997, PER α = 0.2, grad-clip 10, batch 256,
  N = N' = K = 8. Polyak targets (τ = 0.005) and lr 6.25e-5 are kept from the repo
  convention (BTR: hard copies every 500 steps, lr 1e-4 at batch 256).

## The recurrent variant

`miqn_rec_atari.py` is the R2D2-style counterpart, structured exactly like the recurrent
MDQN (GRU instead of frame stacking, per-step hidden states in the buffer, 10-step
stop-gradient burn-in + 20-step learn window, configurable `n_steps = 3`). It uses the
same BTR-style Impala encoder and hyperparameter alignment as `miqn_btr_atari.py`
(γ = 0.997, grad-clip 10, 8 quantile samples), with `hidden_sizes = (512,)` as a Dense
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

## Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| τ (entropy temperature) | 0.03 | paper Table 2 |
| α (Munchausen coefficient) | 0.9 | paper Table 2 |
| l₀ (log-policy clip) | −1 | paper Table 2 |
| κ (quantile Huber) | 1.0 | IQN |
| N, N' (loss quantile samples) | 64, 64 (`miqn_atari`); 8, 8 (btr, rec) | IQN / Dopamine; BTR |
| K (acting / shaping-policy samples) | 32 (`miqn_atari`); 8 (btr, rec) | IQN / Dopamine; BTR |
| n_cos (cosine embedding size) | 64 | IQN |
| n-step | 3 (`sample_sequence_length = 4`) | paper's M-IQN |
| γ | 0.99 (`miqn_atari`); 0.997 (btr, rec) | paper; BTR |
| PER priority exponent / IS β | 0.6 (`miqn_atari`) / 0.2 (btr); 0.5 → 1.0 | rainbow config; BTR |
| Batch size | 512 (`miqn_atari`); 256 (btr) | BTR value; also fits 8GB GPUs — the Impala encoder's 84×84 activations OOM at 512 |
| Target update | Polyak τ = 0.005 | repo convention (paper: hard copy every 8000) |
| Optimizer | Adam 6.25e-5, grad-clip 0.5 (`miqn_atari`) / 10 (btr, rec) | repo convention (paper: Adam 5e-5); BTR clip |

## File map

| What | Where |
|---|---|
| M-IQN quantile-Huber losses | `src/coop_rl/base/loss.py` — `munchausen_quantile_q_learning`, `munchausen_quantile_q_learning_n_step` |
| Update steps / epoch, action selection, TrainState | `src/coop_rl/agents/miqn.py` |
| Implicit quantile dueling head | `src/coop_rl/networks/quantile.py` |
| Network wrappers with `num_quantiles` arg | `src/coop_rl/networks/base.py` — `QuantileFeedForwardNetwork`, `QuantileRecurrentNetwork` |
| Impala encoder + adaptive maxpool (BTR) | `src/coop_rl/networks/resnet.py` — `VisualResNetTorso`, `adaptive_max_pool` |
| Post-GRU deep residual torso (rec only) | `src/coop_rl/networks/torso.py` — `DeepResidualTorso` |
| Atari configs | `src/coop_rl/configs/miqn_atari.py`, `src/coop_rl/configs/miqn_btr_atari.py`, `src/coop_rl/configs/miqn_rec_atari.py` |

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
