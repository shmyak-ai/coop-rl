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
| Prioritized replay | ✓ | `BufferPrioritised` + importance-sampling weights, priorities = per-sequence quantile-Huber loss (same machinery as rainbow) |
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

## Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| τ (entropy temperature) | 0.03 | paper Table 2 |
| α (Munchausen coefficient) | 0.9 | paper Table 2 |
| l₀ (log-policy clip) | −1 | paper Table 2 |
| κ (quantile Huber) | 1.0 | IQN |
| N, N' (loss quantile samples) | 64, 64 | IQN / Dopamine |
| K (acting / shaping-policy samples) | 32 | IQN / Dopamine |
| n_cos (cosine embedding size) | 64 | IQN |
| n-step | 3 (`sample_sequence_length = 4`) | paper's M-IQN |
| PER priority exponent / IS β | 0.6 / 0.5 → 1.0 | rainbow config |
| Target update | Polyak τ = 0.005 | repo convention (paper: hard copy every 8000) |
| Optimizer | Adam 6.25e-5, grad-clip 0.5 | repo convention (paper: Adam 5e-5) |

## File map

| What | Where |
|---|---|
| M-IQN quantile-Huber loss | `src/coop_rl/base/loss.py` — `munchausen_quantile_q_learning` |
| Update step / epoch, action selection, TrainState | `src/coop_rl/agents/miqn.py` |
| Implicit quantile dueling head | `src/coop_rl/networks/quantile.py` |
| Network wrapper with `num_quantiles` arg | `src/coop_rl/networks/base.py` — `QuantileFeedForwardNetwork` |
| Atari config | `src/coop_rl/configs/miqn_atari.py` |

## References

- Vieillard, Pietquin, Geist. *Munchausen Reinforcement Learning.* NeurIPS 2020.
  [arXiv:2007.14430](https://arxiv.org/abs/2007.14430) — M-IQN definition in Appx. B.1.
- Dabney, Ostrovski, Silver, Munos. *Implicit Quantile Networks for Distributional
  Reinforcement Learning.* ICML 2018. [arXiv:1806.06923](https://arxiv.org/abs/1806.06923)
- Gu, Zhu, Lv, Shi, Hou, Xu. *DM-DQN: Dueling Munchausen deep Q network for robot path
  planning.* Complex & Intelligent Systems, 2022.
