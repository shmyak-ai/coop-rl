import chex
import jax
import jax.numpy as jnp
import rlax

# These losses are generally taken from rlax but edited to explicitly take in a batch of data.
# This is because the original rlax losses are not batched and are meant to be used with vmap,
# which is much slower.


def categorical_double_q_learning(
    q_logits_tm1: chex.Array,
    q_atoms_tm1: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_logits_t: chex.Array,
    q_atoms_t: chex.Array,
    q_t_selector: chex.Array,
) -> chex.Array:
    """Computes the categorical double Q-learning loss. Each input is a batch."""
    batch_indices = jnp.arange(a_tm1.shape[0])
    # Scale and shift time-t distribution atoms by discount and reward.
    target_z = r_t[:, jnp.newaxis] + d_t[:, jnp.newaxis] * q_atoms_t
    # Select logits for greedy action in state s_t and convert to distribution.
    p_target_z = jax.nn.softmax(q_logits_t[batch_indices, q_t_selector.argmax(-1)])
    # Project using the Cramer distance and maybe stop gradient flow to targets.
    target = jax.vmap(rlax.categorical_l2_project)(target_z, p_target_z, q_atoms_tm1)
    # Compute loss (i.e. temporal difference error).
    logit_qa_tm1 = q_logits_tm1[batch_indices, a_tm1]
    td_error = -jnp.sum(target * jax.nn.log_softmax(logit_qa_tm1, axis=-1), axis=-1)

    return td_error


def q_learning(
    q_tm1: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_t: chex.Array,
    huber_loss_parameter: chex.Array,
    weights: chex.Array,
) -> jnp.ndarray:
    """Computes the double Q-learning loss. Each input is a batch.

    weights masks invalid samples (e.g. truncated anchors) out of the mean.
    """
    batch_indices = jnp.arange(a_tm1.shape[0])
    # Compute Q-learning n-step TD-error.
    target_tm1 = r_t + d_t * jnp.max(q_t, axis=-1)
    td_error = target_tm1 - q_tm1[batch_indices, a_tm1]
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_error)

    return jnp.sum(batch_loss * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def munchausen_q_learning(
    q_tm1: chex.Array,
    q_tm1_target: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_t_target: chex.Array,
    entropy_temperature: chex.Array,
    munchausen_coefficient: chex.Array,
    clip_value_min: chex.Array,
    huber_loss_parameter: chex.Array,
    weights: chex.Array,
) -> chex.Array:
    """Munchausen Q-learning loss. Each input is a batch.

    weights masks invalid samples (e.g. truncated anchors) out of the mean.
    """
    action_one_hot = jax.nn.one_hot(a_tm1, q_tm1.shape[-1])
    q_tm1_a = jnp.sum(q_tm1 * action_one_hot, axis=-1)
    # Compute double Q-learning loss.
    # Munchausen term : tau * log_pi(a|s)
    munchausen_term = entropy_temperature * jax.nn.log_softmax(
        q_tm1_target / entropy_temperature, axis=-1
    )
    munchausen_term_a = jnp.sum(action_one_hot * munchausen_term, axis=-1)
    munchausen_term_a = jnp.clip(munchausen_term_a, clip_value_min, 0.0)

    # Soft Bellman operator applied to q
    next_v = entropy_temperature * jax.nn.logsumexp(q_t_target / entropy_temperature, axis=-1)
    target_q = jax.lax.stop_gradient(
        r_t + munchausen_coefficient * munchausen_term_a + d_t * next_v
    )
    td_error = target_q - q_tm1_a
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_error)
    batch_loss = jnp.sum(batch_loss * weights) / jnp.maximum(jnp.sum(weights), 1.0)
    return batch_loss


def munchausen_quantile_q_learning(
    z_tm1: chex.Array,
    quantiles_tm1: chex.Array,
    q_tm1_target: chex.Array,
    z_t_target: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    entropy_temperature: chex.Array,
    munchausen_coefficient: chex.Array,
    clip_value_min: chex.Array,
    quantile_huber_kappa: chex.Array,
) -> chex.Array:
    """Munchausen-IQN loss (Vieillard et al. 2020, Appx. B.1). Each input is a batch.

    Regresses the online quantile values z_tm1 (B, N, A) at the sampled fractions
    quantiles_tm1 (B, N) onto the per-target-quantile Munchausen target built from
    z_t_target (B, N', A):

        T_j = r + alpha * clip(tau * ln pi(a|s), l0, 0)
              + d * sum_a' pi(a'|s') * (z_j(s', a') - tau * ln pi(a'|s'))

    with pi = softmax(q_target / tau) and q_target the quantile mean. The quantile
    Huber loss is summed over online quantiles and averaged over target quantiles.
    Returns the per-batch loss vector (B,) so callers can apply importance weights.
    """
    action_one_hot = jax.nn.one_hot(a_tm1, z_tm1.shape[-1])
    # Munchausen term: alpha * clip(tau * ln pi(a|s), l0, 0) at step 0.
    munchausen_term = entropy_temperature * jax.nn.log_softmax(
        q_tm1_target / entropy_temperature, axis=-1
    )
    munchausen_term_a = jnp.sum(action_one_hot * munchausen_term, axis=-1)
    munchausen_term_a = jnp.clip(munchausen_term_a, clip_value_min, 0.0)

    # Soft per-quantile bootstrap: sum_a' pi(a'|s') * (z_j(s', a') - tau * ln pi(a'|s')).
    q_t_target = jnp.mean(z_t_target, axis=1)
    log_pi_t = jax.nn.log_softmax(q_t_target / entropy_temperature, axis=-1)
    pi_t = jax.nn.softmax(q_t_target / entropy_temperature, axis=-1)
    soft_z_t = jnp.sum(
        pi_t[:, jnp.newaxis, :] * (z_t_target - entropy_temperature * log_pi_t[:, jnp.newaxis, :]),
        axis=-1,
    )
    target = (
        r_t[:, jnp.newaxis]
        + munchausen_coefficient * munchausen_term_a[:, jnp.newaxis]
        + d_t[:, jnp.newaxis] * soft_z_t
    )
    target = jax.lax.stop_gradient(target)

    # Pairwise quantile Huber loss over N online x N' target quantiles.
    z_tm1_a = jnp.sum(z_tm1 * action_one_hot[:, jnp.newaxis, :], axis=-1)
    td_error = target[:, jnp.newaxis, :] - z_tm1_a[:, :, jnp.newaxis]
    abs_td_error = jnp.abs(td_error)
    huber = jnp.where(
        abs_td_error <= quantile_huber_kappa,
        0.5 * td_error**2,
        quantile_huber_kappa * (abs_td_error - 0.5 * quantile_huber_kappa),
    )
    indicator = (td_error < 0.0).astype(jnp.float32)
    rho = jnp.abs(quantiles_tm1[:, :, jnp.newaxis] - indicator) * huber / quantile_huber_kappa
    return jnp.sum(jnp.mean(rho, axis=2), axis=1)


def munchausen_q_learning_n_step(
    q_online: chex.Array,
    q_target: chex.Array,
    a_t: chex.Array,
    r_t: chex.Array,
    terminated: chex.Array,
    truncated: chex.Array,
    gamma: float,
    n_steps: int,
    entropy_temperature: chex.Array,
    munchausen_coefficient: chex.Array,
    clip_value_min: chex.Array,
    huber_loss_parameter: chex.Array,
) -> chex.Array:
    """N-step Munchausen Q-learning over (batch, time, ...) learn-window sequences.

    Every timestep t in [0, T - n_steps) anchors an n-step target
    sum_{i<n} gamma^i * r[t+i] + gamma^n * soft_v[t+n], where the Munchausen
    bonus shapes only the anchor step's reward r[t] (intermediate rewards are
    not shaped, matching the M-DQN paper, BTR and BY571). An interior
    termination stops the sum after that step's reward (no bootstrap);
    an interior truncation at offset k bootstraps gamma^k * soft_v from the
    truncated step. Truncated anchors are masked out of the mean.
    """
    action_one_hot = jax.nn.one_hot(a_t, q_target.shape[-1])
    # Munchausen addon r + alpha * clip(tau * ln pi(a|s), l0, 0), applied only at
    # the anchor step (the final unroll level below).
    munchausen_term = entropy_temperature * jax.nn.log_softmax(
        q_target / entropy_temperature, axis=-1
    )
    munchausen_term_a = jnp.sum(action_one_hot * munchausen_term, axis=-1)
    munchausen_term_a = jnp.clip(munchausen_term_a, clip_value_min, 0.0)
    shaped_r = r_t + munchausen_coefficient * munchausen_term_a

    # Soft state value used for every bootstrap.
    soft_v = entropy_temperature * jax.nn.logsumexp(q_target / entropy_temperature, axis=-1)

    terminated = terminated.astype(jnp.float32)
    truncated = truncated.astype(jnp.float32)
    # Backward recursion: G(t, 0) = soft_v[t]; a truncated step's reward and
    # successor are unusable, so bootstrap there instead. Only the final unroll
    # level adds the anchor step's own reward, so only it carries the Munchausen
    # addon (intermediate n-step rewards are not shaped).
    target_q = soft_v
    for h in range(1, n_steps + 1):
        r_h = shaped_r if h == n_steps else r_t
        target_q = jnp.where(
            truncated[:, :-h] == 1,
            soft_v[:, :-h],
            r_h[:, :-h] + gamma * (1.0 - terminated[:, :-h]) * target_q[:, 1:],
        )
    target_q = jax.lax.stop_gradient(target_q)

    anchors = q_online.shape[1] - n_steps
    q_t_a = jnp.sum(q_online[:, :anchors] * action_one_hot[:, :anchors], axis=-1)
    td_error = target_q - q_t_a
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_error)
    weights = 1.0 - truncated[:, :anchors]
    return jnp.sum(batch_loss * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def munchausen_quantile_q_learning_n_step(
    z_online: chex.Array,
    quantiles: chex.Array,
    z_target: chex.Array,
    a_t: chex.Array,
    r_t: chex.Array,
    terminated: chex.Array,
    truncated: chex.Array,
    gamma: float,
    n_steps: int,
    entropy_temperature: chex.Array,
    munchausen_coefficient: chex.Array,
    clip_value_min: chex.Array,
    quantile_huber_kappa: chex.Array,
    q_selector: chex.Array | None = None,
) -> tuple[chex.Array, dict]:
    """N-step Munchausen-IQN loss over (batch, time, ...) learn-window sequences.

    The quantile analogue of munchausen_q_learning_n_step: every timestep t in
    [0, T - n_steps) anchors an n-step target per target quantile j,
    sum_{i<n} gamma^i * shaped_r[t+i] + gamma^n * soft_z[t+n, j], where the
    Munchausen bonus is reward shaping applied to every step's reward and
    soft_z_j = sum_a pi(a|s) * (z_j(s, a) - tau * ln pi(a|s)) is the per-quantile
    soft bootstrap, with pi = softmax(q_target / tau) from the target quantile mean.
    An interior termination stops the sum after that step's reward (no bootstrap);
    an interior truncation at offset k bootstraps gamma^k * soft_z from the
    truncated step. The pairwise quantile Huber loss (sum over the N online
    quantiles, mean over the N' target quantiles) is averaged over the anchors,
    with truncated anchors masked out.

    `q_selector`, if given, is a (batch, time, actions) q-value array (already
    stop_gradient-ed) used *instead of* the target quantile mean to build the
    bootstrap policy pi -- Double-DQN's selection/evaluation split, adapted to
    the soft bootstrap: the selector picks the action weights, the target
    network still supplies the quantile values they weight. The Munchausen
    anchor term keeps using the target network's own log-policy, since that is
    the "previous policy" the M-DQN log-policy bonus is defined against.

    Returns (loss, aux) where aux carries monitoring-only scalars: without them
    an overestimating Q is invisible, because the quantile Huber loss alone
    cannot show the *sign* of the TD error.
    """
    action_one_hot = jax.nn.one_hot(a_t, z_target.shape[-1])
    # Munchausen reward shaping: r + alpha * clip(tau * ln pi(a|s), l0, 0), with pi
    # computed from the target network's quantile-mean q-values.
    q_target = jnp.mean(z_target, axis=2)
    log_pi = jax.nn.log_softmax(q_target / entropy_temperature, axis=-1)
    munchausen_term_a = jnp.sum(action_one_hot * entropy_temperature * log_pi, axis=-1)
    munchausen_clipped = munchausen_term_a <= clip_value_min
    munchausen_term_a = jnp.clip(munchausen_term_a, clip_value_min, 0.0)
    shaped_r = r_t + munchausen_coefficient * munchausen_term_a

    # Bootstrap policy: the target's own log-policy unless a selector is supplied.
    if q_selector is None:
        boot_log_pi = log_pi
    else:
        boot_log_pi = jax.nn.log_softmax(q_selector / entropy_temperature, axis=-1)
    boot_pi = jnp.exp(boot_log_pi)

    # Per-quantile soft bootstrap: sum_a pi(a|s) * (z_j(s, a) - tau * ln pi(a|s)).
    soft_z = jnp.sum(
        boot_pi[:, :, jnp.newaxis, :]
        * (z_target - entropy_temperature * boot_log_pi[:, :, jnp.newaxis, :]),
        axis=-1,
    )

    terminated = terminated.astype(jnp.float32)
    truncated = truncated.astype(jnp.float32)
    # Backward recursion over the N' quantile axis: G(t, 0) = soft_z[t]; a truncated
    # step's reward and successor are unusable, so bootstrap there instead. Only the
    # final unroll level adds the anchor step's own reward, so only it carries the
    # Munchausen addon (BTR/BY571: intermediate n-step rewards are not shaped).
    target = soft_z
    for h in range(1, n_steps + 1):
        r_h = shaped_r if h == n_steps else r_t
        target = jnp.where(
            truncated[:, :-h, jnp.newaxis] == 1,
            soft_z[:, :-h],
            r_h[:, :-h, jnp.newaxis]
            + gamma * (1.0 - terminated[:, :-h, jnp.newaxis]) * target[:, 1:],
        )
    target = jax.lax.stop_gradient(target)

    # Pairwise quantile Huber loss over N online x N' target quantiles per anchor.
    anchors = z_online.shape[1] - n_steps
    z_t_a = jnp.sum(z_online[:, :anchors] * action_one_hot[:, :anchors, jnp.newaxis, :], axis=-1)
    td_error = target[:, :, jnp.newaxis, :] - z_t_a[:, :, :, jnp.newaxis]
    abs_td_error = jnp.abs(td_error)
    huber = jnp.where(
        abs_td_error <= quantile_huber_kappa,
        0.5 * td_error**2,
        quantile_huber_kappa * (abs_td_error - 0.5 * quantile_huber_kappa),
    )
    indicator = (td_error < 0.0).astype(jnp.float32)
    rho = jnp.abs(quantiles[:, :anchors, :, jnp.newaxis] - indicator) * huber / quantile_huber_kappa
    anchor_loss = jnp.sum(jnp.mean(rho, axis=3), axis=2)
    weights = 1.0 - truncated[:, :anchors]
    denom = jnp.maximum(jnp.sum(weights), 1.0)
    loss = jnp.sum(anchor_loss * weights) / denom

    # Monitoring only. td_signed is the headline number: a persistently negative
    # mean (target below prediction) is overestimation, which the Huber loss hides.
    q_online_a = jnp.mean(z_t_a, axis=-1)
    q_target_mean = jnp.mean(target, axis=-1)
    entropy = -jnp.sum(boot_pi * boot_log_pi, axis=-1)
    q_online_all = jnp.mean(z_online, axis=2)
    aux = {
        "td_signed": jnp.sum((q_target_mean - q_online_a) * weights) / denom,
        "td_abs": jnp.sum(jnp.abs(q_target_mean - q_online_a) * weights) / denom,
        "q_online_taken": jnp.sum(q_online_a * weights) / denom,
        "q_target_n_step": jnp.sum(q_target_mean * weights) / denom,
        "munchausen_clip_fraction": jnp.mean(munchausen_clipped.astype(jnp.float32)),
        "bootstrap_policy_entropy": jnp.mean(entropy),
        "entropy_bonus": entropy_temperature * jnp.mean(entropy),
        # Per-*state* spread across actions. The q_online_action_i scalars below
        # average over states first, so a net whose action ranking stops depending
        # on the state -- one that has collapsed to picking by a constant bias --
        # leaves them unchanged while this collapses towards zero.
        "q_action_dispersion": jnp.mean(jnp.std(q_online_all, axis=-1)),
    }
    # Per-action Q as separate scalars: an action the policy never takes gets no
    # gradient, and its value silently drifting away from the others is the
    # signature of that. The action count is static at trace time.
    q_per_action = jnp.mean(q_online_all, axis=(0, 1))
    for i in range(z_online.shape[-1]):
        aux[f"q_online_action_{i}"] = q_per_action[i]
    return loss, aux
