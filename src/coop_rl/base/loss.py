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
) -> jnp.ndarray:
    """Computes the double Q-learning loss. Each input is a batch."""
    batch_indices = jnp.arange(a_tm1.shape[0])
    # Compute Q-learning n-step TD-error.
    target_tm1 = r_t + d_t * jnp.max(q_t, axis=-1)
    td_error = target_tm1 - q_tm1[batch_indices, a_tm1]
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_error)

    return jnp.mean(batch_loss)


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
) -> chex.Array:
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
    batch_loss = jnp.mean(batch_loss)
    return batch_loss
