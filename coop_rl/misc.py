import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import ray


def get_prob_logs_from_logits(logits, actions, n_outputs):
    probs = tf.nn.softmax(logits)
    mask = tf.one_hot(actions, n_outputs, dtype=tf.float32)
    masked_probs = tf.reduce_sum(probs * mask, axis=-1)
    logs = tf.math.log(tf.clip_by_value(masked_probs, 1.e-32, 1.))  # inappropriate values will be masked
    return logs


def prepare_td_lambda(values, returns, rewards, lmb, gamma):
    from collections import deque

    target_values = deque([returns])
    for i in range(values.shape[0] - 1, 0, -1):
        reward = rewards[i, :] if rewards is not None else 0
        target_values.appendleft(reward + gamma * ((1 - lmb) * values[i, :] + lmb * target_values[0]))

    target_values = tf.stack(tuple(target_values))
    return target_values


def tf_prepare_td_lambda_no_rewards(values, returns, lmb, gamma):
    reward = 0
    ta = tf.TensorArray(dtype=tf.float32, size=values.shape[1], dynamic_size=False)
    row = returns
    ta = ta.write(values.shape[1] - 1, row)
    for i in tf.range(values.shape[1] - 1, 0, -1):
        # prev = ta.read(i)  # read does not work properly
        # row = reward + gamma * ((1 - lmb) * values[i, :] + lmb * prev)
        row = reward + gamma * ((1 - lmb) * values[:, i] + lmb * row)
        ta = ta.write(i - 1, row)

    target_values = ta.stack()
    return tf.transpose(target_values)


def get_entropy(logits, mask=None):
    # another way to calculate entropy:
    # probs = tf.nn.softmax(logits)
    # entropy = tf.keras.losses.categorical_crossentropy(probs, probs)
    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)
    # log_probs = tf.math.log(probs)
    if mask is not None:
        probs = probs * mask
        log_probs = log_probs * mask
    entropy = tf.reduce_sum(-probs * log_probs, axis=-1)
    return entropy


def get_non_max(x):
    if x == 0:
        return tf.constant([1, 2, 3], dtype=tf.int32)
    elif x == 1:
        return tf.constant([0, 2, 3], dtype=tf.int32)
    elif x == 2:
        return tf.constant([0, 1, 3], dtype=tf.int32)
    else:
        return tf.constant([0, 1, 2], dtype=tf.int32)


# by Taaam, https://stackoverflow.com/questions/38492608/tensorflow-indexing-into-2d-tensor-with-1d-tensor
def vector_slice(A, B):
    """ Returns values of rows i of A at column B[i]

    where A is a 2D Tensor with shape [None, D]
    and B is a 1D Tensor with shape [None]
    with type int32 elements in [0,D)

    Example:
      A =[[1,2], B = [0,1], vector_slice(A,B) -> [1,4]
          [3,4]]
    """
    linear_index = (tf.shape(A)[1] * tf.range(0, tf.shape(A)[0]))
    linear_A = tf.reshape(A, [-1])
    return tf.gather(linear_A, B + linear_index)


# def project_distribution is from (https://github.com/google/dopamine):
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def project_distribution(supports, weights, target_support,
                         validate_args=False):
    """Projects a batch of (support, weights) onto target_support.
  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.
  This code is not easy to digest, so we will use a running example to clarify
  what is going on, with the following sample inputs:
    * supports =       [[0, 2, 4, 6, 8],
                        [1, 3, 4, 5, 6]]
    * weights =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.5, 0.1, 0.1]]
    * target_support = [4, 5, 6, 7, 8]
  In the code below, comments preceded with 'Ex:' will be referencing the above
  values.
  Args:
    supports: Tensor of shape (batch_size, num_dims) defining supports for the
      distribution.
    weights: Tensor of shape (batch_size, num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Tensor of shape (num_dims) defining support of the projected
      distribution. The values must be monotonically increasing. Vmin and Vmax
      will be inferred from the first and last elements of this tensor,
      respectively. The values in this tensor must be equally spaced.
    validate_args: Whether we will verify the contents of the
      target_support parameter.
  Returns:
    A Tensor of shape (batch_size, num_dims) with the projection of a batch of
    (support, weights) onto target_support.
  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
    target_support_deltas = target_support[1:] - target_support[:-1]
    # delta_z = `\Delta z` in Eq7.
    delta_z = target_support_deltas[0]
    validate_deps = []
    supports.shape.assert_is_compatible_with(weights.shape)
    supports[0].shape.assert_is_compatible_with(target_support.shape)
    target_support.shape.assert_has_rank(1)
    if validate_args:
        # Assert that supports and weights have the same shapes.
        validate_deps.append(
            tf.Assert(
                tf.reduce_all(tf.equal(tf.shape(supports), tf.shape(weights))),
                [supports, weights]))
        # Assert that elements of supports and target_support have the same shape.
        validate_deps.append(
            tf.Assert(
                tf.reduce_all(
                    tf.equal(tf.shape(supports)[1], tf.shape(target_support))),
                [supports, target_support]))
        # Assert that target_support has a single dimension.
        validate_deps.append(
            tf.Assert(
                tf.equal(tf.size(tf.shape(target_support)), 1), [target_support]))
        # Assert that the target_support is monotonically increasing.
        validate_deps.append(
            tf.Assert(tf.reduce_all(target_support_deltas > 0), [target_support]))
        # Assert that the values in target_support are equally spaced.
        validate_deps.append(
            tf.Assert(
                tf.reduce_all(tf.equal(target_support_deltas, delta_z)),
                [target_support]))

    with tf.control_dependencies(validate_deps):
        # Ex: `v_min, v_max = 4, 8`.
        v_min, v_max = target_support[0], target_support[-1]
        # Ex: `batch_size = 2`.
        batch_size = tf.shape(supports)[0]
        # `N` in Eq7.
        # Ex: `num_dims = 5`.
        num_dims = tf.shape(target_support)[0]
        # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
        # Ex: `clipped_support = [[[ 4.  4.  4.  6.  8.]]
        #                         [[ 4.  4.  4.  5.  6.]]]`.
        clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]
        # Ex: `tiled_support = [[[[ 4.  4.  4.  6.  8.]
        #                         [ 4.  4.  4.  6.  8.]
        #                         [ 4.  4.  4.  6.  8.]
        #                         [ 4.  4.  4.  6.  8.]
        #                         [ 4.  4.  4.  6.  8.]]
        #                        [[ 4.  4.  4.  5.  6.]
        #                         [ 4.  4.  4.  5.  6.]
        #                         [ 4.  4.  4.  5.  6.]
        #                         [ 4.  4.  4.  5.  6.]
        #                         [ 4.  4.  4.  5.  6.]]]]`.
        tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
        # Ex: `reshaped_target_support = [[[ 4.]
        #                                  [ 5.]
        #                                  [ 6.]
        #                                  [ 7.]
        #                                  [ 8.]]
        #                                 [[ 4.]
        #                                  [ 5.]
        #                                  [ 6.]
        #                                  [ 7.]
        #                                  [ 8.]]]`.
        reshaped_target_support = tf.tile(target_support[:, None], [batch_size, 1])
        reshaped_target_support = tf.reshape(reshaped_target_support,
                                             [batch_size, num_dims, 1])
        # numerator = `|clipped_support - z_i|` in Eq7.
        # Ex: `numerator = [[[[ 0.  0.  0.  2.  4.]
        #                     [ 1.  1.  1.  1.  3.]
        #                     [ 2.  2.  2.  0.  2.]
        #                     [ 3.  3.  3.  1.  1.]
        #                     [ 4.  4.  4.  2.  0.]]
        #                    [[ 0.  0.  0.  1.  2.]
        #                     [ 1.  1.  1.  0.  1.]
        #                     [ 2.  2.  2.  1.  0.]
        #                     [ 3.  3.  3.  2.  1.]
        #                     [ 4.  4.  4.  3.  2.]]]]`.
        numerator = tf.abs(tiled_support - reshaped_target_support)
        quotient = 1 - (numerator / delta_z)
        # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
        # Ex: `clipped_quotient = [[[[ 1.  1.  1.  0.  0.]
        #                            [ 0.  0.  0.  0.  0.]
        #                            [ 0.  0.  0.  1.  0.]
        #                            [ 0.  0.  0.  0.  0.]
        #                            [ 0.  0.  0.  0.  1.]]
        #                           [[ 1.  1.  1.  0.  0.]
        #                            [ 0.  0.  0.  1.  0.]
        #                            [ 0.  0.  0.  0.  1.]
        #                            [ 0.  0.  0.  0.  0.]
        #                            [ 0.  0.  0.  0.  0.]]]]`.
        clipped_quotient = tf.clip_by_value(quotient, 0, 1)
        # Ex: `weights = [[ 0.1  0.6  0.1  0.1  0.1]
        #                 [ 0.1  0.2  0.5  0.1  0.1]]`.
        weights = weights[:, None, :]
        # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
        # in Eq7.
        # Ex: `inner_prod = [[[[ 0.1  0.6  0.1  0.  0. ]
        #                      [ 0.   0.   0.   0.  0. ]
        #                      [ 0.   0.   0.   0.1 0. ]
        #                      [ 0.   0.   0.   0.  0. ]
        #                      [ 0.   0.   0.   0.  0.1]]
        #                     [[ 0.1  0.2  0.5  0.  0. ]
        #                      [ 0.   0.   0.   0.1 0. ]
        #                      [ 0.   0.   0.   0.  0.1]
        #                      [ 0.   0.   0.   0.  0. ]
        #                      [ 0.   0.   0.   0.  0. ]]]]`.
        inner_prod = clipped_quotient * weights
        # Ex: `projection = [[ 0.8 0.0 0.1 0.0 0.1]
        #                    [ 0.8 0.1 0.1 0.0 0.0]]`.
        projection = tf.reduce_sum(inner_prod, 3)
        projection = tf.reshape(projection, [batch_size, num_dims])
        return projection


# @ray.remote(num_gpus=1)
def use_gpu():
    """
    Call to check ids of available GPUs:
    ray.init()
    use_gpu.remote()
    """
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))


def plot_2d_array(array, name):
    fig = plt.figure(1)
    # make a color map of fixed colors
    # cmap = mpl.colors.ListedColormap(['blue', 'black', 'red'])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(  # noqa
        'my_colormap',
        ['blue', 'black', 'red'],
        256
    )
    # bounds = [-6, -2, 2, 6]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)  # noqa

    # tell imshow about color map so that only set colors are used
    # img = plt.imshow(array, interpolation='nearest', cmap=cmap, norm=norm)
    img = plt.imshow(array, interpolation='nearest', cmap=cmap)

    # make a color bar
    # plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-5, 0, 5])
    plt.colorbar(img)
    # plt.show()
    fig.savefig("data/pictures/" + name + ".png")
    plt.close(fig)


@ray.remote
class GlobalVarActor:
    def __init__(self):
        self.global_v = 1
        self.current_weights = None, None
        self.done = False

    def set_global_v(self, v):
        self.global_v = v

    def get_global_v(self):
        return self.global_v

    def set_current_weights(self, w):
        self.current_weights = w

    def get_current_weights(self):
        return self.current_weights

    def set_done(self, done):
        self.done = done

    def get_done(self):
        return self.done
