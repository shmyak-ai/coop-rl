import tensorflow as tf

import reverb
from typing import List


def initialize_dataset(server_port, table_name, observations_shape, batch_size, n_points, is_episode=False):
    maps_shape = tf.TensorShape(observations_shape[0])
    scalars_shape = tf.TensorShape(observations_shape[1])

    actions_tf_shape = tf.TensorShape([])
    rewards_tf_shape = tf.TensorShape([])
    dones_tf_shape = tf.TensorShape([])

    if is_episode:
        observations_tf_shape = ([n_points] + maps_shape, [n_points] + scalars_shape)
        obs_dtypes = tf.nest.map_structure(lambda x: tf.uint8, observations_tf_shape)

        dataset = reverb.ReplayDataset(
            server_address=f'localhost:{server_port}',
            table=table_name,
            max_in_flight_samples_per_worker=2 * batch_size,
            dtypes=(tf.int32, obs_dtypes, tf.float32, tf.float32),
            shapes=([n_points] + actions_tf_shape,
                    observations_tf_shape,
                    [n_points] + rewards_tf_shape,
                    [n_points] + dones_tf_shape))
    else:
        observations_tf_shape = (maps_shape, scalars_shape)
        obs_dtypes = tf.nest.map_structure(lambda x: tf.uint8, observations_tf_shape)

        dataset = reverb.ReplayDataset(
            server_address=f'localhost:{server_port}',
            table=table_name,
            max_in_flight_samples_per_worker=2 * batch_size,
            dtypes=(tf.int32, obs_dtypes, tf.float32, tf.float32),
            shapes=(actions_tf_shape, observations_tf_shape, rewards_tf_shape, dones_tf_shape))
        dataset = dataset.batch(n_points)

    dataset = dataset.batch(batch_size)

    return dataset


def initialize_dataset_with_logits(server_port, table_name, observations_shape, batch_size, n_points,
                                   is_episode=False):
    maps_shape = tf.TensorShape(observations_shape[0])
    scalars_shape = tf.TensorShape(observations_shape[1])

    actions_tf_shape = tf.TensorShape([])
    logits_tf_shape = tf.TensorShape([4, ])
    rewards_tf_shape = tf.TensorShape([])
    dones_tf_shape = tf.TensorShape([])
    total_rewards_tf_shape = tf.TensorShape([])
    progress_tf_shape = tf.TensorShape([])
    # episode_dones_tf_shape = tf.TensorShape([])
    # episode_steps_tf_shape = tf.TensorShape([])

    if is_episode:
        observations_tf_shape = ([n_points] + maps_shape, [n_points] + scalars_shape)
        obs_dtypes = tf.nest.map_structure(lambda x: tf.uint8, observations_tf_shape)

        dataset = reverb.ReplayDataset(
            server_address=f'localhost:{server_port}',
            table=table_name,
            max_in_flight_samples_per_worker=2 * batch_size,
            dtypes=(tf.int32, tf.float32, obs_dtypes, tf.float32, tf.float32),
            shapes=([n_points] + actions_tf_shape,
                    [n_points] + logits_tf_shape,
                    observations_tf_shape,
                    [n_points] + rewards_tf_shape,
                    [n_points] + dones_tf_shape))
    else:
        observations_tf_shape = (maps_shape, scalars_shape)
        obs_dtypes = tf.nest.map_structure(lambda x: tf.uint8, observations_tf_shape)

        dataset = reverb.ReplayDataset(
            server_address=f'localhost:{server_port}',
            table=table_name,
            max_in_flight_samples_per_worker=2 * batch_size,
            dtypes=(tf.int32, tf.float32, obs_dtypes, tf.float32, tf.float32, tf.float32, tf.float32),
            shapes=(actions_tf_shape, logits_tf_shape, observations_tf_shape, rewards_tf_shape, dones_tf_shape,
                    total_rewards_tf_shape, progress_tf_shape))
        dataset = dataset.batch(n_points)

    dataset = dataset.batch(batch_size)

    return dataset


class UniformBuffer:
    def __init__(self,
                 num_tables: int = 1,
                 min_size: int = 64,
                 max_size: int = 100000,
                 checkpointer=None):
        self._min_size = min_size
        self._table_names = [f"uniform_table_{i}" for i in range(num_tables)]
        self._server = reverb.Server(
            tables=[
                reverb.Table(
                    name=self._table_names[i],
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=int(max_size),
                    rate_limiter=reverb.rate_limiters.MinSize(min_size),
                ) for i in range(num_tables)
            ],
            # Sets the port to None to make the server pick one automatically.
            port=None,
            checkpointer=checkpointer
        )

    @property
    def table_names(self) -> List[str]:
        return self._table_names

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def server_port(self) -> int:
        return self._server.port
