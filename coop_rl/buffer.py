import tensorflow as tf

import reverb
from typing import List


def get_1d_dataset(server_ip, server_port, table_name, observations_shape, batch_size, n_points):

    actions_tf_shape = tf.TensorShape([])
    observations_tf_shape = tf.TensorShape(observations_shape)
    rewards_tf_shape = tf.TensorShape([])
    dones_tf_shape = tf.TensorShape([])

    obs_dtypes = tf.nest.map_structure(lambda x: tf.uint8, observations_tf_shape)

    dataset = reverb.ReplayDataset(
        server_address=f'{server_ip}:{server_port}',
        table=table_name,
        max_in_flight_samples_per_worker=2 * batch_size,
        dtypes=(tf.int32, obs_dtypes, tf.float32, tf.float32),
        shapes=(actions_tf_shape, observations_tf_shape, rewards_tf_shape, dones_tf_shape))
    dataset = dataset.batch(n_points)
    dataset = dataset.batch(batch_size)

    return dataset


class UniformBuffer:
    def __init__(
        self,
        port: int = 8000,
        num_tables: int = 1,
        table_names: List[str] = ["uniform_table_0"],
        min_size: int = 64,
        max_size: int = 100000,
        checkpointer=None
        ):

        self._min_size = min_size
        self._table_names = table_names
        self._server = reverb.Server(
            tables=[
                reverb.Table(
                    name=self._table_names[i],
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=int(max_size),
                    rate_limiter=reverb.rate_limiters.MinSize(min_size),
                    signature={
                        'actions':
                            tf.TensorSpec([i + 2, *ACTION_SPEC.shape], ACTION_SPEC.dtype),
                        'observations':
                            tf.TensorSpec([i + 2, *OBSERVATION_SPEC.shape],
                                          OBSERVATION_SPEC.dtype),
            }
                ) for i in range(num_tables)
            ],
            port=port,
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
