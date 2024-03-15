import tensorflow as tf

import reverb


class DQNUniformBuffer:
    def __init__(
        self,
        run_config,
        checkpointer=None
        ):

        min_size = run_config.batch_size
        max_size = run_config.buffer_size

        action_spec = tf.TensorSpec([run_config.n_outputs], tf.int32)
        observation_spec = tf.TensorSpec(run_config.input_shape, tf.float32)
        rewards_spec = tf.TensorSpec([], tf.float32)
        dones_spec = tf.TensorSpec([], tf.float32)

        self._min_size = min_size
        self._table_names = run_config.table_names
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
                            tf.TensorSpec([i + 2, *action_spec.shape], action_spec.dtype),
                        'observations':
                            tf.TensorSpec([i + 2, *observation_spec.shape], observation_spec.dtype),
                        'rewards':
                            tf.TensorSpec([i + 2, *rewards_spec.shape], rewards_spec.dtype),
                        'dones':
                            tf.TensorSpec([i + 2, *dones_spec.shape], dones_spec.dtype),
            }
                ) for i in range(run_config.tables_number)
            ],
            port=run_config.buffer_server_port,
            checkpointer=checkpointer
        )

    @property
    def table_names(self) -> list[str]:
        return self._table_names

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def server_port(self) -> int:
        return self._server.port
