"""
The abstract classes for the agents and workers
"""

import tensorflow as tf
import reverb
import gymnasium as gym

from coop_rl.networks import get_value_dense
from coop_rl.buffer import get_1d_dataset


class Member:
    member_config = {
        'model':  {
            'dense_value': get_value_dense,
        }
    }

    def __init__(self, run_config, data):
        self._buffer_client = reverb.Client(
            f'{run_config.buffer_server_ip}:{run_config.buffer_server_port}'
        )
        self._model = Member.member_config['model'][run_config.model](
            self._input_shape, self._n_outputs, is_duel=False)
        if data is not None:
            self._model.set_weights(data['weights'])
        self._table_names = run_config.table_names


class Agent(Member):
    agent_config = {
        'optimizer':  {
            'adam': tf.keras.optimizers.Adam,
        },
        'loss':  {
            'huber': tf.keras.losses.Huber,
        },
        'dataset': {
            '1d': get_1d_dataset,
        },
    }

    def __init__(self, run_config, data):
        super().__init__(run_config, data)

        self._optimizer = Agent.agent_config['optimizer'][run_config.optimizer]
        self._loss_fn = Agent.agent_config['loss'][run_config.loss]

        # initialize datasets to sample data from a server
        datasets = [Agent.agent_config['dataset'][run_config.dataset](
            run_config.buffer_server_ip,
            run_config.buffer_server_port,
            run_config.table_names[i],
            run_config.input_shape,
            run_config.batch_size,
            i + 2) for i in range(run_config.tables_number)]
        self._iterators = [iter(datasets[i]) for i in range(run_config.tables_number)]

        if not run_config.debug:
            self._training_step = tf.function(self._training_step)

    def _check_buffer(self):
        # wait if there are not enough data in the buffer
        items_created = []
        for table_name in self._table_names:
            server_info = self._buffer_client.server_info()[table_name]
            items_total = server_info.current_size + server_info.num_deleted_episodes
            items_created.append(items_total)

        if items_created[-1] < self._sample_batch_size:
            return False
        else:
            return True

    def _sample_experience(self):
        raise NotImplementedError

    def _training_step(self):
        raise NotImplementedError

    def train(self):
        """
        Runs a step of training.
        """
        samples = self._sample_experience()
        self._traininig_step(samples)


class Worker(Member):

    def __init__(self, run_config):
        super().__init__(run_config)
        self._train_env = gym.make(run_config.env_name)

    def policy(self):
        raise NotImplementedError

    def collect(self):
        raise NotImplementedError
