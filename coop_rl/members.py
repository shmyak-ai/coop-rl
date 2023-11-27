"""
The abstract classes for the agents and workers
"""

import tensorflow as tf
import reverb

from coop_rl.networks import get_dense_value


class Member:
    member_config = {
        'model':  {
            'dense_value': get_dense_value,
        }
    }

    def __init__(self, run_config):
        self._buffer_client = reverb.Client(
            f'{run_config.buffer_server_ip}:{run_config.buffer_server_port}'
        )
        self._model = Member.member_config['model'][run_config.model]()
        self._table_names = run_config.table_names


class Agent(Member):
    agent_config = {
        'optimizer':  {
            'adam': tf.keras.optimizers.Adam,
        },
        'loss':  {
            'huber': tf.keras.losses.Huber,
        }
    }

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
    def policy(self):
        raise NotImplementedError

    def collect(self):
        raise NotImplementedError
