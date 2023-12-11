"""
The basic classes for the actors
"""

import tensorflow as tf
import reverb
import gymnasium as gym
import ray

from coop_rl.networks import DenseCritic


@ray.remote(num_cpus=0)
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


class Member:
    member_config = {
        'model':  {
            'dense_critic': DenseCritic,
        }
    }

    def __init__(self, run_config, data):
        # clients are used by workers to add data to the replay buffer
        # agents use it only to check if there are enough data, is it necessary?
        # replace with try: next(iterator) probably
        self._buffer_client = reverb.Client(
            f'{run_config.buffer_server_ip}:{run_config.buffer_server_port}'
        )
        self._model = Member.member_config['model'][run_config.model](
            run_config.n_features,
            run_config.n_layers,
            run_config.seed,
        )
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
    }

    def __init__(self, run_config, data):
        super().__init__(run_config, data)

        self._optimizer = Agent.agent_config['optimizer'][run_config.optimizer]
        self._loss_fn = Agent.agent_config['loss'][run_config.loss]

        # initialize datasets to sample data from a server
        datasets = [reverb.TrajectoryDataset.from_table_signature(
            server_address=f'{run_config.buffer_server_ip}:{run_config.buffer_server_port}',
            table=run_config.table_names[i],
            max_in_flight_samples_per_worker=10,
        ) for i in range(run_config.tables_number)]
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
