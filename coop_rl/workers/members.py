"""
The basic classes for ray actors
"""
import itertools

import tensorflow as tf
import reverb
import gymnasium as gym

from coop_rl.networks import DenseCritic


class Member:
    """
    Members are agents and workers.
    They all possess an exchange actor for communications and a model (nn network).
    """
    member_config = {
        'model':  {
            'dense_critic': DenseCritic,
        }
    }

    def __init__(self, run_config, exchange_actor, weights):
        self._run_config = run_config
        self._exchange_actor = exchange_actor
        self._model = Member.member_config['model'][run_config.model](
            run_config.n_features,
            run_config.n_layers,
            run_config.seed,
        )
        if weights is not None:
            self._model.set_weights(weights['weights'])


class Agent(Member):
    """
    Agents contain an oprimizer, a loss function, a client to make a buffer checkpoints,
    and iterators for sampling.
    """
    agent_config = {
        'optimizer':  {
            'adam': tf.keras.optimizers.Adam,
        },
        'loss':  {
            'huber': tf.keras.losses.Huber,
        },
    }

    def __init__(self, run_config, exchange_actor, weights, make_checkpoint):
        super().__init__(run_config, exchange_actor, weights)

        self._optimizer = Agent.agent_config['optimizer'][run_config.optimizer]
        self._optimizer.learning_rate.assign(run_config.learning_rate)
        self._loss_fn = Agent.agent_config['loss'][run_config.loss]

        # to make checkpoints during training
        self._buffer_client = reverb.Client(
            f'{run_config.buffer_server_ip}:{run_config.buffer_server_port}'
        )
        self._table_names = run_config.table_names
        self._make_checkpoint = make_checkpoint

        # initialize datasets to sample data from a server
        datasets = [reverb.TrajectoryDataset.from_table_signature(
            server_address=f'{run_config.buffer_server_ip}:{run_config.buffer_server_port}',
            table=run_config.table_names[i],
            max_in_flight_samples_per_worker=10,
        ) for i in range(run_config.tables_number)]
        self._iterators = [iter(datasets[i]) for i in range(run_config.tables_number)]

        if not run_config.debug:
            self._training_step = tf.function(self._training_step)
        self.do_train = self.do_train_generator()

    def do_train_generator(self):
        for i in itertools.count():
            samples = self._sample_experience()
            self._traininig_step(samples)
            yield i

    def _sample_experience(self):
        raise NotImplementedError

    def _training_step(self):
        """A RL algorithmic step"""
        raise NotImplementedError

    def train(self):
        """Saving, printing, adjusting etc., calls _training_step through next(self.do_train)"""
        raise NotImplementedError

    def training(self):
        """Train stepping, calls train"""
        raise NotImplementedError


class Worker(Member):
    """
    Workers are collectors and evaluators.
    They contain an environment.
    """

    def __init__(self, run_config, exchange_actor, weights):
        super().__init__(run_config, exchange_actor, weights)

        self._env = gym.make(run_config.env_name)
