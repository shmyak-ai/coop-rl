import gymnasium as gym
import ray


def check_environment(run_config):
    train_env = gym.make(run_config.env_name)
    input_shape = train_env.observation_space.shape
    n_outputs = train_env.action_space.n
    return input_shape, n_outputs


@ray.remote(num_cpus=0)
class ExchangeActor:

    def __init__(self, num_collectors):
        self.num_collectors = num_collectors
        self.collector_id = 0
        self.done = False
        self.weights = None

    def set_done(self, done):
        self.done = done

    def is_done(self):
        return self.done

    def increment_collector_id(self):
        self.collector_id += 1
        if self.collector_id >= self.num_collectors:
            self.collector_id = 0

    def get_collector_id(self):
        return self.collector_id

    def set_weights(self, w):
        self.weights = w

    def get_weights(self):
        return self.weights
