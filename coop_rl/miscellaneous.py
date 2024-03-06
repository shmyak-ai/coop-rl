import gymnasium as gym
import ray


def check_environment(run_config):
    train_env = gym.make(run_config.env_name)
    input_shape = train_env.observation_space.shape
    n_outputs = train_env.action_space.n
    return input_shape, n_outputs


@ray.remote(num_cpus=0)
class ExchangeActor:

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