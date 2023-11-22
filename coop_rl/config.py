import tensorflow as tf
import tensorflow_addons as tfa


CONF_DQN = {
    "agent": "DQN",
    "environment": "gym_goose:goose-full_control-v3",
    "multicall": False,
    "debug": True,
    #
    "buffer": "n_points",
    "n_points": 5,  # 2 points is a 1 step update, 3 points is a 2 steps update, and so on
    "all_trajectories": False,
    # "buffer": "full_episode",
    "buffer_size": 500000,
    "batch_size": 64,
    "init_episodes": 100,
    #
    "iterations_number": 20000,
    "eval_interval": 2000,
    "start_epsilon": .5,  # start for polynomial decay eps schedule, 1 is random, it should be real (double)
    "final_epsilon": .1,
    "optimizer": tf.keras.optimizers.Adam(lr=1.e-5),
    "loss": tf.keras.losses.Huber(),
    "discount_rate": tf.constant(.999, dtype=tf.float32)
}


CONF_PercentileDQN = {
    "agent": "percentileDQN",
    "environment": "gym_goose:goose-full_control-v3",
    "multicall": True,
    "debug": False,
    #
    "buffer": "n_points",
    "n_points": 5,  # 2 points is a 1 step update, 3 points is a 2 steps update, and so on
    "all_trajectories": False,
    # "buffer": "full_episode",
    "buffer_size": 500000,
    "batch_size": 64,
    "init_episodes": 100,
    #
    "iterations_number": 20000,
    "eval_interval": 2000,
    "start_epsilon": .5,  # start for polynomial decay eps schedule, it should be real (double)
    "final_epsilon": .1,
    "optimizer": tf.keras.optimizers.Adam(lr=1.e-5),
    "loss": tf.keras.losses.Huber(),
    "discount_rate": tf.constant(.999, dtype=tf.float32)
}


CONF_CategoricalDQN = {
    "agent": "categoricalDQN",
    "environment": "gym_goose:goose-full_control-v3",
    "multicall": True,
    "debug": False,
    #
    "buffer": "n_points",
    "n_points": 5,  # 2 points is a 1 step update, 3 points is a 2 steps update, and so on
    "all_trajectories": False,
    # "buffer": "full_episode",
    "buffer_size": 500000,
    "batch_size": 64,
    "init_episodes": 100,
    #
    "iterations_number": 20000,
    "eval_interval": 2000,
    "start_epsilon": .5,  # start for polynomial decay eps schedule, it should be real (double)
    "final_epsilon": .1,
    "optimizer": tf.keras.optimizers.Adam(lr=1.e-5),
    # "optimizer": tf.keras.optimizers.RMSprop(lr=1.e-4, rho=0.95, momentum=0.0,
    #                                          epsilon=0.00001, centered=True),
    "loss": None,  # it is hard-coded in the categorical algorithm
    "discount_rate": tf.constant(.99, dtype=tf.float32)
}


CONF_ActorCritic = {
    "agent": "actor-critic",
    "environment": "gym_goose:goose-v7",
    "setup": "complex",
    "debug": False,
    "collectors": 1,
    "default_lr": 1e-8,
    #
    "buffer": "full_episode",
    "n_points": 33,  # if full episode, it collects an episode first and then splits it to n_points pieces
    # "buffer": "n_points",
    # "all_trajectories": False,
    "buffer_size": 3000000,
    "batch_size": 100,
    "init_episodes": 25,  # not required by 'complex' setup
    #
    "iterations_number": 100000,
    "save_interval": 2000,
    "entropy_c": tf.constant(2.e-3),
    "entropy_c_decay": tf.constant(0.3),
    # "optimizer": tf.keras.optimizers.Adam(lr=1.e-6),
    "optimizer": tfa.optimizers.AdamW(weight_decay=1.e-5, learning_rate=1.e-6),
    "loss": None,
    # "loss": tf.keras.losses.Huber(),
    # "loss": tf.keras.losses.MeanSquaredError(),
    # "discount_rate": tf.constant(.999, dtype=tf.float32),
    "discount_rate": None,  # gamma, if full_episode, there is 1 hardcoded
    "lambda": tf.constant(.8, dtype=tf.float32)
}
