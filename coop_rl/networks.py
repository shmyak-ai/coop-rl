import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DenseBlock(layers.Layer):
    """
    A keras dense block.
    """
    def __init__(self, n_features, n_layers, seed):
        super().__init__()

        self._n_features = n_features
        self._n_layers = n_layers
        self._seed = seed
        self._dense = [layers.Dense(
            n_features,
            activation=tf.nn.tanh,
            # kernel_initializer=keras.initializers.VarianceScaling(
            #     scale=2.0, mode='fan_in', distribution='truncated_normal', seed=seed
            #     )
            kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform(
                seed=seed
            )
            ) for _ in range(n_layers)]

    def call(self, input_tensor):
        x = input_tensor
        for dense in self._dense:
            x = dense(x)
        return x

    def get_config(self):
        return {
            "n_features": self._n_features,
            "n_layers": self._n_layers,
            "seed": self._seed,
            }


class DenseDropBlock(layers.Layer):
    """
    A keras dense block with a dropout layer.
    """
    def __init__(self, n_features, n_layers, seed):
        super().__init__()

        self._n_features = n_features
        self._n_layers = n_layers
        self._seed = seed
        initializer = keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal', seed=seed
            )
        self._dense = [layers.Dense(
            n_features,
            activation=tf.nn.silu,
            kernel_initializer=initializer
            ) for _ in range(n_layers)]
        self._drop = [layers.Dropout(0.2, seed=seed) for _ in range(n_layers)]

    def call(self, input_tensor, training=False):
        x = input_tensor
        for dense, drop in zip(self._dense, self._drop):
            x = dense(x)
            x = drop(x, training=training)
        return x

    def get_config(self):
        return {
            "n_features": self._n_features,
            "n_layers": self._n_layers,
            "seed": self._seed,
            }


class DenseActor(layers.Layer):
    """
    A keras dense net providing logits.
    """
    def __init__(self, n_features, n_layers, num_outputs, seed=None):
        super().__init__()

        self._n_features = n_features
        self._n_layers = n_layers
        self._num_outputs = num_outputs
        self._seed = seed
        self._logits_block = DenseBlock(n_features, n_layers, seed)
        self._logits_out = layers.Dense(
            num_outputs,
            name="logits_out",
            activation=None,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=0.1,
                seed=seed
            ),
            bias_initializer=tf.keras.initializers.Zeros(),
        )

    def call(self, input_tensor, training=False):
        logits_layers = self._logits_block(input_tensor, training=training)
        logits = self._logits_out(logits_layers)
        return logits

    def get_config(self):
        return {
            "n_features": self._n_features,
            "n_layers": self._n_layers,
            "num_outputs": self._num_outputs,
            "seed": self._seed,
            }


class DenseCritic(layers.Layer):
    """
    A keras dense net providing a value.
    """
    def __init__(self, n_features, n_layers, seed=None):
        super().__init__()

        self._n_features = n_features
        self._n_layers = n_layers
        self._seed = seed
        self._value_block = DenseBlock(n_features, n_layers, seed)
        self._value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-0.03, maxval=0.03, seed=seed
            ),
        )

    def call(self, input_tensor, training=False):
        value_layers = self._value_block(input_tensor, training=training)
        value = self._value_out(value_layers)
        return value

    def get_config(self):
        return {
            "n_features": self._n_features,
            "n_layers": self._n_layers,
            "seed": self._seed,
            }


class DenseActorCritic(keras.Model):
    """
    A keras dense net.
    """
    def __init__(self, num_outputs, seed=None):
        super().__init__()

        self._n_features = 1024
        self._n_layers = 3
        self._num_outputs = num_outputs
        self._seed = seed
        self._actor = DenseActor(self._n_features, self._n_layers, num_outputs, seed)
        self._critic = DenseCritic(self._n_features, self._n_layers, seed)

    def call(self, input_tensor, training=False):
        logits = self._actor(input_tensor, training)
        value = self._critic(input_tensor, training)
        return logits, value

    def get_config(self):
        return {
            "n_features": self._n_features,
            "n_layers": self._n_layers,
            "num_outputs": self._num_outputs,
            "seed": self._seed,
            }


class ResidualUnit(layers.Layer):
    """
    1D convolutions to process row data.
    """
    def __init__(self, filters, initializer, activation):
        super().__init__()

        self._filters = filters
        self._activation = activation
        self._conv = layers.Conv1D(filters, 1, kernel_initializer=initializer, use_bias=False)
        self._norm = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self._conv(inputs)
        x = self._norm(x, training=training)
        return self._activation(inputs + x)

    def compute_output_shape(self, batch_input_shape):
        batch, x, _ = batch_input_shape
        return [batch, x, self._filters]


class ResidualBlock(layers.Layer):
    """
    A block of 1d residual units.
    """
    def __init__(self):
        super().__init__()

        n_filters = 64
        n_layers = 10

        initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'
            )
        activation = tf.keras.activations.relu

        self._conv = tf.keras.layers.Conv1D(
            n_filters, 1, kernel_initializer=initializer, use_bias=False
            )
        self._norm = tf.keras.layers.BatchNormalization()
        self._activation = tf.keras.layers.ReLU()
        self._residual_block = [
            ResidualUnit(n_filters, initializer, activation) for _ in range(n_layers)
            ]

        self._depthwise = tf.keras.layers.DepthwiseConv1D(15)
        self._flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=False):
        x = self._conv(inputs)
        x = self._norm(x, training=training)
        x = self._activation(x)

        for layer in self._residual_block:
            x = layer(x, training=training)

        x = self._depthwise(x)
        x = self._flatten(x)

        return x

    def get_config(self):
        pass


class ResNet(tf.keras.Model):
    """
    A 1d residual net providing both logits and a value.
    """
    def __init__(self, seed):
        super().__init__()

        initializer = keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal', seed=seed
            )
        self.block_1 = ResidualBlock()
        self.layer_out = tf.keras.layers.Dense(
            3,
            name="my_out",
            activation=None,
            kernel_initializer=initializer
        )
        self.value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=initializer
        )

    def call(self, inputs, training=False):
        x = self.block_1(inputs, training=training)
        layer_out = self.layer_out(x)
        value_out = self.value_out(x)
        return layer_out, value_out
