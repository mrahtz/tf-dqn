from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Lambda, Flatten
import tensorflow as tf


def MLPPolicy(obs_shape, n_actions, n_hidden=64):
    obs = Input(shape=obs_shape)
    h = Dense(n_hidden, 'relu')(obs)
    qs = Dense(n_actions, None)(h)
    model = Model(inputs=obs, outputs=qs)
    return model


def check_normalised(obs):
    asserts = [tf.assert_greater_equal(obs, 0.0), tf.assert_less_equal(obs, 1.0)]
    with tf.control_dependencies(asserts):
        obs = tf.identity(obs)
    return obs


def CNNPolicy(obs_shape, n_actions):
    """
    Human-level control through deep reinforcement learning:

        The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the input image and applies a rectifier nonlinearity.
        The second hidden layer convolves 64 filters of 4 x 4 with stride 2, again followed by a rectifier nonlinearity.
        This is followed by a third convolutional layer that convolves 64 filters of 3 x 3 with stride 1 followed by a rectifier.
        The final hidden layer is fully-connected and consists of 512 rectifier units.
        The output layer is a fully-connected linear layer with a single output for each valid action.
    """
    obs = Input(shape=obs_shape)
    # We assume observations are already normalised
    obs2 = Lambda(check_normalised)(obs)
    h = Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(obs2)
    h = Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(h)
    h = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(h)
    h = Flatten()(h)
    h = Dense(units=512, activation='relu')(h)
    qs = Dense(units=n_actions, activation=None)(h)
    model = Model(inputs=obs, outputs=qs)
    return model
