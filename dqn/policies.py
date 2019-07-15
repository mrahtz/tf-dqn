import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten


class CNNFeatures(Sequential):
    def __init__(self):
        """
        'Human-level control through deep reinforcement learning':
            The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the input image and applies a rectifier nonlinearity.
            The second hidden layer convolves 64 filters of 4 x 4 with stride 2, again followed by a rectifier nonlinearity.
            This is followed by a third convolutional layer that convolves 64 filters of 3 x 3 with stride 1 followed by a rectifier.
            The final hidden layer is fully-connected and consists of 512 rectifier units.
            The output layer is a fully-connected linear layer with a single output for each valid action.
        """
        layers = [
            CheckCNNObsNormalized,
            Conv2D(filters=32, kernel_size=8, strides=4, activation='relu'),
            Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'),
            Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
            Flatten(),
            Dense(units=512, activation='relu')
        ]
        super().__init__(layers)


class CheckCNNObsNormalized(Layer):
    def call(self, obs, **kwargs):
        assert obs.shape.as_list() == [None, 84, 84, 4]
        asserts = [tf.assert_greater_equal(obs, 0.0), tf.assert_less_equal(obs, 1.0)]
        with tf.control_dependencies(asserts):
            obs = tf.identity(obs)
        return obs


class MLPFeatures(Sequential):
    def __init__(self):
        layers = [
            Dense(64, 'relu'),
            Dense(64, 'relu'),
        ]
        super().__init__(layers)


class Policy(Sequential):
    def __init__(self, n_actions, features_cls, dueling):
        features_model = features_cls()
        if dueling:
            layers = [
                features_model,
                DuelingQs(n_actions)
            ]
        else:
            layers = [
                features_model,
                Dense(n_actions, activation=None)
            ]
        super().__init__(layers)


class DuelingQs(Layer):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.v_layer = Dense(1, activation=None)
        self.advs_layer = Dense(n_actions, activation=None)

    def call(self, features, **kwargs):
        v = self.v_layer(features)
        advs = self.advs_layer(features)

        assert v.shape.as_list() == [None, 1]
        assert advs.shape.as_list() == [None, self.n_actions]

        mean_adv = tf.reduce_mean(advs, axis=1, keepdims=True)
        centered_advs = advs - mean_adv
        qs = v + centered_advs

        assert mean_adv.shape.as_list() == [None, 1]
        assert centered_advs.shape.as_list() == [None, self.n_actions]
        assert qs.shape.as_list() == [None, self.n_actions]

        return qs
