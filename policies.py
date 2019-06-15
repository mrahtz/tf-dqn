import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Lambda, Flatten


def make_policy(obs_shape, n_actions, feature_extractor, dueling):
    obs = Input(shape=obs_shape)
    features = feature_extractor(obs)
    if dueling:
        v = Dense(1, activation=None)(features)
        advs = Dense(n_actions, activation=None)(features)
        qs = Lambda(compute_dueling_q)([v, advs])
    else:
        qs = Dense(units=n_actions, activation=None)(features)
    model = Model(inputs=obs, outputs=qs)
    return model


def mlp_features(obs, n_hidden):
    h = obs
    for n in n_hidden:
        h = Dense(n, activation='relu')(h)
    return h


def cnn_features(obs):
    """
    'Human-level control through deep reinforcement learning':
        The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the input image and applies a rectifier nonlinearity.
        The second hidden layer convolves 64 filters of 4 x 4 with stride 2, again followed by a rectifier nonlinearity.
        This is followed by a third convolutional layer that convolves 64 filters of 3 x 3 with stride 1 followed by a rectifier.
        The final hidden layer is fully-connected and consists of 512 rectifier units.
        The output layer is a fully-connected linear layer with a single output for each valid action.
    """
    # We assume observations are already normalised
    obs2 = Lambda(check_normalised)(obs)
    h = Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(obs2)
    h = Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(h)
    h = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(h)
    h = Flatten()(h)
    h = Dense(units=512, activation='relu')(h)
    return h


def compute_dueling_q(val_advs_tuple):
    v, advs = val_advs_tuple
    n_actions = advs.shape.as_list()[1]
    assert v.shape.as_list() == [None, 1]
    assert advs.shape.as_list() == [None, n_actions]

    mean_adv = tf.reduce_mean(advs, axis=1, keepdims=True)
    centered_advs = advs - mean_adv
    qs = v + centered_advs

    assert mean_adv.shape.as_list() == [None, 1]
    assert centered_advs.shape.as_list() == [None, n_actions]
    assert qs.shape.as_list() == [None, n_actions]

    return qs


def check_normalised(obs):
    assert obs.shape.as_list() == [None, 84, 84, 4]
    asserts = [tf.assert_greater_equal(obs, 0.0), tf.assert_less_equal(obs, 1.0)]
    with tf.control_dependencies(asserts):
        obs = tf.identity(obs)
    return obs
