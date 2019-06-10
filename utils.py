import os

import tensorflow as tf
from tensorflow.python.util import deprecation


def tf_disable_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tf_disable_deprecation_warnings():
    deprecation._PRINT_DEPRECATION_WARNINGS = False


def tensor_index(params, indices):
    assert len(params.shape) == 2
    assert len(indices.shape) == 1

    indices = tf.expand_dims(indices, axis=1)
    result = tf.batch_gather(params, indices)
    result = result[:, 0]

    return result
