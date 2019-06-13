import os
import time

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


class RateMeasure:
    def __init__(self, val):
        self.prev_t = self.prev_value = None
        self.reset(val)

    def reset(self, val):
        self.prev_value = val
        self.prev_t = time.time()

    def measure(self, val):
        val_change = val - self.prev_value
        cur_t = time.time()
        interval = cur_t - self.prev_t
        rate = val_change / interval

        self.prev_t = cur_t
        self.prev_value = val

        return rate
