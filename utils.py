import os

from tensorflow.python.util import deprecation


def tf_disable_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tf_disable_deprecation_warnings():
    deprecation._PRINT_DEPRECATION_WARNINGS = False
