import unittest

import numpy as np
from numpy.testing import assert_raises, assert_approx_equal, assert_allclose

from model import Model
from replay_buffer import ReplayBuffer, ReplayBatch
from utils import tf_disable_warnings, tf_disable_deprecation_warnings

tf_disable_warnings()
tf_disable_deprecation_warnings()


class Tests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_experience_buffer(self):
        buf = ReplayBuffer(obs_shape=(), max_size=2)

        self.store_in_buf(buf, 1)
        self.check_buf_equals(buf, [1])

        self.store_in_buf(buf, 2)
        self.check_buf_equals(buf, [1, 2])

        self.store_in_buf(buf, 3)
        self.check_buf_equals(buf, [2, 3])

    def check_buf_equals(self, buf, values):
        samples = {buf.sample(batch_size=1).obs1[0] for _ in range(100)}
        self.assertEqual(samples, set(values))

    def store_in_buf(self, buf, value):
        buf.store(obs1=value, acts=None, rews=None, obs2=None, done=None)

    def test_model_train(self):
        model = Model(obs_shape=(1,), n_actions=2, lr=1e-3, n_hidden=64, seed=0, discount=0.99)
        obs1 = [np.random.rand()]
        obs2 = [np.random.rand()]
        act = 0
        rew = 1
        batch = ReplayBatch(obs1=[obs1], acts=[act], rews=[rew], obs2=[obs2], done=[True])
        for _ in range(200):
            model.train(batch)
        assert_approx_equal(model.q(obs1, act), rew, significant=3)

    def test_model_update_target(self):
        model = Model(obs_shape=(1,), n_actions=2, n_hidden=3, seed=0, discount=0.99, lr=1e-3)
        obs = [np.random.rand()]

        q1s, q2s = model.sess.run([model.q1s, model.q2s], feed_dict={model.obs1_ph: [obs],
                                                                     model.obs2_ph: [obs]})
        assert_raises(AssertionError, assert_allclose, q1s, q2s)

        model.update_target()
        q1s, q2s = model.sess.run([model.q1s, model.q2s], feed_dict={model.obs1_ph: [obs],
                                                                     model.obs2_ph: [obs]})
        assert_allclose(q1s, q2s)


if __name__ == '__main__':
    unittest.main()
