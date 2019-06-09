import unittest

import numpy as np
from numpy.testing import assert_raises, assert_approx_equal, assert_allclose

from model import Model
from train import ExperienceBuffer


class Tests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_experience_buffer(self):
        buf = ExperienceBuffer(obs_shape=(), max_size=2)

        buf.store(o1=1, a=None, r=None, o2=None, done=None)
        samples = {buf.sample(batch_size=1).o1s[0] for _ in range(100)}
        self.assertEqual(samples, {1})

        buf.store(o1=2, a=None, r=None, o2=None, done=None)
        samples = {buf.sample(batch_size=1).o1s[0] for _ in range(100)}
        self.assertEqual(samples, {1, 2})

        buf.store(o1=3, a=None, r=None, o2=None, done=None)
        samples = {buf.sample(batch_size=1).o1s[0] for _ in range(100)}
        self.assertEqual(samples, {2, 3})

    def test_model_train(self):
        model = Model(obs_shape=(1,), n_actions=2)
        obs1 = [np.random.rand()]
        obs2 = [np.random.rand()]
        act = 0
        r = 1
        for _ in range(200):
            model.train(o1s=[obs1], acts=[act], rs=[r], o2s=[obs2], dones=[True])
        assert_approx_equal(model.q(obs1, act), r, significant=3)

    def test_model_update_target(self):
        model = Model(obs_shape=(1,), n_actions=2, n_hidden=3)
        obs = [np.random.rand()]

        q1s, q2s = model.sess.run([model.q1s, model.q2s], feed_dict={model.o1_ph: [obs],
                                                                     model.o2_ph: [obs]})
        assert_raises(AssertionError, assert_allclose, q1s, q2s)

        model.update_target()
        q1s, q2s = model.sess.run([model.q1s, model.q2s], feed_dict={model.o1_ph: [obs],
                                                                     model.o2_ph: [obs]})
        assert_allclose(q1s, q2s)


if __name__ == '__main__':
    unittest.main()
