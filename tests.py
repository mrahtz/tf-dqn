import unittest

import gym
import numpy as np
import tensorflow as tf
from numpy.testing import assert_raises, assert_approx_equal, assert_allclose

import train
from model import Model
from replay_buffer import ReplayBuffer, ReplayBatch
from utils import tf_disable_warnings, tf_disable_deprecation_warnings, tensor_index

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

        q1s, q2s = model.sess.run([model.q1s_main, model.q2s_target], feed_dict={model.obs1_ph: [obs],
                                                                                 model.obs2_ph: [obs]})
        assert_raises(AssertionError, assert_allclose, q1s, q2s)

        model.update_target()
        q1s, q2s = model.sess.run([model.q1s_main, model.q2s_target], feed_dict={model.obs1_ph: [obs],
                                                                                 model.obs2_ph: [obs]})
        assert_allclose(q1s, q2s)

    def test_tensor_index(self):
        sess = tf.Session()
        params = tf.constant([[1, 2, 3], [4, 5, 6]])
        indices = tf.constant([0, 1])
        result = sess.run(tensor_index(params, indices))
        np.testing.assert_array_equal(result, [1, 5])


class EndToEnd(unittest.TestCase):
    def test_cartpole(self):
        r = train.ex.run(config_updates={'train_n_steps': 20000, 'seed': 0})
        model = r.result

        env = gym.make('CartPole-v0')
        env.seed(0)
        episode_rewards = []
        for _ in range(5):
            obs, done = env.reset(), False
            rewards = []
            while not done:
                obs, reward, done, info = env.step(model.step(obs, random_action_prob=0))
                rewards.append(reward)
            episode_rewards.append(sum(rewards))
        self.assertEqual(min(episode_rewards), 200)


if __name__ == '__main__':
    unittest.main()
