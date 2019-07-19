import unittest
from collections import Counter
from functools import partial

import gym
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import plot, show, grid, xlabel, ylabel
from numpy.testing import assert_raises, assert_approx_equal, assert_allclose

import dqn.train as train
from dqn.model import Model
from dqn.policies import Policy, MLPFeatures
from dqn.replay_buffer import ReplayBuffer, ReplayBatch, PrioritizedReplayBuffer
from dqn.utils import tf_disable_warnings, tf_disable_deprecation_warnings, tensor_index, huber_loss

tf_disable_warnings()
tf_disable_deprecation_warnings()


class Smoke(unittest.TestCase):
    def test(self):
        train.ex.run(config_updates={'train_n_steps': 0})


class Unit(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_tensor_index(self):
        sess = tf.Session()
        params = tf.constant([[1, 2, 3], [4, 5, 6]])
        indices = tf.constant([0, 1])
        result = sess.run(tensor_index(params, indices))
        np.testing.assert_array_equal(result, [1, 5])

    def test_huber_loss(self):
        xs = tf.Variable([0.0, 0.1, 0.5, 1.0, 1.5, 2.0])
        y = huber_loss(xs)
        grads = tf.gradients([y], xs)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        expected_grads = [0.0, 0.1, 0.5, 1.0, 1.0, 1.0]
        actual_grads = sess.run(grads)[0]
        np.testing.assert_array_almost_equal(actual_grads, expected_grads)

    @staticmethod
    def plot_huber_loss():
        sess = tf.Session()
        x_ph = tf.placeholder(tf.float32)
        loss = huber_loss(x_ph, delta=1)

        xs = np.linspace([-5, -4], [5, 4], axis=1)
        losses = sess.run(loss, feed_dict={x_ph: xs})
        plot(xs[0], losses[0])
        plot(xs[1], losses[1])
        grid()
        xlabel("x")
        ylabel("Huber loss")
        show()

    def test_experience_buffer(self):
        buf = ReplayBuffer(obs_shape=(), max_size=2)

        self._store_in_buf(buf, 1)
        self._check_buf_equals(buf, [1])

        self._store_in_buf(buf, 2)
        self._check_buf_equals(buf, [1, 2])

        self._store_in_buf(buf, 3)
        self._check_buf_equals(buf, [2, 3])

    def test_prioritized_replay_buffer(self):
        tds, sample_probs = self._get_sample_probs(beta0=0)
        for td, prob in zip(tds, sample_probs):
            np.testing.assert_approx_equal(prob, td / sum(tds), significant=2)

    @staticmethod
    def _store_in_buf(buf, value):
        buf.store(obs1=value, acts=None, rews=None, obs2=None, done=None)

    @staticmethod
    def _store_in_buf_with_td(buf, value, td):
        buf.store_with_td(obs1=value, acts=None, rews=None, obs2=None, done=None, td=td)

    @staticmethod
    def _check_buf_equals(buf, values):
        samples = {buf.sample(batch_size=1).obs1[0] for _ in range(100)}
        assert samples == set(values)

    @staticmethod
    def _get_sample_probs(beta0):
        buf = PrioritizedReplayBuffer(obs_shape=(), max_size=3, beta0=beta0)
        tds = [1, 2, 3]
        for td in tds:
            Unit._store_in_buf_with_td(buf, td, td)

        samples = []
        n_samples = 10000
        for _ in range(n_samples):
            batch = buf.sample(batch_size=1)
            samples.append(batch.obs1[0])

        counts = Counter(samples)
        tds, probs = [], []
        for td, count in counts.items():
            tds.append(td)
            probs.append(count / n_samples)

        return tds, probs

    def test_model_train(self):
        model = self._get_model()
        (obs1, obs2), act, rew = np.random.rand(2, 1), 0, 1
        batch = ReplayBatch(obs1=[obs1], acts=[act], rews=[rew], obs2=[obs2], done=[True])
        for _ in range(200):
            model.train(batch)
        expected_q = rew
        actual_q = model.q(obs1, act)
        assert_approx_equal(actual_q, expected_q, significant=3)

    def test_model_update_target(self):
        model = self._get_model()
        self._check_main_target_same(model)
        batch = ReplayBatch(obs1=[[1]], acts=[1], rews=[1], obs2=[[1]], done=[True])
        model.train(batch)
        self._check_main_target_different(model)
        model.update_target()
        self._check_main_target_same(model)

    @staticmethod
    def _get_model():
        policy_fn = partial(Policy, features_cls=MLPFeatures, dueling=False)
        model = Model(obs_shape=(1,), n_actions=2, lr=1e-3, seed=0, discount=0.99, double_dqn=False,
                      policy_fn=policy_fn, gradient_clip=10)
        return model

    @staticmethod
    def _check_main_target_different(model):
        obs = [np.random.rand()]
        q1s, q2s = model.sess.run([model.q1s_main, model.q2s_target], feed_dict={model.obs1_ph: [obs],
                                                                                 model.obs2_ph: [obs]})
        assert_raises(AssertionError, assert_allclose, q1s, q2s)
        return obs

    @staticmethod
    def _check_main_target_same(model):
        obs = [np.random.rand()]
        q1s, q2s = model.sess.run([model.q1s_main, model.q2s_target], feed_dict={model.obs1_ph: [obs],
                                                                                 model.obs2_ph: [obs]})
        assert_allclose(q1s, q2s)


class EndToEnd(unittest.TestCase):
    def test_cartpole(self):
        run = train.ex.run(config_updates={'train_n_steps': 20_000,
                                           'lr': 4e-4,
                                           'seed': 0,
                                           'features': 'mlp',
                                           'dueling': False,
                                           'double_dqn': False,
                                           'prioritized': False})
        model = run.result

        env = gym.make('CartPole-v0')
        env.seed(0)
        episode_rewards = self._run_n_test_episodes(env, model, 5)
        self.assertEqual(min(episode_rewards), 200)

    @staticmethod
    def _run_n_test_episodes(env, model, n):
        episode_rewards = []
        for _ in range(n):
            obs, done, rewards = env.reset(), False, []
            while not done:
                obs, reward, done, info = env.step(model.step(obs, random_action_prob=0))
                rewards.append(reward)
            episode_rewards.append(sum(rewards))
        return episode_rewards


if __name__ == '__main__':
    unittest.main()
