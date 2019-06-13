import unittest
from functools import partial

import gym
import numpy as np
import tensorflow as tf
from numpy.testing import assert_raises, assert_approx_equal, assert_allclose

import train
from model import Model
from policies import mlp_features, make_policy
from replay_buffer import ReplayBuffer, ReplayBatch
from utils import tf_disable_warnings, tf_disable_deprecation_warnings, tensor_index

tf_disable_warnings()
tf_disable_deprecation_warnings()


class UnitTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_experience_buffer(self):
        buf = ReplayBuffer(obs_shape=(), max_size=2)

        store_in_buf(buf, 1)
        check_buf_equals(buf, [1])

        store_in_buf(buf, 2)
        check_buf_equals(buf, [1, 2])

        store_in_buf(buf, 3)
        check_buf_equals(buf, [2, 3])

    def test_model_train(self):
        model = get_model()
        (obs1, obs2), act, rew = np.random.rand(2, 1), 0, 1
        batch = ReplayBatch(obs1=[obs1], acts=[act], rews=[rew], obs2=[obs2], done=[True])
        for _ in range(200):
            model.train(batch)
        expected_q = rew
        actual_q = model.q(obs1, act)
        assert_approx_equal(actual_q, expected_q, significant=3)

    def test_model_update_target(self):
        model = get_model()
        check_main_target_same(model)
        batch = ReplayBatch(obs1=[[1]], acts=[1], rews=[1], obs2=[[1]], done=[True])
        model.train(batch)
        check_main_target_different(model)
        model.update_target()
        check_main_target_same(model)

    def test_tensor_index(self):
        sess = tf.Session()
        params = tf.constant([[1, 2, 3], [4, 5, 6]])
        indices = tf.constant([0, 1])
        result = sess.run(tensor_index(params, indices))
        np.testing.assert_array_equal(result, [1, 5])


def store_in_buf(buf, value):
    buf.store(obs1=value, acts=None, rews=None, obs2=None, done=None)


def check_buf_equals(buf, values):
    samples = {buf.sample(batch_size=1).obs1[0] for _ in range(100)}
    assert samples == set(values)


def get_model():
    policy_fn = partial(make_policy, feature_extractor=mlp_features, dueling=False)
    model = Model(obs_shape=(1,), n_actions=2, lr=1e-3, seed=0, discount=0.99, double_dqn=False, policy_fn=policy_fn)
    return model


def check_main_target_different(model):
    obs = [np.random.rand()]
    q1s, q2s = model.sess.run([model.q1s_main, model.q2s_target], feed_dict={model.obs1_ph: [obs],
                                                                             model.obs2_ph: [obs]})
    assert_raises(AssertionError, assert_allclose, q1s, q2s)
    return obs


def check_main_target_same(model):
    obs = [np.random.rand()]
    q1s, q2s = model.sess.run([model.q1s_main, model.q2s_target], feed_dict={model.obs1_ph: [obs],
                                                                             model.obs2_ph: [obs]})
    assert_allclose(q1s, q2s)


class EndToEnd(unittest.TestCase):
    def test_cartpole(self):
        feature_extractor = partial(mlp_features, n_hidden=[64])
        r = train.ex.run(config_updates={'train_n_steps': 30_000, 'seed': 0,
                                         'feature_extractor': feature_extractor, 'dueling': False, 'double_dqn': False})
        model = r.result

        env = gym.make('CartPole-v0')
        env.seed(0)
        episode_rewards = run_test_episodes(env, model)
        self.assertEqual(min(episode_rewards), 200)


def run_test_episodes(env, model):
    episode_rewards = []
    for _ in range(5):
        obs, done, rewards = env.reset(), False, []
        while not done:
            obs, reward, done, info = env.step(model.step(obs, random_action_prob=0))
            rewards.append(reward)
        episode_rewards.append(sum(rewards))
    return episode_rewards


if __name__ == '__main__':
    unittest.main()
