import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from replay_buffer import ReplayBatch
from utils import tensor_index


class Model:

    def __init__(self, obs_shape, n_actions, seed, discount, n_hidden, lr, double_dqn, save_dir=None):
        self.save_args(locals())
        self.n_actions = n_actions
        self.save_dir = save_dir

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.random.set_random_seed(seed)

            obs1_ph = tf.placeholder(tf.float32, (None,) + obs_shape, name='o1')
            obs2_ph = tf.placeholder(tf.float32, (None,) + obs_shape, name='o2')
            acts_ph = tf.placeholder(tf.int32, (None,), name='a1')
            rews_ph = tf.placeholder(tf.float32, (None,), name='r')
            done_ph = tf.placeholder(tf.float32, (None,), name='done')

            h = Dense(n_hidden, 'relu')
            qs = Dense(n_actions, None)
            with tf.variable_scope('main'):
                q1s_main = qs(h(obs1_ph))
                q2s_main = qs(h(obs2_ph))
                assert q1s_main.shape.as_list() == [None, n_actions]
                assert q2s_main.shape.as_list() == [None, n_actions]

            h_target = Dense(n_hidden, 'relu')
            qs_target = Dense(n_actions, None)
            with tf.variable_scope('target'):
                q2s_target = qs_target(h_target(obs2_ph))
                assert q2s_target.shape.as_list() == [None, n_actions]

            params = tf.trainable_variables('main')
            params_target = tf.trainable_variables('target')
            target_update_ops = [param_target.assign(param) for param_target, param in zip(params_target, params)]

            pi = tf.math.argmax(q1s_main, axis=1)
            assert pi.shape.as_list() == [None]

            q1_main = tensor_index(q1s_main, acts_ph)
            assert q1_main.shape.as_list() == [None]
            if double_dqn:
                a = tf.math.argmax(q2s_main, axis=1)
                q_target = tensor_index(q2s_target, a)
            else:
                q_target = tf.reduce_max(q2s_target, axis=1)
            backup = rews_ph + discount * (1 - done_ph) * q_target
            backup = tf.stop_gradient(backup)
            assert backup.shape.as_list() == [None]
            loss = tf.reduce_mean((q1_main - backup) ** 2)
            assert loss.shape.as_list() == []
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(loss)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

        self.obs1_ph = obs1_ph
        self.acts_ph = acts_ph
        self.rews_ph = rews_ph
        self.obs2_ph = obs2_ph
        self.done_ph = done_ph
        self.q1s_main = q1s_main
        self.q2s_target = q2s_target
        self.q1 = q1_main
        self.pi = pi
        self.loss = loss
        self.train_op = train_op
        self.sess = sess
        self.saver = saver
        self.target_update_ops = target_update_ops

    def save_args(self, locals):
        locals = dict(locals)
        self.args = locals
        del self.args['self']
        self.args['save_dir'] = None

    def __getstate__(self):
        with self.graph.as_default():
            vars = tf.trainable_variables()
        names = [v.name for v in vars]
        values = self.sess.run(vars)
        params = {k: v for k, v in zip(names, values)}
        state = self.args, params
        return state

    def __setstate__(self, state):
        args, params = state
        self.__init__(**args)
        with self.graph.as_default():
            ops = []
            for var in tf.trainable_variables():
                assign = var.assign(params[var.name])
                ops.append(assign)
            self.sess.run(ops)

    def step(self, obs, random_action_prob):
        if np.random.rand() < random_action_prob:
            return np.random.choice(self.n_actions)
        else:
            return self.sess.run(self.pi, feed_dict={self.obs1_ph: [obs]})[0]

    def q(self, obs, act):
        return self.sess.run(self.q1,
                             feed_dict={self.obs1_ph: [obs],
                                        self.acts_ph: [act]})[0]

    def update_target(self):
        self.sess.run(self.target_update_ops)

    def train(self, batch: ReplayBatch):
        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.obs1_ph: batch.obs1,
                                           self.acts_ph: batch.acts,
                                           self.rews_ph: batch.rews,
                                           self.obs2_ph: batch.obs2,
                                           self.done_ph: batch.done})
        return loss

    def save(self):
        self.saver.save(self.sess, os.path.join(self.save_dir, 'model'))

    def load(self, load_dir):
        ckpt = tf.train.latest_checkpoint(load_dir)
        self.saver.restore(self.sess, ckpt)
