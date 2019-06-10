import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense


class Model:
    def __init__(self, obs_shape, n_actions, seed=0, discount=0.99, n_hidden=64, lr=1e-5):
        self.n_actions = n_actions

        tf.random.set_random_seed(seed)
        self.o1_ph = tf.placeholder(tf.float32, (None,) + obs_shape, name='o1')
        self.o2_ph = tf.placeholder(tf.float32, (None,) + obs_shape, name='o2')
        self.a1_ph = tf.placeholder(tf.int32, (None,), name='a1')
        self.r_ph = tf.placeholder(tf.float32, (None,), name='r')
        self.done_ph = tf.placeholder(tf.float32, (None,), name='done')

        h = Dense(n_hidden, 'relu')
        qs = Dense(n_actions, None)
        with tf.variable_scope('q1s'):
            self.q1s = qs(h(self.o1_ph))
            assert self.q1s.shape.as_list() == [None, n_actions]

        h_target = Dense(n_hidden, 'relu')
        qs_target = Dense(n_actions, None)
        with tf.variable_scope('q2s'):
            self.q2s = qs_target(h_target(self.o2_ph))
            assert self.q2s.shape.as_list() == [None, n_actions]

        params_target = tf.trainable_variables('q2s')
        params = tf.trainable_variables('q1s')
        self.target_update_ops = [param_target.assign(param)
                                  for param_target, param in zip(params_target, params)]

        self.pi = tf.math.argmax(self.q1s, axis=1)
        assert self.pi.shape.as_list() == [None]

        self.q1 = tf.reduce_sum(self.q1s * tf.one_hot(self.a1_ph, depth=n_actions), axis=1)
        assert self.q1.shape.as_list() == [None]
        self.backup = self.r_ph + discount * (1 - self.done_ph) * tf.reduce_max(self.q2s, axis=1)
        self.backup = tf.stop_gradient(self.backup)
        assert self.backup.shape.as_list() == [None]
        self.loss = tf.reduce_mean((self.q1 - self.backup) ** 2)
        assert self.loss.shape.as_list() == []
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs, random_action_prob):
        if np.random.rand() < random_action_prob:
            return np.random.choice(self.n_actions)
        else:
            return self.sess.run(self.pi, feed_dict={self.o1_ph: [obs]})[0]

    def update_target(self):
        self.sess.run(self.target_update_ops)

    def q(self, obs, act):
        return self.sess.run(self.q1, feed_dict=({self.o1_ph: [obs], self.a1_ph: [act]}))

    def train(self, o1s, acts, rs, o2s, dones):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.o1_ph: o1s,
                                                                       self.a1_ph: acts,
                                                                       self.r_ph: rs,
                                                                       self.o2_ph: o2s,
                                                                       self.done_ph: dones})
        return loss
