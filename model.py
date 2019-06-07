import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense


class Model:
    def __init__(self, obs_shape, n_actions, seed=0, epsilon=0.99):
        tf.random.set_random_seed(seed)
        o1_ph = tf.placeholder('float32',
                               (None,) + obs_shape,
                               name='o1')
        o2_ph = tf.placeholder('float32',
                               (None,) + obs_shape,
                               name='o2')
        a1_ph = tf.placeholder('int32',
                               (None,),
                               name='a1')
        r_ph = tf.placeholder(tf.float32,
                              (None,),
                              name='r')
        done_ph = tf.placeholder(tf.bool,
                                 (None,),
                                 name='done')

        h = Dense(32, 'relu')
        qs = Dense(n_actions, None)
        q1s = qs(h(o1_ph))
        q2s = qs(h(o2_ph))
        pi = tf.math.argmax(q1s, axis=1)
        # TODO there must be a better way of doing this
        q1 = tf.reduce_sum(q1s * tf.one_hot(a1_ph, depth=n_actions), axis=1)
        done_float = tf.cast(done_ph, tf.float32)
        bellman_backup = r_ph + (1 - done_float) * 0.99 * tf.reduce_max(q2s, axis=1)
        loss = tf.reduce_mean((q1 - bellman_backup) ** 2)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.o1_ph = o1_ph
        self.o2_ph = o2_ph
        self.a1_ph = a1_ph
        self.pi = pi
        self.r_ph = r_ph
        self.q1 = q1
        self.done_ph = done_ph
        self.sess = sess
        self.train_op = train_op
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.loss = loss

    def step(self, obs, deterministic):
        a = self.sess.run(self.pi, feed_dict={self.o1_ph: [obs]})[0]
        if not deterministic and (np.random.rand() < (1 - self.epsilon)):
            return np.random.choice(self.n_actions)
        return a

    def q(self, obs, act):
        return self.sess.run(self.q1, feed_dict=({self.o1_ph: [obs], self.a1_ph: [act]}))

    def train(self, o1s, acts, rs, o2s, dones):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.o1_ph: o1s,
                                                                       self.a1_ph: acts,
                                                                       self.r_ph: rs,
                                                                       self.o2_ph: o2s,
                                                                       self.done_ph: dones})
        return loss
